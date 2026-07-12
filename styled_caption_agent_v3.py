"""
styled_caption_agent_v4.py

Video caption agent - four models, chained, calibrated for accuracy.

Chain per video (5 calls, videos run in parallel):
    1. Gemini 3.5 Flash  (Google AI Studio) - watch video, scene summary
    2. Qwen3-VL-235B     (OpenRouter)       - validate Gemini vs frames
    3. Claude Sonnet 4.6 (Anthropic)        - merge + write 4 styled captions
    4. Gemma 3 27B       (Google AI Studio) - judge vs style checklist
    5. Claude Sonnet 4.6 (Anthropic)        - rewrite ONLY if Gemma flags

Calibrated against the official Track 2 guidance:
    - Accurate captions, correct style, specific video details,
      NO major hallucinations, complete outputs.
    - Register matters: formal = 1-2 clear sentences; sarcastic /
      humorous_tech / humorous_non_tech = ONE punchy line each.
    - Safe specifics (colors, objects, actions, setting) are wanted.
      Risky specifics (guessed place names, quoted on-screen text,
      technical metadata, material guesses) lose points and are banned.

Contract:
    Reads  /input/tasks.json
    Writes /output/results.json   (atomic, after every task)
    Exit 0 on success.

Install:
    pip install -U google-genai opencv-python requests scenedetect

Env keys:
    GEMINI_API_KEY      (Gemini + Gemma, both Google AI Studio)
    OPENROUTER_API_KEY  (Qwen3-VL)
    ANTHROPIC_API_KEY   (Claude)
"""

import os
import re
import cv2
import json
import sys
import time
import base64
import shutil
import tempfile
import subprocess
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from google import genai


# ============================================================
# CONFIG
# ============================================================

TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")

# Keys: paste here for local runs, or leave "" to use env vars.
# Leave EMPTY in the submitted image if the harness injects env vars.
GEMINI_API_KEY = ""
OPENROUTER_API_KEY = ""
ANTHROPIC_API_KEY = ""

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"
GEMMA_MODEL = "gemma-4-31b-it"          # same Google AI Studio key

QWEN_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

SUPPORTED_STYLES = ["formal", "sarcastic", "humorous_tech",
                    "humorous_non_tech"]

PARALLEL_VIDEOS = 8
MAX_FRAMES = 5
FRAME_MAX_SIDE = 1024

ENABLE_GEMMA_JUDGE = True     # stage 4 + conditional rewrite
API_RETRIES = 2
API_RETRY_DELAY = 3
HARD_TIMEOUT = int(os.environ.get("HARD_TIMEOUT", 540))   # 9 min

LAST_RESORT_CAPTION = ("A short video clip (automatic captioning was "
                       "unavailable for this item).")


# ============================================================
# STYLE DEFINITIONS  (register-calibrated)
# ============================================================

STYLE_GUIDES = {
    "formal": (
        "Professional, objective, factual. One or two clear sentences of "
        "polished description - what a museum label or news-agency "
        "alt-text would say. Names the subject with its obvious visual "
        "attributes, the action, and the setting. No jokes, no "
        "editorialising, no technical metadata."),
    "sarcastic": (
        "Dry, ironic, lightly mocking - but STILL ACCURATE about what is "
        "in the video. ONE short sentence. Deadpan understatement or mock "
        "grandeur about what is actually happening. The irony targets the "
        "situation, never a real person's appearance. Witty, never crude."),
    "humorous_tech": (
        "Funny, using a technology or programming metaphor (deployment, "
        "rollback, latency, buffering, merge conflict, stack trace...) "
        "mapped onto something REAL in the video. ONE or two short lines. "
        "The tech joke must reference actual video content - the humour "
        "comes from the metaphor, the accuracy comes from the content."),
    "humorous_non_tech": (
        "Funny, everyday, relatable humour with ZERO technical jargon - "
        "no code, servers, deployments, frame rates or software words of "
        "any kind. ONE short sentence of observational comedy about what "
        "is actually visible."),
}

# Generic register exemplars (NOT tied to any evaluation clip - these
# teach LENGTH and TONE only, and must never be copied).
REGISTER_EXEMPLARS = """
Register to match (these are about unrelated generic scenes - copy the
LENGTH and TONE, never the wording or the content):

formal: "A young orange tabby cat sits among dense green foliage,
looking directly at the camera with an alert expression."
  -> one or two calm, factual sentences. Specific but not exhaustive.

sarcastic: "A cat outdoors, clearly plotting something elaborate and
fully confident it will succeed."
  -> ONE line. Dry. Still tells you what is in the video.

humorous_tech: "A small autonomous agent has entered the garden and is
scanning for input. Rollback plan: none."
  -> ONE or two short lines. Tech metaphor wrapped around the real subject.

humorous_non_tech: "A tiny cat has gone outside and is now judging
everything it sees with great authority."
  -> ONE line. Everyday humour. No technical words at all.
"""

# The rules that protect ACCURACY. Shared by the writer and the judge.
ACCURACY_RULES = """
ACCURACY RULES (these decide the score):
1. Every concrete claim must come from the FACT SHEET. Never invent
   objects, animals, people, actions, text or sounds.
2. ALWAYS name the main subject with its most obvious visual attribute
   ("a tan dog", "an orange kitten", "a red athletic track") - never a
   bare noun like "a dog". Also state what it is doing and where.
3. Include a FEW concrete visual specifics (colour, object type, action,
   setting, weather, camera movement). These are wanted.
4. BANNED - these lose points and must never appear:
   - Place, city, country, landmark, brand or company names GUESSED from
     appearance (skylines, bridges, buildings, logos). Only allowed if
     the FACT SHEET marks the name as literally read from on-screen text
     AND confirmed.
   - Quoted on-screen text, sign text, numbers or dates unless the FACT
     SHEET explicitly marks them CONFIRMED. When in doubt, omit.
   - Material or species guesses stated as fact ("granite", "ginkgo",
     "Labrador") unless the FACT SHEET states them.
   - Technical metadata: resolution, frame rate, fps, file duration,
     "N-second clip", codec, bitrate. Never in any caption.
   - Audio/silence meta-commentary: do NOT mention "no audio track",
     "silent mode", "no audio stream", or that the video is silent in
     ANY style. Audio status is a file property, not video content.
   - Precise counts of things that are hard to count. Say "several".
5. Prefer omitting an uncertain detail over risking a wrong one. A short
   correct caption beats a long caption with one error.
"""


# ============================================================
# UTILITIES
# ============================================================

NO_RETRY_CODES = ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404")


def with_retries(label, fn):
    delay = API_RETRY_DELAY
    for attempt in range(1, API_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if any(code in str(exc) for code in NO_RETRY_CODES):
                raise
            if attempt == API_RETRIES:
                raise
            print(f"[retry] {label}: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def _resolve_key(hardcoded, env_name):
    value = (hardcoded or "").strip()
    if not value or set(value.lower()) <= {"x"}:
        value = os.environ.get(env_name, "").strip()
    if not value or set(value.lower()) <= {"x"}:
        raise ValueError(f"{env_name} is not set.")
    return value


def get_google_key():
    return _resolve_key(GEMINI_API_KEY, "GEMINI_API_KEY")


def get_openrouter_key():
    return _resolve_key(OPENROUTER_API_KEY, "OPENROUTER_API_KEY")


def get_anthropic_key():
    return _resolve_key(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")


THINK_TAG = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>",
                       re.DOTALL | re.IGNORECASE)


def clean_text(text):
    return THINK_TAG.sub("", text or "").strip()


def parse_chat_response(data):
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("[warn] model output truncated at max_tokens")
    msg = choice.get("message", {}) or {}
    for field in ("content", "reasoning_content", "reasoning"):
        text = clean_text(msg.get(field) or "")
        if text:
            return text
    raise RuntimeError("empty model response")


def extract_json_object(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in model output")


def write_results_atomic(results):
    path = Path(RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


# ============================================================
# MODEL CLIENTS
# ============================================================

def image_to_data_url(path):
    with open(path, "rb") as f:
        return ("data:image/jpeg;base64,"
                + base64.b64encode(f.read()).decode("utf-8"))


def call_qwen(prompt, image_paths, max_tokens=4000):
    key = get_openrouter_key()
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(p)}})
    payload = {"model": QWEN_MODEL, "max_tokens": max_tokens,
               "temperature": 0.3, "top_p": 0.9,
               "messages": [{"role": "user", "content": content}]}
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}

    def post():
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload,
                          timeout=240)
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        return parse_chat_response(r.json())

    return with_retries("Qwen", post)


def call_claude(prompt, max_tokens=2000, temperature=0.7):
    key = get_anthropic_key()
    payload = {"model": CLAUDE_MODEL, "max_tokens": max_tokens,
               "temperature": temperature,
               "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}

    def post():
        r = requests.post(ANTHROPIC_URL, headers=headers, json=payload,
                          timeout=240)
        if r.status_code != 200:
            raise RuntimeError(f"Anthropic HTTP {r.status_code}: "
                               f"{r.text[:300]}")
        blocks = r.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks
                       if b.get("type") == "text").strip()
        if not text:
            raise RuntimeError("empty Claude response")
        return text

    return with_retries("Claude", post)


def call_gemma(client, prompt):
    """Gemma 4 31B via Google AI Studio. NO retries - if it 500s, skip."""
    resp = client.models.generate_content(model=GEMMA_MODEL,
                                          contents=[prompt])
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty Gemma response")
    return text


# ============================================================
# VIDEO + FRAMES
# ============================================================

def obtain_video(source, work_dir):
    if not str(source).lower().startswith(("http://", "https://")):
        if not Path(source).is_file():
            raise FileNotFoundError(f"no such video: {source}")
        return str(source)
    path = os.path.join(work_dir, "input_video.mp4")
    r = requests.get(source, stream=True, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path


def probe_audio(video_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "UNKNOWN (ffprobe unavailable)"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=90)
        codecs = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        if codecs:
            return f"AUDIO PRESENT (codec: {', '.join(codecs)})"
        return "NO AUDIO TRACK - the video is silent"
    except Exception:
        return "UNKNOWN (audio probe failed)"


def get_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": round(fps, 2) if fps else 0,
            "frames": n,
            "duration": round(n / fps, 2) if fps else 0,
            "audio": probe_audio(video_path)}


def detect_scenes(video_path):
    try:
        from scenedetect import detect, ContentDetector
        return [(s.get_seconds(), e.get_seconds())
                for s, e in detect(str(video_path), ContentDetector())]
    except Exception:
        return []


def pick_timestamps(video_path, duration, fps, n=MAX_FRAMES):
    scenes = detect_scenes(video_path)
    if len(scenes) >= 2:
        mids = sorted({round(s + (e - s) / 2.0, 2) for s, e in scenes})
        if len(mids) > n:
            step = len(mids) / n
            mids = [mids[int(i * step)] for i in range(n)]
        return mids
    last = max(duration - (1.0 / fps if fps else 0.05), 0.0)
    if n <= 1 or last <= 0:
        return [0.0]
    return [round(i * last / (n - 1), 2) for i in range(n)]


def extract_frames(video_path, timestamps, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video for frames")
    paths = []
    for i, t in enumerate(timestamps, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = FRAME_MAX_SIDE / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        p = Path(out_dir) / f"frame_{i}.jpg"
        cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(str(p))
    cap.release()
    if not paths:
        raise RuntimeError("no frames extracted")
    return paths


# ============================================================
# STAGE 1 - GEMINI: WATCH AND SUMMARIZE
# ============================================================

GEMINI_PROMPT = """
You are a careful video analyst. Watch this video (visuals and audio) and
describe what is actually there, at SCENE level.

VERIFIED TECHNICAL FACT (measured from the file - never contradict it):
Audio: {audio}

Write a factual description a captioner can rely on. Be specific about
what is plainly visible, and honest about what is not.

RULES:
- Name the main subject with its obvious visual attributes (colour, type,
  clothing, size), what it does, and where.
- Do NOT guess place names, city names, landmarks, brands, species or
  materials from appearance. If you are not certain, describe it plainly
  ("a large building", "a leafy tree", "a light-coloured rock face").
- Report on-screen text ONLY if you can actually read it. Mark each piece
  of text as CONFIRMED (clearly legible) or UNSURE.
- Never mention resolution, frame rate or file duration.
- If something is unclear, say "unclear" rather than inventing it.

Return exactly this structure:

MAIN_SUBJECT:
The main subject with its key visual attributes.

ACTION:
What the subject does over the clip.

SETTING:
Where this takes place (describe it; do not name it).

VISUAL_DETAILS:
- concrete visible detail
- concrete visible detail
- concrete visible detail

CAMERA_AND_LIGHT:
Camera angle/movement, lighting, mood.

ON_SCREEN_TEXT:
- text (CONFIRMED) or text (UNSURE), or "none"

AUDIO:
What is heard, consistent with the verified audio fact above.

UNCERTAIN:
- anything you are NOT confident about
"""


def gemini_summarize(client, video_path, audio_status):
    uploaded = None
    try:
        uploaded = client.files.upload(file=video_path)
        while getattr(uploaded.state, "name", str(uploaded.state)) \
                == "PROCESSING":
            time.sleep(4)
            uploaded = client.files.get(name=uploaded.name)
        if getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError("Gemini failed to process the video")

        prompt = GEMINI_PROMPT.format(audio=audio_status)

        def go():
            cfg = None
            try:
                from google.genai import types
                enum = getattr(types, "MediaResolution", None)
                val = (getattr(enum, GEMINI_MEDIA_RESOLUTION)
                       if enum and hasattr(enum, GEMINI_MEDIA_RESOLUTION)
                       else GEMINI_MEDIA_RESOLUTION)
                cfg = types.GenerateContentConfig(media_resolution=val)
            except Exception:
                cfg = None
            try:
                if cfg is not None:
                    return client.models.generate_content(
                        model=GEMINI_MODEL, contents=[uploaded, prompt],
                        config=cfg)
            except Exception as exc:
                print(f"[warn] media_resolution rejected ({exc})")
            return client.models.generate_content(
                model=GEMINI_MODEL, contents=[uploaded, prompt])

        resp = with_retries("Gemini", go)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("empty Gemini response")
        return text
    finally:
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ============================================================
# STAGE 2 - QWEN: VALIDATE (corrections only, no detail hunting)
# ============================================================

QWEN_PROMPT = """
You are a fact-checker. Below is another model's description of a video.
You are shown {n} still frames sampled from that same video.

Your ONLY job: check the description against the frames.

DESCRIPTION TO CHECK:
{summary}

Decide three things:
1. WRONG - statements in the description that the frames contradict
   (wrong colour, wrong object, wrong action, wrong setting, an object
   that is simply not there).
2. MISSING - only OBVIOUS, scene-level things a viewer would notice
   immediately and the description failed to mention (a major subject, a
   large object, the general setting). Do NOT hunt for small details, do
   not read small text, do not add micro-observations.
3. RISKY - statements that look like guesses rather than observations:
   named places/landmarks/brands, species or material names, quoted text
   you cannot clearly read in the frames.

Be conservative. If the frames do not clearly contradict something, it is
NOT wrong - still frames cannot show motion, sound, or events between
frames, so never call something wrong just because you cannot see it here.
Do not narrate your reasoning. Start your reply directly with "WRONG:".

Return exactly:

WRONG:
- statement -> what the frames actually show
(or "none")

MISSING:
- obvious scene-level thing the description omitted
(or "none")

RISKY:
- statement that appears guessed rather than observed
(or "none")
"""


def qwen_validate(frame_paths, gemini_summary):
    prompt = QWEN_PROMPT.format(n=len(frame_paths),
                                summary=gemini_summary)
    return call_qwen(prompt, frame_paths, max_tokens=4000)


# ============================================================
# STAGE 3 - CLAUDE: MERGE + WRITE THE FOUR CAPTIONS
# ============================================================

def build_write_prompt(gemini_summary, qwen_report, audio_status, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)
    return f"""
You are an expert caption writer. You get two independent sources about
ONE video. First reconcile them into a fact sheet, then write captions.

SOURCE A - full-video analysis (saw video and audio):
{gemini_summary}

SOURCE B - frame fact-check (saw still frames only, did not see audio or
motion between frames):
{qwen_report}

VERIFIED TECHNICAL FACT: Audio: {audio_status}

HOW TO RECONCILE:
- If B says a statement in A is WRONG, trust B for static visual facts
  (colour, object identity, setting).
- If B lists something MISSING, add it.
- If B lists something RISKY, DROP it from the fact sheet entirely -
  guessed names, unreadable text and species/material guesses must not
  reach the captions.
- B saw only still frames: never let B override A about motion, actions,
  events over time, or audio.
- Anything neither source is confident about is simply left out.

{ACCURACY_RULES}

Now write ONE caption per requested style.

STYLES:
{style_lines}

{REGISTER_EXEMPLARS}

LENGTH (this is scored - exceeding these WILL lower your score):
- formal: EXACTLY one or two sentences. Not three. Combine details
  into flowing clauses rather than adding sentences.
- sarcastic: ONE sentence only. Must still name the subject.
- humorous_tech: ONE or two SHORT lines. Must still name the subject.
- humorous_non_tech: ONE sentence only. Must still name the subject.
  Must contain ZERO technical words (no code, servers, deployment,
  render, pipeline, threads, nodes, fps, audio track, etc).

Every caption, in every style, must make clear WHAT the video shows -
the humour wraps around the real subject, action and setting, it never
replaces them.

OUTPUT: reply with ONLY a raw JSON object (no markdown, no fences, no
commentary) with exactly these keys: {keys}
Each value is the caption string.
"""


def validate_captions(obj, styles):
    if not isinstance(obj, dict):
        raise ValueError("captions payload is not an object")
    out = {}
    for s in styles:
        v = obj.get(s)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"missing caption for '{s}'")
        out[s] = " ".join(v.split())
    return out


def claude_write(gemini_summary, qwen_report, audio_status, styles):
    prompt = build_write_prompt(gemini_summary, qwen_report,
                                audio_status, styles)
    for attempt in (1, 2):
        raw = call_claude(prompt, max_tokens=1500, temperature=0.8)
        try:
            return validate_captions(extract_json_object(raw), styles)
        except Exception as exc:
            print(f"[warn] caption JSON invalid (try {attempt}): {exc}")
            prompt += ("\n\nREMINDER: output ONLY the raw JSON object "
                       "with every requested style key.")
    raise RuntimeError("caption generation returned invalid JSON twice")


# ============================================================
# STAGE 4 - GEMMA: JUDGE AGAINST THE OFFICIAL CRITERIA
# ============================================================

def build_judge_prompt(gemini_summary, qwen_report, captions):
    return f"""
You are a strict caption judge for a video captioning competition. Score
captions exactly the way the official evaluation judge would, using the
official scoring criteria below.

WHAT THE VIDEO CONTAINS (the only ground truth you have):

Full-video analysis:
{gemini_summary}

Frame fact-check:
{qwen_report}

CAPTIONS TO JUDGE:
{json.dumps(captions, ensure_ascii=False, indent=2)}

====================================================================
OFFICIAL SCORING CRITERIA (from the competition rules):
1. Caption accuracy (0-1): how faithfully the caption reflects the
   video content. Focus on: accurate captions, specific video details,
   no major hallucinations.
2. Style match (0-1): how well the caption matches the requested tone.

OFFICIAL STYLE CHECKLIST:
- Formal: clear and professional.
- Sarcastic: sarcastic but still accurate.
- Humorous tech: tech humor plus real video details.
- humorous_non_tech: funny, everyday humour with no technical jargon.

====================================================================

EXPECTED REGISTER (what high-scoring captions look like):
- formal: 1-2 calm factual sentences. Describes subject+action+setting.
  NOT a spec sheet, NOT an inventory list. No metadata.
- sarcastic: ONE short, dry line. Still tells you what is in the video.
  Often uses "Ah yes" or deadpan framing.
- humorous_tech: ONE or two lines. Uses a tech METAPHOR mapped to real
  content. "When you..." format is common but not required.
- humorous_non_tech: ONE short line. Everyday observational humour.
  ZERO technical words. "When you..." format is common but not required.
- ALL styles: short. The references are 10-30 words each. Captions
  exceeding 40 words are almost certainly too long.

FAIL a caption if ANY of these are true:
- It states something not supported by the ground truth (hallucination).
- It does not make clear what the video shows (subject, action, setting).
- It names a place, city, landmark, brand, species or material not in
  the ground truth.
- It quotes on-screen text not marked CONFIRMED in the ground truth.
- It mentions resolution, fps, file duration, audio track, or
  "silent mode".
- formal: more than 2 sentences, or reads like an inventory/spec sheet
  rather than the calm descriptive register of the references.
- sarcastic or humorous_non_tech: more than 1 sentence.
- humorous_non_tech contains ANY tech word (code, server, deploy,
  render, pipeline, thread, node, fps, audio, buffer, etc).
- The tone does not match its style (a sarcastic caption that is not
  actually ironic, or a humorous caption that is not actually funny).
- The caption is significantly longer or more wordy than expected
  for its style.

OUTPUT: reply with ONLY a raw JSON object (no markdown, no fences):
{{"verdicts": {{"formal": {{"pass": true, "fix": ""}},
"sarcastic": {{"pass": false, "fix": "specific instruction to fix"}},
"humorous_tech": {{"pass": true, "fix": ""}},
"humorous_non_tech": {{"pass": true, "fix": ""}}}}}}

"fix" must be a concrete, actionable instruction (what to remove, add,
shorten, or rephrase). Empty string when the caption passes.
"""


def gemma_judge(client, gemini_summary, qwen_report, captions):
    raw = call_gemma(client, build_judge_prompt(gemini_summary,
                                                qwen_report, captions))
    verdicts = extract_json_object(raw).get("verdicts", {})
    if not isinstance(verdicts, dict):
        raise ValueError("judge returned no verdicts")
    return verdicts


# ============================================================
# STAGE 5 - CLAUDE: REWRITE ONLY WHAT FAILED
# ============================================================

def build_rewrite_prompt(gemini_summary, qwen_report, captions,
                         verdicts, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)
    return f"""
A judge reviewed your captions. Fix the ones that failed.

GROUND TRUTH about the video:

Full-video analysis:
{gemini_summary}

Frame fact-check:
{qwen_report}

YOUR CURRENT CAPTIONS:
{json.dumps(captions, ensure_ascii=False, indent=2)}

JUDGE VERDICTS (fix every entry where "pass" is false; keep the passing
captions EXACTLY as they are):
{json.dumps(verdicts, ensure_ascii=False, indent=2)}

STYLES:
{style_lines}

{ACCURACY_RULES}

LENGTH: formal 1-2 sentences; sarcastic ONE sentence; humorous_tech one
or two short lines; humorous_non_tech ONE sentence.

OUTPUT: reply with ONLY a raw JSON object containing ALL of these keys:
{keys}
"""


def claude_rewrite(gemini_summary, qwen_report, captions, verdicts,
                   styles):
    prompt = build_rewrite_prompt(gemini_summary, qwen_report, captions,
                                  verdicts, styles)
    raw = call_claude(prompt, max_tokens=1500, temperature=0.7)
    return validate_captions(extract_json_object(raw), styles)


# ============================================================
# PER-VIDEO CHAIN
# ============================================================

def process_task(task, google_client):
    task_id = str(task.get("task_id", "unknown"))
    styles = [s for s in (task.get("styles") or SUPPORTED_STYLES)
              if s in STYLE_GUIDES] or SUPPORTED_STYLES
    print(f"\n===== TASK {task_id} =====")

    gemini_summary = ""
    qwen_report = "none"
    audio_status = "UNKNOWN"

    try:
        with tempfile.TemporaryDirectory() as td:
            video_path = obtain_video(task["video_url"], td)
            meta = get_metadata(video_path)
            audio_status = meta["audio"]
            print(f"[{task_id}] {meta['duration']}s | {audio_status}")

            # Stage 1 - Gemini
            print(f"[{task_id}] Stage 1: Gemini summarize...")
            gemini_summary = gemini_summarize(google_client, video_path,
                                              audio_status)
            print(f"\n--- [{task_id}] GEMINI ---\n{gemini_summary}")

            # Stage 2 - Qwen validate
            try:
                print(f"[{task_id}] Stage 2: Qwen validate...")
                ts = pick_timestamps(video_path, meta["duration"],
                                     meta["fps"])
                frames = extract_frames(video_path, ts, td)
                qwen_report = qwen_validate(frames, gemini_summary)
                print(f"\n--- [{task_id}] QWEN ---\n{qwen_report}")
            except Exception as exc:
                print(f"[warn][{task_id}] Qwen validation skipped: {exc}")
                qwen_report = "Validation unavailable."

    except Exception as exc:
        print(f"[error][{task_id}] video/analysis failed: {exc}")

    if not gemini_summary:
        return {"task_id": task_id,
                "captions": {s: LAST_RESORT_CAPTION for s in styles}}

    # Stage 3 - Claude writes
    try:
        print(f"[{task_id}] Stage 3: Claude write captions...")
        captions = claude_write(gemini_summary, qwen_report,
                                audio_status, styles)
        print(f"\n--- [{task_id}] CAPTIONS ---\n"
              f"{json.dumps(captions, ensure_ascii=False, indent=2)}")
    except Exception as exc:
        print(f"[error][{task_id}] caption writing failed: {exc}")
        return {"task_id": task_id,
                "captions": {s: LAST_RESORT_CAPTION for s in styles}}

    # Stage 4 + 5 - Gemma judges, Claude rewrites only if needed
    if ENABLE_GEMMA_JUDGE:
        try:
            print(f"[{task_id}] Stage 4: Gemma judge...")
            verdicts = gemma_judge(google_client, gemini_summary,
                                   qwen_report, captions)
            print(f"[{task_id}] verdicts: "
                  f"{json.dumps(verdicts, ensure_ascii=False)}")
            failed = [s for s in styles
                      if not verdicts.get(s, {}).get("pass", True)]
            if failed:
                print(f"[{task_id}] Stage 5: Claude rewrite {failed}...")
                revised = claude_rewrite(gemini_summary, qwen_report,
                                         captions, verdicts, styles)
                # keep passing captions untouched
                for s in styles:
                    if s in failed:
                        captions[s] = revised[s]
                print(f"\n--- [{task_id}] REVISED ---\n"
                      f"{json.dumps(captions, ensure_ascii=False, indent=2)}")
            else:
                print(f"[{task_id}] all captions passed - no rewrite")
        except Exception as exc:
            print(f"[warn][{task_id}] judge stage skipped: {exc}")

    return {"task_id": task_id, "captions": captions}


# ============================================================
# BATCH MAIN
# ============================================================

def load_tasks(path):
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array")
    valid = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict) or "video_url" not in t:
            print(f"[warn] skipping malformed task at index {i}")
            continue
        t.setdefault("task_id", f"task_{i + 1}")
        valid.append(t)
    if not valid:
        raise ValueError("no valid tasks")
    return valid


def preflight():
    print("[preflight] checking API keys...")
    client = genai.Client(api_key=get_google_key())
    get_openrouter_key()
    get_anthropic_key()
    print("[preflight] OK (Google, OpenRouter, Anthropic)")
    return client


def main():
    print("=" * 56)
    print("STYLED VIDEO CAPTION AGENT v4")
    print("Gemini -> Qwen -> Claude -> Gemma -> Claude")
    print("=" * 56)

    tasks = load_tasks(TASKS_PATH)
    print(f"Loaded {len(tasks)} task(s) | parallel: {PARALLEL_VIDEOS}")

    google_client = preflight()

    results = [None] * len(tasks)
    lock = threading.Lock()
    done = {"n": 0}

    def run(i, task):
        try:
            res = process_task(task, google_client)
        except Exception as exc:
            print(f"[error] task {task.get('task_id')} crashed: {exc}")
            styles = task.get("styles") or SUPPORTED_STYLES
            res = {"task_id": str(task.get("task_id", "unknown")),
                   "captions": {s: LAST_RESORT_CAPTION for s in styles}}
        with lock:
            results[i] = res
            done["n"] += 1
            write_results_atomic([r for r in results if r is not None])
            print(f"[checkpoint] {done['n']}/{len(tasks)} written")

    with ThreadPoolExecutor(max_workers=PARALLEL_VIDEOS) as pool:
        futures = [pool.submit(run, i, t) for i, t in enumerate(tasks)]
        for f in as_completed(futures):
            if f.exception():
                print(f"[error] worker crashed: {f.exception()}")

    final = [r for r in results if r is not None]
    write_results_atomic(final)
    print("\n" + "=" * 56)
    print(f"DONE - {len(final)}/{len(tasks)} result(s) -> {RESULTS_PATH}")
    print("=" * 56)


if __name__ == "__main__":
    import signal

    def _timeout(signum, frame):
        print("\n[TIMEOUT] approaching limit - exiting with what we have")
        os._exit(0)   # hard exit; results already written atomically

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(HARD_TIMEOUT)

    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
