"""
styled_caption_agent_v3.py

AI agent that watches video clips and generates captions in requested
styles (formal / sarcastic / humorous_tech / humorous_non_tech).

Contract:
    1. Reads tasks from input/host.json on startup:
       [{"task_id": "v1", "video_url": "...", "styles": [...]}, ...]
    2. Writes output/result.json before exiting:
       [{"task_id": "v1", "captions": {"formal": "...", ...}}, ...]
    result.json is (re)written after EVERY task, atomically, so a
    crash mid-batch still leaves valid JSON on disk. It is the ONLY
    file this agent writes: all stage outputs (Gemini report, Qwen
    blind/zoom findings, Claude merge and captions, Kimi verdicts)
    are printed to the console for verification; frames and crops
    live in a temp directory that is deleted automatically.

Model roles (four independent families - no model judges its own work):
    Gemini 3.5 Flash  (video-native)  -> full video + audio analysis,
                                         media_resolution=HIGH
    Qwen3-VL-235B     (OpenRouter)    -> blind frame pass + native-res
                                         zoom crops (the "eyes")
    Claude Sonnet 4.6 (Anthropic API) -> evidence merge into a VERIFIED
                                         FACT SHEET, then styled caption
                                         writing grounded ONLY in it
    Kimi K2.6         (Moonshot API)  -> judge: fact-grounding + style
                                         adherence, max one revision

Carried over from the validated v2.2 pipeline: ffprobe audio ground
truth, scene-aware frame picking, reasoning-model-safe parsing,
minimum zoom-box size, illegible-crop retry with 3x context,
neutral-evidence merge rule, verbatim-text gate, fallback zoom grid.

Install:
    pip install -U google-genai opencv-python requests scenedetect

Configuration (see CONFIG block below):
    Paths   - TASKS_PATH / RESULTS_PATH env vars win; else the grading
              container's /input and /output are auto-detected; else
              the local Windows defaults in the CONFIG block are used.
    Keys    - paste into the constants at the top of CONFIG, or set
              env vars of the same names: GEMINI_API_KEY,
              OPENROUTER_API_KEY, ANTHROPIC_API_KEY, MOONSHOT_API_KEY
              (Moonshot only needed while ENABLE_JUDGE is True).
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from google import genai


# ============================================================
# CONFIG
# ============================================================

# ---- Paths -------------------------------------------------
# The grading harness mounts /input/tasks.json and expects
# /output/results.json — these names are specified in the hackathon
# rules and MUST match exactly or you score zero.
TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")

# ---- API keys ----------------------------------------------
# In the Docker container, these are injected as environment variables.
# NEVER hardcode keys in the submitted image — it's public.
GEMINI_API_KEY = ""
OPENROUTER_API_KEY = ""
ANTHROPIC_API_KEY = ""
MOONSHOT_API_KEY = ""

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"

QWEN_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

KIMI_MODEL = "kimi-k2.6"
MOONSHOT_URL = "https://api.moonshot.ai/v1/chat/completions"

ENABLE_JUDGE = False          # Disabled: saves ~30s/video; captions are
                              # fact-grounded by the merge stage already.
                              # Kimi-k2.6 thinking mode also truncates
                              # before emitting JSON in practice.

SUPPORTED_STYLES = ["formal", "sarcastic", "humorous_tech",
                    "humorous_non_tech"]

STYLE_GUIDES = {
    "formal": "Professional, objective, factual tone. No jokes, no "
              "editorializing. Reads like alt-text for a news agency.",
    "sarcastic": "Dry, ironic, lightly mocking. Deadpan understatement "
                 "or mock enthusiasm about what is actually visible. "
                 "Witty, never mean-spirited or crude.",
    "humorous_tech": "Funny, with technology or programming references "
                     "(latency, buffering, merge conflicts, load "
                     "balancing, npm install...). Jokes must map onto "
                     "things actually visible in the video.",
    "humorous_non_tech": "Funny, everyday humour with NO technical "
                         "jargon whatsoever. Relatable, observational "
                         "comedy about what is actually visible.",
}

MAX_OVERVIEW_FRAMES = 8
OVERVIEW_MAX_SIDE = 1344
MAX_ZOOM_REGIONS = 6
CROP_MARGIN = 0.15
CROP_MIN_SIDE = 448
MIN_BOX_FRAC = 0.08
RETRY_CONTEXT_SCALE = 3.0
MAX_CROP_RETRIES = 4

API_RETRIES = 3
API_RETRY_DELAY = 5

LAST_RESORT_CAPTION = ("A short video clip (automatic captioning was "
                       "unavailable for this item).")


# ============================================================
# SMALL UTILITIES
# ============================================================

NO_RETRY_CODES = ("HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404")


def with_retries(label, fn):
    delay = API_RETRY_DELAY
    for attempt in range(1, API_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            message = str(exc)
            if any(code in message for code in NO_RETRY_CODES):
                raise  # deterministic client error - retrying is futile
            if attempt == API_RETRIES:
                raise
            print(f"[retry] {label} failed (attempt {attempt}): {exc}. "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def _resolve_key(hardcoded, env_name):
    """Pasted constant first, then env var; x-only placeholders count
    as unset."""
    value = (hardcoded or "").strip()
    if not value or set(value.lower()) <= {"x"}:
        value = os.environ.get(env_name, "").strip()
    if not value or set(value.lower()) <= {"x"}:
        raise ValueError(
            f"{env_name} is not set. Paste it into the constant at the "
            f"top of this file or set the environment variable.")
    return value


def get_gemini_key():
    return _resolve_key(GEMINI_API_KEY, "GEMINI_API_KEY")


def get_openrouter_key():
    return _resolve_key(OPENROUTER_API_KEY, "OPENROUTER_API_KEY")


def get_anthropic_key():
    return _resolve_key(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY")


def get_moonshot_key():
    return _resolve_key(MOONSHOT_API_KEY, "MOONSHOT_API_KEY")


THINK_TAG_PATTERN = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>",
                               re.DOTALL | re.IGNORECASE)


def clean_vlm_text(text):
    return THINK_TAG_PATTERN.sub("", text or "").strip()


def parse_chat_response(data):
    """OpenAI-compatible chat response -> usable text.

    Reasoning-model-safe: strips <think> blocks and falls back to
    reasoning_content (Fireworks-style) or reasoning (OpenRouter-style)
    when content is empty. Warns on truncation.
    """
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("[warn] Model output truncated at max_tokens.")
    message = choice.get("message", {}) or {}
    content = clean_vlm_text(message.get("content") or "")
    if not content:
        content = clean_vlm_text(message.get("reasoning_content") or "")
    if not content:
        content = clean_vlm_text(message.get("reasoning") or "")
    if not content:
        raise RuntimeError("Model returned an empty message")
    return content


def extract_section(text, header):
    pattern = rf"{header}:\s*(.*?)(?=\n[A-Z][A-Z_ ]+:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_json_object(text):
    """Pull the first JSON object out of model output (fences, chatter
    and prefixes tolerated)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON object in model output")


def write_results_atomic(results, path=None):
    """Write results.json atomically so it is never half-written."""
    path = Path(path or RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


# ============================================================
# MODEL CLIENTS (all plain `requests`, one retry helper)
# ============================================================

def image_to_data_url(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_qwen_vision(prompt_text, image_paths, max_tokens=6000):
    """Qwen3-VL (thinking) on OpenRouter. Non-greedy sampling per the
    Qwen thinking-model guidance."""
    api_key = get_openrouter_key()
    content = [{"type": "text", "text": prompt_text}]
    for path in image_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(path)}})
    payload = {
        "model": QWEN_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    def post():
        response = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}: "
                               f"{response.text[:500]}")
        return parse_chat_response(response.json())

    return with_retries("Qwen (OpenRouter) call", post)


def call_claude(prompt_text, max_tokens=2500, temperature=0.3,
                system=None):
    """Claude Sonnet 4.6 on the Anthropic Messages API."""
    api_key = get_anthropic_key()
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    if system:
        payload["system"] = system
    headers = {"x-api-key": api_key,
               "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}

    def post():
        response = requests.post(ANTHROPIC_URL, headers=headers,
                                 json=payload, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(f"Anthropic HTTP {response.status_code}: "
                               f"{response.text[:500]}")
        data = response.json()
        text = "".join(block.get("text", "")
                       for block in data.get("content", [])
                       if block.get("type") == "text").strip()
        if not text:
            raise RuntimeError("Claude returned an empty message")
        return text

    return with_retries("Claude call", post)


def call_kimi(prompt_text, max_tokens=4000, temperature=1.0):
    """Kimi K2.6 on Moonshot's OpenAI-compatible endpoint.

    kimi-k2.6 requires temperature=1 and defaults to thinking mode,
    whose reasoning can consume the whole token budget before any JSON
    appears - so thinking is disabled for the judge role. If the API
    rejects any of these params, the call retries once with them
    stripped so a model swap can never break the judge stage.
    """
    api_key = get_moonshot_key()
    payload = {
        "model": KIMI_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt_text}],
    }
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    def post():
        response = requests.post(MOONSHOT_URL, headers=headers,
                                 json=payload, timeout=300)
        if (response.status_code == 400
                and any(word in response.text for word in
                        ("temperature", "top_p", "thinking"))):
            print("[warn] Moonshot rejected request params; retrying "
                  "with defaults.")
            fallback = {k: v for k, v in payload.items()
                        if k not in ("temperature", "top_p", "thinking")}
            response = requests.post(MOONSHOT_URL, headers=headers,
                                     json=fallback, timeout=300)
        if response.status_code != 200:
            raise RuntimeError(f"Moonshot HTTP {response.status_code}: "
                               f"{response.text[:500]}")
        return parse_chat_response(response.json())

    return with_retries("Kimi call", post)


# ============================================================
# VIDEO ACQUISITION + TECHNICAL GROUND TRUTH
# ============================================================

def obtain_video(source, work_dir):
    if not str(source).lower().startswith(("http://", "https://")):
        local = Path(source)
        if not local.is_file():
            raise FileNotFoundError(f"Local video not found: {local}")
        return str(local)
    video_path = os.path.join(work_dir, "input_video.mp4")
    response = requests.get(source, stream=True, timeout=300)
    response.raise_for_status()
    with open(video_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return video_path


def probe_audio_stream(video_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return ("UNKNOWN - ffprobe not installed; audio presence could "
                "not be verified")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return f"UNKNOWN - ffprobe error: {result.stderr.strip()[:200]}"
        codecs = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if codecs:
            return (f"AUDIO TRACK PRESENT - {len(codecs)} stream(s), "
                    f"codec(s): {', '.join(codecs)}")
        return "NO AUDIO TRACK - the file is video-only and plays silently"
    except Exception as exc:
        return f"UNKNOWN - audio probe failed: {exc}"


def get_video_metadata(video_path):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError("Could not open video file")
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    duration = frame_count / fps if fps else 0
    return {
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
        "audio_status": probe_audio_stream(video_path),
    }


def format_technical_facts(metadata):
    return (f"- Resolution: {metadata['resolution']}\n"
            f"- Frame rate: {metadata['fps']} fps\n"
            f"- Duration: {metadata['duration_seconds']} seconds "
            f"({metadata['frame_count']} frames)\n"
            f"- Audio: {metadata['audio_status']}")


# ============================================================
# SCENE-AWARE FRAME EXTRACTION + ZOOM CROPPING (from v2.2)
# ============================================================

def detect_scenes(video_path):
    try:
        from scenedetect import detect, ContentDetector
        scene_list = detect(str(video_path), ContentDetector())
        return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
    except Exception as exc:
        print(f"[warn] Scene detection unavailable ({exc}); "
              f"using even spacing.")
        return []


def evenly_spaced(duration, fps, count):
    last = max(duration - (1.0 / fps if fps else 0.04), 0.0)
    if count <= 1:
        return [0.0]
    return [round(i * last / (count - 1), 3) for i in range(count)]


def pick_frame_timestamps(video_path, duration, fps,
                          max_frames=MAX_OVERVIEW_FRAMES):
    scenes = detect_scenes(video_path)
    if len(scenes) >= 2:
        candidates = []
        for start, end in scenes:
            span = max(end - start, 0.0)
            candidates.append(min(start + min(0.2, span * 0.1),
                                  max(duration - 0.05, 0.0)))
            candidates.append(start + span / 2.0)
        candidates = sorted(set(round(t, 3) for t in candidates))
        if len(candidates) > max_frames:
            step = len(candidates) / max_frames
            candidates = [candidates[int(i * step)]
                          for i in range(max_frames)]
        return sorted(set(candidates))
    count = min(max_frames, max(int(duration) + 2, 3))
    return evenly_spaced(duration, fps, count)


def extract_frames(video_path, timestamps, output_dir):
    full_dir = Path(output_dir) / "frames_full"
    over_dir = Path(output_dir) / "frames_overview"
    full_dir.mkdir(parents=True, exist_ok=True)
    over_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError("Could not open video for frame extraction")

    frames = []
    for index, timestamp in enumerate(timestamps, start=1):
        video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        success, frame = video.read()
        if not success:
            continue
        full_path = full_dir / f"frame_{index}_{timestamp:.2f}s.jpg"
        cv2.imwrite(str(full_path), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        height, width = frame.shape[:2]
        scale = OVERVIEW_MAX_SIDE / max(width, height)
        overview = (cv2.resize(frame, (int(width * scale),
                                       int(height * scale)),
                               interpolation=cv2.INTER_AREA)
                    if scale < 1.0 else frame)
        over_path = over_dir / f"frame_{index}_{timestamp:.2f}s.jpg"
        cv2.imwrite(str(over_path), overview,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        frames.append({"index": index, "timestamp": timestamp,
                       "full_path": str(full_path),
                       "overview_path": str(over_path)})
    video.release()
    if not frames:
        raise RuntimeError("No frames could be extracted")
    return frames


def normalize_box(box_rel, min_frac=MIN_BOX_FRAC):
    x, y, w, h = box_rel
    w, h = max(w, 0.005), max(h, 0.005)
    if w < min_frac:
        x, w = x - (min_frac - w) / 2.0, min_frac
    if h < min_frac:
        y, h = y - (min_frac - h) / 2.0, min_frac
    x = min(max(x, 0.0), 1.0 - w)
    y = min(max(y, 0.0), 1.0 - h)
    return (x, y, min(w, 1.0), min(h, 1.0))


def expand_box(box_rel, scale):
    x, y, w, h = box_rel
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = min(w * scale, 1.0), min(h * scale, 1.0)
    nx = min(max(cx - nw / 2.0, 0.0), 1.0 - nw)
    ny = min(max(cy - nh / 2.0, 0.0), 1.0 - nh)
    return (nx, ny, nw, nh)


def save_zoom_crop(full_frame_path, box_rel, out_path,
                   margin=CROP_MARGIN, min_side=CROP_MIN_SIDE):
    image = cv2.imread(str(full_frame_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    x, y, w, h = normalize_box(box_rel)
    pad = margin * max(w, h)
    x1 = int(max(x - pad, 0.0) * width)
    y1 = int(max(y - pad, 0.0) * height)
    x2 = int(min(x + w + pad, 1.0) * width)
    y2 = int(min(y + h + pad, 1.0) * height)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if max(ch, cw) < min_side:
        scale = min_side / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_CUBIC)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return str(out_path)


# ============================================================
# PROMPTS
# ============================================================

DETAIL_CHECKLIST = """
Systematically check and report on EVERY item below. Prefer specific,
distinguishing observations over generic wording:
- People/animals present, their actions, clothing, and interactions.
- Vehicle or object types present and any distinctive colors/liveries.
- Structure of the scene: road/room/terrain layout, notable fixtures.
- Infrastructure and props: signs, screens, platforms, equipment.
- Setting: indoor/outdoor, building types, vegetation, season cues,
  weather, lighting direction and shadow length.
- Camera: angle, static or moving, cuts, motion blur, speed-ramp cues.
- Text: signs, screens, labels. Transcribe ONLY if actually legible;
  otherwise say "text visible but illegible".
- Anything else that distinguishes this exact video from similar ones.
"""


def build_gemini_caption_prompt(technical_facts):
    return f"""
You are an expert video captioning agent.

Analyze this video carefully using both visual and audio information.

VERIFIED TECHNICAL FACTS (measured directly from the file - treat as
ground truth, especially the audio status):
{technical_facts}

Important rules:
- Do not guess names, places, brands, identities, or relationships.
- If something is unclear, say "unclear".
- Prefer specific visual details over generic wording.
- Mention audio only in a way consistent with the verified audio status.
{DETAIL_CHECKLIST}

Return the response using exactly this structure:

BEST_CAPTION:
One strong, natural caption.

DETAILED_CAPTION:
One or two detailed sentences describing the video.

SCENE_TYPE:
General type of video.

VISUAL_STYLE:
Camera style, lighting, mood, overall look.

SETTING:
Location or environment.

MAIN_SUBJECTS:
- Subject 1

SUBJECT_DETAILS:
- Important visual detail 1

TIMELINE:
00:00 - Event description

AUDIO_SUMMARY:
Speech, music, sounds, or the verified audio status.

IMPORTANT_VISUAL_DETAILS:
- Detail 1

CONFIDENCE:
High, medium, or low.

LIMITATIONS:
- Limitation 1
"""


def build_qwen_blind_prompt(timestamps, technical_facts):
    ts_text = ", ".join(f"{t:.2f}s" for t in timestamps)
    return f"""
You are a meticulous visual analyst. You are given {len(timestamps)}
still frames sampled from a single video at these timestamps: {ts_text}.
You have NOT seen the video itself and know nothing else about it.

VERIFIED TECHNICAL FACTS about the source file:
{technical_facts}

Describe only what is visually supported by the frames. Do not guess
names, places, or brands. Still frames cannot establish playback speed:
note motion blur if present, but do NOT speculate about time-lapse
versus real time.
{DETAIL_CHECKLIST}

Then propose regions worth magnifying: small, distant, or ambiguous
objects whose identity would become clearer at full resolution. Boxes
use RELATIVE coordinates: x, y = top-left corner, w, h = width and
height, all fractions of the image between 0 and 1. Propose at most
{MAX_ZOOM_REGIONS} regions.

CRITICAL OUTPUT RULES:
- Do NOT narrate your reasoning or deliberation.
- Your reply must start directly with "FRAME_OBSERVATIONS:" and contain
  ONLY the sections below, in this order.
- Every zoom region must be on its own line starting with "REGION:" and
  match the format exactly, with plain decimal numbers, e.g.:
  REGION: frame=2 box=0.40,0.50,0.20,0.15 reason=possible bus in median lane

Return exactly this structure:

FRAME_OBSERVATIONS:
Concise description covering the checklist.

DISTINCTIVE_DETAILS:
- Detail 1

TEXT_FOUND:
- Legible transcriptions, or "none legible"

BLIND_CAPTION:
One specific, natural caption based only on these frames.

ZOOM_REGIONS:
REGION: frame=1 box=0.42,0.55,0.20,0.15 reason=short reason here
"""


def build_qwen_zoom_prompt(crop_descriptions):
    listing = "\n".join(crop_descriptions)
    return f"""
You are a meticulous visual analyst. Each image below is a MAGNIFIED
native-resolution crop taken from frames of the same video:

{listing}

For each crop, identify exactly what is visible: object types, vehicle
categories, colors and liveries, infrastructure, clothing, logos.
Transcribe any legible text character-by-character and tag every
transcription with (LEGIBILITY: CLEAR) or (LEGIBILITY: PARTIAL); if
text is present but unreadable, write ILLEGIBLE. Do not guess beyond
the pixels.

Do NOT narrate your reasoning. Your reply must start directly with
"CROP_FINDINGS:" and contain only the sections below.

Return exactly this structure:

CROP_FINDINGS:
CROP 1: findings
CROP 2: findings

NEW_DISTINCTIVE_DETAILS:
- Any detail a viewer of the full frames likely missed
"""


def build_merge_prompt(technical_facts, gemini_report, blind_report,
                       zoom_findings):
    return f"""
You are the evidence editor for a video-captioning system. You are
given several independent evidence sources about ONE video. Produce a
verified fact sheet and one neutral base caption.

EVIDENCE 1 - VERIFIED TECHNICAL FACTS (ground truth):
{technical_facts}

EVIDENCE 2 - FULL-VIDEO ANALYSIS (model saw video + audio at reduced
visual resolution; strongest for motion, timing, audio):
{gemini_report}

EVIDENCE 3 - BLIND FRAME ANALYSIS (model saw only still frames, never
saw Evidence 2; strongest for static scene composition):
{blind_report}

EVIDENCE 4 - MAGNIFIED CROP FINDINGS (native-resolution zoom-ins;
STRONGEST for small objects, fine details, and text):
{zoom_findings}

Rules:
- Include only details supported by the evidence. No new guesses.
- For small/fine visual details trust Evidence 4, then 3, then 2. For
  motion, temporal order, and audio trust Evidence 2, but audio
  statements must match Evidence 1.
- NEUTRAL-EVIDENCE RULE: if a crop is blurred, mis-aimed, or reports
  text as ILLEGIBLE / not discernible, that crop is NEUTRAL - it
  neither confirms nor refutes. NEVER remove or "correct" a detail
  from Evidence 2 or 3 just because a crop failed to show it clearly.
- VERBATIM-TEXT GATE: include quoted text (signs, names, brands) in the
  fact sheet only if (a) two sources transcribe it identically, or
  (b) one source transcribes it tagged LEGIBILITY: CLEAR. Rule (a)
  stands even when a magnified crop captured only a PARTIAL fragment
  of the same text - a partial crop is NEUTRAL and cannot veto two
  identical full transcriptions. Text tagged PARTIAL must NEVER appear
  in quotation marks anywhere in VERIFIED_FACTS or BASE_CAPTION;
  describe it generically (e.g. "a partially legible phone number",
  "a blue street-name sign in Korean") and note the fragment under
  CONFLICTS. If transcriptions disagree by even one character, do the
  same.
- Do not name cities, countries, brands, or people unless literally
  transcribed under the gate above.

Return exactly this structure:

VERIFIED_FACTS:
- fact 1 (source: E2/E3/E4)
- fact 2 (source: ...)

CONFLICTS:
- contradictions and which side you trusted, or "none"

BASE_CAPTION:
One neutral, factual, specific caption (1-2 sentences).
"""


def build_style_prompt(fact_sheet, base_caption, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES.get(s, s)}'
                            for s in styles)
    keys_hint = ", ".join(f'"{s}"' for s in styles)
    return f"""
You are a professional caption writer. Below is a VERIFIED FACT SHEET
about one video, plus a neutral base caption. Write one caption per
requested style.

VERIFIED FACT SHEET (the ONLY source of truth):
{fact_sheet}

BASE CAPTION:
{base_caption}

REQUESTED STYLES:
{style_lines}

HARD RULES:
- Every claim in every caption must come from the fact sheet. Do NOT
  invent objects, people, animals, events, sounds, or places.
- Never quote any text the fact sheet marks as PARTIAL, partially
  legible, or unconfirmed - refer to it generically or omit it.
- Do not enumerate technical metadata (resolution, frame rate, file
  duration) in any caption unless the video is literally about those
  properties; at most one such number may appear where it serves the
  style (e.g. a sarcastic jab).
- The formal caption must be at most two sentences and read like
  polished alt-text, not a spec sheet.
- Humor and sarcasm come from framing, wordplay, exaggeration of REAL
  facts, or ironic observation - never from fabricated content.
  Grounded example: joking that heavy visible traffic is "buffering"
  is fine; inventing "a dog crossing the road" is NOT.
- 1-2 sentences per caption. Each caption must stand alone.
- Keep humor good-natured; nothing crude, cruel, or political.

OUTPUT FORMAT: respond with ONLY a raw JSON object (no markdown, no
code fences, no commentary) whose keys are exactly: {keys_hint}
and whose values are the caption strings.
"""


def build_judge_prompt(fact_sheet, captions):
    return f"""
You are a strict quality judge for styled video captions. You are given
the VERIFIED FACT SHEET for a video and one caption per style.

VERIFIED FACT SHEET (the only source of truth):
{fact_sheet}

CAPTIONS TO JUDGE:
{json.dumps(captions, ensure_ascii=False, indent=2)}

STYLE DEFINITIONS:
{json.dumps({k: STYLE_GUIDES[k] for k in captions if k in STYLE_GUIDES},
            ensure_ascii=False, indent=2)}

For each caption decide:
- fact_ok: true only if every concrete claim (objects, actions,
  settings, text) is supported by the fact sheet. Figurative language
  and irony are fine; invented concrete details are not.
- style_ok: true only if the caption clearly matches its style
  definition (e.g. humorous_non_tech must contain NO technical jargon;
  humorous_tech must contain a tech/programming reference).
- issues: short explanation when either check fails, else "".

OUTPUT FORMAT: respond with ONLY a raw JSON object (no markdown, no
fences) shaped exactly like:
{{"verdicts": {{"<style>": {{"fact_ok": true, "style_ok": true,
"issues": ""}}}}, "overall_pass": true}}
"""


def build_revision_prompt(fact_sheet, captions, verdicts, styles):
    keys_hint = ", ".join(f'"{s}"' for s in styles)
    return f"""
You are revising styled video captions that a judge flagged.

VERIFIED FACT SHEET (the ONLY source of truth):
{fact_sheet}

CURRENT CAPTIONS:
{json.dumps(captions, ensure_ascii=False, indent=2)}

JUDGE VERDICTS (fix every entry where fact_ok or style_ok is false;
keep passing captions unchanged):
{json.dumps(verdicts, ensure_ascii=False, indent=2)}

STYLE DEFINITIONS:
{json.dumps({k: STYLE_GUIDES[k] for k in styles if k in STYLE_GUIDES},
            ensure_ascii=False, indent=2)}

Same hard rules as before: every claim must come from the fact sheet;
humor from framing of real facts only; 1-2 sentences each.

OUTPUT FORMAT: ONLY a raw JSON object with keys exactly: {keys_hint}
"""


# ============================================================
# EVIDENCE STAGES
# ============================================================

def get_file_state(uploaded_file):
    return getattr(uploaded_file.state, "name", str(uploaded_file.state))


def wait_for_gemini_file(client, uploaded_file):
    while get_file_state(uploaded_file) == "PROCESSING":
        time.sleep(5)
        uploaded_file = client.files.get(name=uploaded_file.name)
    if get_file_state(uploaded_file) == "FAILED":
        raise RuntimeError("Gemini failed to process the uploaded video")
    return uploaded_file


def resolve_media_resolution_config():
    if not GEMINI_MEDIA_RESOLUTION:
        return None
    try:
        from google.genai import types
        value = GEMINI_MEDIA_RESOLUTION
        enum = getattr(types, "MediaResolution", None)
        if enum is not None and hasattr(enum, GEMINI_MEDIA_RESOLUTION):
            value = getattr(enum, GEMINI_MEDIA_RESOLUTION)
        return types.GenerateContentConfig(media_resolution=value)
    except Exception as exc:
        print(f"[warn] media_resolution unsupported by this SDK ({exc}).")
        return None


def gemini_generate(client, model, contents, use_media_resolution=False):
    config = resolve_media_resolution_config() if use_media_resolution \
        else None

    def call(with_config):
        if with_config is not None:
            return client.models.generate_content(
                model=model, contents=contents, config=with_config)
        return client.models.generate_content(model=model,
                                              contents=contents)

    try:
        return call(config)
    except Exception as exc:
        if config is not None:
            print(f"[warn] media_resolution rejected ({exc}); "
                  f"retrying without it.")
            return call(None)
        raise


def analyze_video_with_gemini(client, video_path, technical_facts):
    uploaded_file = None
    try:
        uploaded_file = client.files.upload(file=video_path)
        uploaded_file = wait_for_gemini_file(client, uploaded_file)
        response = with_retries(
            "Gemini video analysis",
            lambda: gemini_generate(
                client, GEMINI_MODEL,
                [uploaded_file, build_gemini_caption_prompt(
                    technical_facts)],
                use_media_resolution=True))
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text
    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


REGION_PATTERN = re.compile(
    r"REGION:\s*frame\s*=\s*(\d+)[\s,]+box\s*=\s*[\(\[]?\s*"
    r"([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*[\)\]]?"
    r"[\s,]+reason\s*=\s*(.+)",
    re.IGNORECASE)

CROP_FINDING_PATTERN = re.compile(
    r"CROP\s+(\d+)\s*:(.*?)(?=\nCROP\s+\d+\s*:|\nNEW_DISTINCTIVE|\Z)",
    re.DOTALL | re.IGNORECASE)

ILLEGIBLE_PATTERN = re.compile(
    r"ILLEGIBLE|not\s+legible|no\s+text\s+is\s+legible|not\s+discernible"
    r"|too\s+blurr|cannot\s+be\s+read|unreadable",
    re.IGNORECASE)


def parse_zoom_regions(blind_output, frames):
    regions, by_index = [], {f["index"]: f for f in frames}
    for match in REGION_PATTERN.finditer(blind_output):
        frame_idx = int(match.group(1))
        if frame_idx not in by_index:
            continue
        try:
            box = tuple(float(match.group(i)) for i in range(2, 6))
        except ValueError:
            continue
        regions.append({"frame": by_index[frame_idx], "box": box,
                        "reason": match.group(6).strip()[:120]})
        if len(regions) >= MAX_ZOOM_REGIONS:
            break
    return regions


def fallback_zoom_regions(frames):
    mid, first = frames[len(frames) // 2], frames[0]
    grid = [
        (mid, (0.35, 0.30, 0.30, 0.35), "fallback grid - center"),
        (mid, (0.02, 0.35, 0.30, 0.35), "fallback grid - left side"),
        (mid, (0.68, 0.35, 0.30, 0.35), "fallback grid - right side"),
        (first, (0.30, 0.62, 0.40, 0.36), "fallback grid - foreground"),
    ]
    return [{"frame": f, "box": b, "reason": r} for f, b, r in grid]


def find_illegible_crops(zoom_output, records):
    by_number = {rec["n"]: rec for rec in records}
    failed = []
    for match in CROP_FINDING_PATTERN.finditer(zoom_output):
        number = int(match.group(1))
        if number in by_number and ILLEGIBLE_PATTERN.search(match.group(2)):
            failed.append(by_number[number])
    return failed[:MAX_CROP_RETRIES]


def blind_frame_analysis(frames, technical_facts):
    prompt = build_qwen_blind_prompt([f["timestamp"] for f in frames],
                                     technical_facts)
    return call_qwen_vision(prompt, [f["overview_path"] for f in frames],
                            max_tokens=6000)


def zoom_pass(regions, output_dir):
    if not regions:
        return "No zoom regions were proposed.", []
    crop_dir = Path(output_dir) / "crops"
    records, descriptions = [], []
    for region in regions:
        number = len(records) + 1
        saved = save_zoom_crop(
            region["frame"]["full_path"], region["box"],
            crop_dir / f"crop_{number}_frame"
                       f"{region['frame']['index']}.jpg")
        if not saved:
            continue
        records.append({"n": number, "region": region, "path": saved})
        descriptions.append(
            f"CROP {number}: from frame {region['frame']['index']} "
            f"(t={region['frame']['timestamp']:.2f}s), proposed because: "
            f"{region['reason']}")
    if not records:
        return "All proposed regions were unusable.", []

    zoom_output = call_qwen_vision(build_qwen_zoom_prompt(descriptions),
                                   [r["path"] for r in records],
                                   max_tokens=4000)
    return zoom_output, [r["path"] for r in records]


# ============================================================
# FACT SHEET, STYLES, JUDGE
# ============================================================

def merge_evidence_with_claude(technical_facts, gemini_report,
                               blind_report, zoom_findings):
    merge_output = call_claude(
        build_merge_prompt(technical_facts, gemini_report, blind_report,
                           zoom_findings),
        max_tokens=2500, temperature=0.2)
    facts = extract_section(merge_output, "VERIFIED_FACTS")
    base = extract_section(merge_output, "BASE_CAPTION")
    base = base.replace("\n", " ").strip()
    if not facts or not base:
        raise RuntimeError("Merge output missing VERIFIED_FACTS or "
                           "BASE_CAPTION")
    return merge_output, facts, base


def gemini_only_fact_sheet(gemini_report):
    """Fallback fact base when frame verification is unavailable."""
    parts = []
    for header in ("DETAILED_CAPTION", "SETTING", "SUBJECT_DETAILS",
                   "IMPORTANT_VISUAL_DETAILS", "AUDIO_SUMMARY"):
        body = extract_section(gemini_report, header)
        if body:
            parts.append(f"{header}:\n{body}")
    facts = "\n".join(parts) or gemini_report
    base = (extract_section(gemini_report, "BEST_CAPTION")
            .replace("\n", " ").strip())
    return facts, base


def validate_captions(candidate, styles):
    if not isinstance(candidate, dict):
        raise ValueError("Captions payload is not a JSON object")
    captions = {}
    for style in styles:
        value = candidate.get(style)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty caption for '{style}'")
        captions[style] = " ".join(value.split())
    return captions


def generate_styled_captions(fact_sheet, base_caption, styles):
    prompt = build_style_prompt(fact_sheet, base_caption, styles)
    for attempt in (1, 2):
        raw = call_claude(prompt, max_tokens=1500, temperature=0.8)
        try:
            return validate_captions(extract_json_object(raw), styles)
        except Exception as exc:
            print(f"[warn] Style JSON invalid (attempt {attempt}): {exc}")
            prompt += ("\n\nREMINDER: output ONLY the raw JSON object, "
                       "with every requested style key present.")
    raise RuntimeError("Style generation returned invalid JSON twice")


def judge_and_revise(fact_sheet, captions, styles):
    raw = call_kimi(build_judge_prompt(fact_sheet, captions),
                    max_tokens=4000)
    verdicts = extract_json_object(raw).get("verdicts", {})
    print("Judge verdicts:", json.dumps(verdicts, ensure_ascii=False))
    flagged = [s for s in styles
               if not verdicts.get(s, {}).get("fact_ok", True)
               or not verdicts.get(s, {}).get("style_ok", True)]
    if not flagged:
        return captions, verdicts, False
    print(f"Judge flagged {flagged}; requesting one revision...")
    raw = call_claude(build_revision_prompt(fact_sheet, captions,
                                            verdicts, styles),
                      max_tokens=1500, temperature=0.7)
    revised = validate_captions(extract_json_object(raw), styles)
    return revised, verdicts, True


# ============================================================
# PER-TASK RUNNER (graceful degradation ladder)
# ============================================================

def process_task(task, gemini_client):
    task_id = str(task.get("task_id", "unknown"))
    styles = [s for s in (task.get("styles") or SUPPORTED_STYLES)]
    task_start = time.time()
    print(f"\n########## TASK {task_id} ({', '.join(styles)}) ##########")

    fact_sheet = base_caption = ""
    gemini_report = blind_report = zoom_findings = ""

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = obtain_video(task["video_url"], temp_dir)
            metadata = get_video_metadata(video_path)
            technical_facts = format_technical_facts(metadata)
            print(technical_facts)

            frames = []
            try:
                timestamps = pick_frame_timestamps(
                    video_path, metadata["duration_seconds"],
                    metadata["fps"])
                frames = extract_frames(video_path, timestamps, temp_dir)
                print(f"Extracted {len(frames)} frames.")
            except Exception as exc:
                print(f"[warn] Frame extraction failed: {exc}")

            # === PARALLEL: Gemini (video+audio) and Qwen blind (frames)
            # are completely independent — run them simultaneously to
            # save ~60s per video. ===
            gemini_exc = qwen_exc = None

            def run_gemini():
                nonlocal gemini_report
                t0 = time.time()
                print("[Stage 5] Gemini video analysis...")
                gemini_report = analyze_video_with_gemini(
                    gemini_client, video_path, technical_facts)
                print(f"[Stage 5] Gemini done ({time.time()-t0:.1f}s)")

            def run_qwen_blind():
                nonlocal blind_report
                if not frames:
                    return
                t0 = time.time()
                print("[Stage 6] Qwen blind frame analysis...")
                blind_report = blind_frame_analysis(frames,
                                                    technical_facts)
                print(f"[Stage 6] Qwen blind done ({time.time()-t0:.1f}s)")

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_gemini = pool.submit(run_gemini)
                fut_qwen = pool.submit(run_qwen_blind)
                for fut in as_completed([fut_gemini, fut_qwen]):
                    exc = fut.exception()
                    if exc and fut is fut_gemini:
                        gemini_exc = exc
                    elif exc:
                        qwen_exc = exc

            if gemini_exc:
                raise gemini_exc

            print("\n=== GEMINI OUTPUT ===")
            print(gemini_report)

            if qwen_exc:
                print(f"[warn] Qwen blind failed: {qwen_exc}")
            elif blind_report:
                print("\n=== QWEN BLIND OUTPUT ===")
                print(blind_report)

            # === SEQUENTIAL: zoom pass depends on blind output ===
            if blind_report and frames:
                try:
                    regions = parse_zoom_regions(blind_report, frames)
                    if not regions:
                        print("[warn] No REGION lines parsed; using "
                              "fallback grid.")
                        regions = fallback_zoom_regions(frames)
                    t0 = time.time()
                    print(f"\n[Stage 7] Zoom pass on {len(regions)} "
                          f"region(s)...")
                    zoom_findings, _ = zoom_pass(regions, temp_dir)
                    print(f"[Stage 7] Zoom done ({time.time()-t0:.1f}s)")
                    print("\n=== QWEN ZOOM OUTPUT ===")
                    print(zoom_findings)
                except Exception as exc:
                    print(f"[warn] Zoom pass degraded: {exc}")

        try:
            t0 = time.time()
            print("\n[Stage 8] Claude evidence merge...")
            merge_output, fact_sheet, base_caption = \
                merge_evidence_with_claude(
                    technical_facts, gemini_report,
                    blind_report or "UNAVAILABLE",
                    zoom_findings or "UNAVAILABLE")
            print(f"[Stage 8] Merge done ({time.time()-t0:.1f}s)")
            print("\n=== CLAUDE MERGE OUTPUT ===")
            print(merge_output)
        except Exception as exc:
            print(f"[warn] Merge degraded to Gemini-only facts: {exc}")
            fact_sheet, base_caption = gemini_only_fact_sheet(
                gemini_report)

    except Exception as exc:
        print(f"[error] Evidence gathering failed for {task_id}: {exc}")

    if not base_caption:
        captions = {style: LAST_RESORT_CAPTION for style in styles}
        elapsed = time.time() - task_start
        print(f"[TASK {task_id}] FAILED — {elapsed:.1f}s")
        return {"task_id": task_id, "captions": captions}

    try:
        t0 = time.time()
        print("\n[Stage 9] Claude styled caption generation...")
        captions = generate_styled_captions(fact_sheet, base_caption,
                                            styles)
        print(f"[Stage 9] Styles done ({time.time()-t0:.1f}s)")
        print("\n=== STYLED CAPTIONS ===")
        print(json.dumps(captions, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[warn] Style generation failed ({exc}); using base "
              f"caption for all styles.")
        return {"task_id": task_id,
                "captions": {style: base_caption for style in styles}}

    if ENABLE_JUDGE:
        try:
            print("\n[Stage 10] Kimi judge...")
            captions, _, revised = judge_and_revise(fact_sheet, captions,
                                                    styles)
            if revised:
                print("\n=== REVISED CAPTIONS (after judge) ===")
                print(json.dumps(captions, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[warn] Judge stage skipped: {exc}")

    elapsed = time.time() - task_start
    print(f"\n[TASK {task_id}] DONE — {elapsed:.1f}s")
    return {"task_id": task_id, "captions": captions}


# ============================================================
# BATCH MAIN
# ============================================================

def load_tasks(path):
    raw = Path(path).read_text(encoding="utf-8")
    tasks = json.loads(raw)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a JSON array")
    valid = []
    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or "video_url" not in task:
            print(f"[warn] Skipping malformed task at index {i}")
            continue
        task.setdefault("task_id", f"task_{i + 1}")
        valid.append(task)
    if not valid:
        raise ValueError("No valid tasks found in tasks.json")
    return valid


def preflight():
    print("[Stage 0] Preflight: checking API keys...")
    gemini_client = genai.Client(api_key=get_gemini_key())
    get_openrouter_key()
    get_anthropic_key()
    if ENABLE_JUDGE:
        try:
            get_moonshot_key()
        except ValueError as exc:
            print(f"[warn] {exc} Judge stage will be skipped.")
    print("Preflight OK.")
    return gemini_client


def main():
    print("=" * 54)
    print("STYLED VIDEO CAPTION AGENT v3")
    print("=" * 54)

    tasks = load_tasks(TASKS_PATH)
    print(f"Loaded {len(tasks)} task(s) from {TASKS_PATH}")

    gemini_client = preflight()

    results = []
    for task in tasks:
        try:
            result = process_task(task, gemini_client)
        except Exception as exc:
            print(f"[error] Unhandled failure in task "
                  f"{task.get('task_id')}: {exc}")
            styles = task.get("styles") or SUPPORTED_STYLES
            result = {"task_id": str(task.get("task_id", "unknown")),
                      "captions": {s: LAST_RESORT_CAPTION
                                   for s in styles}}
        results.append(result)
        write_results_atomic(results)
        print(f"[checkpoint] {len(results)}/{len(tasks)} task(s) "
              f"written to {RESULTS_PATH}")

    write_results_atomic(results)
    print("\n" + "=" * 54)
    print(f"DONE - {len(results)} result(s) in {RESULTS_PATH}")
    for r in results:
        print(f"  {r['task_id']}: {list(r['captions'].keys())}")
    print("=" * 54)


if __name__ == "__main__":
    import signal

    # Hard budget: the grading harness kills at 10 min; we self-stop
    # at 9 min to flush whatever results we have and exit cleanly.
    HARD_TIMEOUT = int(os.environ.get("HARD_TIMEOUT", 540))

    def _timeout_handler(signum, frame):
        print("\n[TIMEOUT] Approaching 10-minute limit — writing "
              "results and exiting.")
        raise SystemExit(0)

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(HARD_TIMEOUT)

    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[fatal] {exc}")
        sys.exit(1)
