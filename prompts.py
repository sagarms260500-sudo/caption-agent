STYLE_GUIDES = {
    "formal": (
        "Professional, objective, factual. One or two clear sentences. "
        "Describe the subject with its key visual attributes, what it "
        "does, and the setting. Include specific visible details."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking but STILL ACCURATE. ONE sentence. "
        "Deadpan understatement or mock grandeur about what is actually "
        "happening. Must still describe what is in the video."
    ),
    "humorous_tech": (
        "Describe the subject first. Then wrap the scene in ONE natural "
        "software, programming, AI, networking, robotics, or engineering "
        "analogy that fits the scene. Do not stack multiple technical "
        "metaphors. One or two short lines."
    ),
    "humorous_non_tech": (
        "Funny everyday humour, ZERO technical words of any kind. ONE "
        "sentence. Observational comedy about what is actually visible."
    ),
}

GEMINI_PROMPT = """
You are a careful video analyst. Watch this video and describe what is there.

Rules:
- Name the main subject with obvious visual attributes, what it does, where.
- Include only visually distinctive details that help uniquely identify the scene.
  Prefer: colors, important objects, notable background elements, lighting,
  weather, camera viewpoint.
  Mention clothing only when it is essential for identifying the subject
  or is the primary visual focus of the scene.
- Do NOT guess place names, landmarks, brands, species or materials.
- Never guess. If a detail is not visually certain, omit it from the main
  sections. Put uncertain observations only under UNCERTAIN_OBSERVATIONS.
- Report on-screen text ONLY if clearly legible. If text is partially
  readable, blurry, distant, or obstructed, write "None" instead of guessing.

Return exactly:

MAIN_SUBJECT:
The main subject with key visual attributes.

ACTION:
What the subject does.

SETTING:
Describe only the visible environment.
Never infer the city, country, landmark, region, or venue.

DISTINCTIVE_VISUAL_DETAILS:
- detail
- detail
- detail

LIGHTING_AND_VIEW:
- Lighting conditions
- Viewpoint only if important to understanding the scene

ON_SCREEN_TEXT:
- clearly legible text, or "None"

UNCERTAIN_OBSERVATIONS:
Only include observations that might be incorrect. Write "None" if fully confident.
"""

QWEN_PROMPT = """
You are a fact-checker. Below is a description of a video. You see {n} frames from it.

DESCRIPTION:
{summary}

Check the description against the frames. Report only:

1. WRONG - statements the frames contradict (wrong color, wrong object, missing object)
2. MISSING - OBVIOUS scene-level things a viewer would immediately notice
3. RISKY - guesses rather than observations:
   named places, cities, landmarks, brands from appearance ("Shibuya",
   "Statue of Liberty", "Starbucks"), place-style qualifiers
   ("Shibuya-style", "European-looking"), guessed language or script
   ("South Asian text", "Japanese characters"), guessed ethnicity,
   species or material names ("granite", "ginkgo"), text you cannot
   clearly read in the frames

Never suggest adding speculative details simply to make the description
more specific. If unsure, prefer omission over guessing.
Never recommend adding details solely to make the description richer.
Accuracy is more important than completeness.

Also verify consistency. Check whether the fact sheet invents a different
main subject than the one visible across the video. Do not report natural
movement or temporary occlusion as WRONG.

Be conservative. Still frames cannot show motion or sound.
Do not narrate your reasoning. Start directly with "WRONG:".

WRONG
- Only report facts that are clearly contradicted by the frames.
- Do not report uncertain or ambiguous observations as WRONG.

MISSING:
- obvious thing omitted (or "none")

RISKY:
- guessed claim (or "none")
"""

ACCURACY_RULES = """
ACCURACY RULES:
1. Every claim must come from the VERIFIED FACT SHEET. Never invent anything.
2. ALWAYS name the subject with its key visual attribute ("a tan dog",
   "an orange kitten", "a woman in a red jacket").
3. Include specific details only when they are clearly supported by the
   VERIFIED FACT SHEET. Prefer omission over uncertain specificity.
4. BANNED (these lose points):
   - Guessed place/city/landmark/brand names, including qualifiers
     like "X-style" or "resembling X"
   - Guessed language, script, ethnicity, or region of origin
   - Quoted on-screen text unless clearly legible in the fact sheet
   - Material or species guesses stated as fact
   - Resolution, fps, duration, "N-second clip"
   - Audio/silence references ("no audio track", "silent mode")
5. SUBJECT-FIRST: every caption must name the real subject in the first
   few words. Never open with a camera part, a tech term, or abstract
   framing.
"""


def build_caption_prompt(gemini_summary, qwen_report, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)

    return f"""
You are an expert caption writer. You have two sources about ONE video:

VERIFIED FACT SHEET (saw the full video):
{gemini_summary}

VISUAL VALIDATION (saw frames only, checked the fact sheet):
{qwen_report}

HOW TO RECONCILE:
- If VISUAL VALIDATION says WRONG, trust it for visual facts.
- If it says MISSING, add it only if it is an obvious visual fact and can
  be included naturally without making the caption verbose.
- If it says RISKY, DROP it entirely.
- Never replace a dropped RISKY fact with a new guessed fact.
- VISUAL VALIDATION cannot override the fact sheet about motion or actions.

All four captions MUST describe exactly the same video.
Do not introduce new facts for one style.
Do not remove important facts from another.
Only the writing style changes.

{ACCURACY_RULES}

Write ONE caption per style:
{style_lines}

STYLE RULES:
- formal: 1-2 sentences. Be specific — include visible details that
  distinguish this video from similar ones.
- sarcastic: ONE sentence. Ironic but still tells you what is in the video.
- humorous_tech: Subject first, then ONE tech analogy that fits the scene.
- humorous_non_tech: ONE sentence. ZERO technical words of any kind
  (no code, server, deploy, render, pipeline, node, fps, buffer, etc).

FINAL VALIDATION — before returning JSON, verify:
- every caption begins with the visible subject
- every caption describes exactly the same verified facts
- only wording and tone differ
- no caption omits the primary subject or primary action
- no caption introduces new facts
- no caption introduces new objects
- no caption introduces new actions
- no caption introduces places
- no caption introduces brands
- no caption introduces occupations
- no caption introduces identities
- no caption introduces emotions
- no caption introduces unreadable text
- no caption contains any RISKY item
- each caption satisfies its requested style
If any check fails, rewrite ONLY that caption.

OUTPUT: ONLY a raw JSON object with keys: {keys}
"""
