STYLE_GUIDES = {
    "formal": (
        "Professional, objective, factual. One or two calm sentences. "
        "1-2 sentences, 25-35 words.. START with the subject. No jokes, no metadata."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking but STILL ACCURATE. ONE sentence, "
        "10-20 words. START with the subject. Deadpan understatement."
    ),
    "humorous_tech": (
        "Funny tech metaphor mapped onto real video content. ONE or two "
        "short lines, 15-25 words. START with the subject, not jargon."
    ),
    "humorous_non_tech": (
        "Funny everyday humour, ZERO technical words. ONE sentence, "
        "10-20 words. START with the subject. Observational comedy."
    ),
}

GEMINI_PROMPT = """
You are a careful video analyst. Watch this video and describe what is there.

Rules:
- Name the main subject with obvious visual attributes, what it does, where.
- Do NOT guess place names, landmarks, brands, species or materials.
- Report on-screen text ONLY if clearly legible. Mark as CONFIRMED or UNSURE.
- If something is unclear, say "unclear".

Return exactly:

MAIN_SUBJECT:
The main subject with key visual attributes.

ACTION:
What the subject does.

SETTING:
Where this takes place (describe, do not name).

VISUAL_DETAILS:
- detail
- detail
- detail

CAMERA:
Camera angle, movement, lighting.

ON_SCREEN_TEXT:
- text (CONFIRMED) or "none"

UNCERTAIN:
- anything not confident about
"""

QWEN_PROMPT = """
You are a fact-checker. Below is a description of a video. You see {n} frames from it.

DESCRIPTION:
{summary}

Check the description against the frames. Report only:

1. WRONG - statements the frames contradict (wrong color, wrong object, missing object)
2. MISSING - OBVIOUS scene-level things a viewer would immediately notice
3. RISKY - guesses rather than observations: named places, cities, landmarks,
   monuments, brands identified from appearance (e.g. "Statue of Liberty",
   "Shibuya", "New York", "Starbucks" based on what something looks like),
   species or material names, text you cannot clearly read

Be conservative. Still frames cannot show motion or sound.
Do not narrate your reasoning. Start directly with "WRONG:".

WRONG:
- statement -> correction (or "none")

MISSING:
- obvious thing omitted (or "none")

RISKY:
- guessed claim (or "none")
"""

ACCURACY_RULES = """
ACCURACY RULES:
1. Every claim must come from the FACT SHEET. Never invent anything.
2. ALWAYS name the subject with its key attribute ("a tan dog", "an orange kitten").
3. Include a FEW visual specifics (color, object, action, setting).
4. BANNED (these lose points):
   - Guessed place/city/landmark/brand names
   - Quoted on-screen text unless CONFIRMED
   - Material or species guesses
   - Resolution, fps, duration, "N-second clip"
   - Audio/silence references ("no audio track", "silent mode")
   - Clothing inventories (at most ONE distinctive item)
5. Short correct caption beats long caption with one error.
"""

REGISTER_EXAMPLES = """
Register to match (LENGTH and TONE only, never copy content):

formal: "A young orange tabby cat sits among dense green foliage,
looking directly at the camera with an alert expression."

sarcastic: "A cat outdoors, clearly plotting something elaborate
and fully confident it will succeed."

humorous_tech: "The cat has entered the garden like a freshly deployed
agent scanning for input. Rollback plan: none."

humorous_non_tech: "A tiny cat has gone outside and is now judging
everything it sees with great authority."
"""


def build_caption_prompt(gemini_summary, qwen_report, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)

    return f"""
You are an expert caption writer. Two sources about ONE video:

SOURCE A (saw the video):
{gemini_summary}

SOURCE B (saw frames only, checked A):
{qwen_report}

HOW TO RECONCILE:
- If B says WRONG, trust B for visual facts.
- If B says MISSING, add it.
- If B says RISKY, DROP it entirely.
- B cannot override A about motion or actions.

{ACCURACY_RULES}

Write ONE caption per style:
{style_lines}

{REGISTER_EXAMPLES}

LENGTH:
- formal: 1-2 sentences, 20-35 words. No clothing lists.
- sarcastic: ONE sentence, 10-20 words.
- humorous_tech: 1-2 short lines, 15-25 words. Subject first, not jargon.
- humorous_non_tech: ONE sentence, 10-20 words. ZERO tech words.

SUBJECT-FIRST: every caption must name the subject in the first few words.

OUTPUT: ONLY a raw JSON object with keys: {keys}
"""
