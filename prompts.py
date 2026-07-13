STYLE_GUIDES = {
    "formal": (
        "Professional, objective, factual. One or two calm sentences. "
        "START with the visible subject. No jokes, no metadata, no guessing."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking but STILL ACCURATE. ONE sentence. "
        "START with the visible subject. Deadpan understatement only."
    ),
    "humorous_tech": (
        "Funny tech metaphor mapped onto real video content. ONE or two short lines. "
        "START with the visible subject, not jargon."
    ),
    "humorous_non_tech": (
        "Funny everyday humour, ZERO technical words. ONE sentence. "
        "START with the visible subject. Observational comedy only."
    ),
}

GEMINI_PROMPT = """
You are a careful video analyst. Watch this video and describe only what is visibly there.

Rules:
- Name the main subject with obvious visual attributes, what it does, and where it is.
- Describe the scene directly. Do NOT use place-style or lookalike labels.
  BANNED examples: "Shibuya-style", "Tokyo-looking", "European-style",
  "South Asian-looking", "looks like X", "X-style", "X-looking".
- Do NOT guess place names, cities, countries, landmarks, brands, regions,
  cultures, ethnicities, nationalities, languages, scripts, species, or materials.
- Do NOT use sensitive appearance wording unless it is essential and clearly visible.
  Prefer neutral wording like "hands", "a person", "a woman", "a man", "a child".
- Do NOT guess exact numbers, IDs, signs, or text unless clearly legible.
  If text is not clearly readable, say "unreadable text/signage".
- Report on-screen text ONLY if clearly legible. Mark exact words as CONFIRMED.
- If something is unclear, say "unclear".
- Do not mention audio, silence, video length, fps, or resolution.

Return exactly:

MAIN_SUBJECT:
The main subject with key visual attributes.

ACTION:
What the subject does.

SETTING:
Where this takes place, described visually without naming or guessing a place.

VISUAL_DETAILS:
- detail
- detail
- detail

CAMERA:
Camera angle, movement, lighting.

ON_SCREEN_TEXT:
- exact clearly legible text (CONFIRMED), "unreadable text/signage", or "none"

UNCERTAIN:
- anything not confident about
"""

QWEN_PROMPT = """
You are a strict visual fact-checker. Below is a description of a video. You see {n} frames from it.

DESCRIPTION:
{summary}

Check the description against the frames. Report only:

1. WRONG - statements the frames contradict
2. MISSING - OBVIOUS scene-level things a viewer would immediately notice
3. RISKY - guesses rather than observations. Flag these for removal:
   - named places, cities, countries, regions, landmarks, monuments, brands
   - place-style / culture-style / lookalike labels:
     "Shibuya-style", "Tokyo-looking", "European-style", "South Asian-looking",
     "looks like X", "X-style", "X-looking"
   - guessed language, script, ethnicity, nationality, culture, or region
   - sensitive appearance wording that is not essential
   - specific species, breed, or material guesses
   - exact numbers, IDs, signs, or text unless clearly and fully legible
   - any on-screen text that is blurry, partial, tiny, or uncertain
   - audio, silence, resolution, fps, duration, or clip-length claims

For text:
- only accept exact, clearly legible transcription
- if text is unclear, flag it as RISKY
- if the description assigns a language/script/region to text, flag it as RISKY

Be conservative. Still frames cannot prove motion or sound.
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
2. SUBJECT-FIRST IS REQUIRED:
   - Every caption must begin with the real visible subject in the first few words.
   - Wrong openers:
     "The macro lens..."
     "Deployment initiated..."
     "System scan..."
     "This video shows..."
     "The footage captures..."
     "In this clip..."
     "A scene of..."
     "A view of..."
3. ALWAYS name the subject with a clear visible attribute when available:
   "a tan dog", "an orange kitten", "a city road", "a young woman at a computer".
4. Include a few visual specifics: color, object, action, and setting.
5. BANNED (these lose points):
   - guessed place/city/country/region/landmark/brand names
   - place-style / lookalike labels:
     "Shibuya-style", "Tokyo-looking", "European-style", "South Asian-looking",
     "looks like X", "X-style", "X-looking"
   - guessed language/script/ethnicity/nationality/culture/region
   - sensitive appearance wording unless essential
   - exact numbers, IDs, signs, or text unless CONFIRMED and clearly legible
   - quoted on-screen text unless CONFIRMED
   - material or species guesses
   - resolution, fps, duration, "N-second clip"
   - audio/silence references
   - clothing inventories (at most ONE distinctive item)
6. If SOURCE B marks something RISKY, drop it completely.
7. Short correct caption beats long caption with one error.
"""

REGISTER_EXAMPLES = """
Register to match LENGTH and TONE only. Never copy content.

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
You are an expert caption writer. Two sources about ONE video.

SOURCE A:
{gemini_summary}

SOURCE B:
{qwen_report}

HOW TO RECONCILE:
- If B says WRONG, trust B for visual facts.
- If B says MISSING, add it if it fits naturally.
- If B says RISKY, DROP it entirely.
- Do not soften a risky claim; remove it.
- B cannot override A about motion unless B directly contradicts a visible fact.

{ACCURACY_RULES}

Write ONE caption per style:
{style_lines}

{REGISTER_EXAMPLES}

LENGTH:
- formal: 1-2 sentences, 25-35 words, concise
- sarcastic: ONE sentence, 10-20 words
- humorous_tech: 1-2 short lines, 15-25 words, subject first
- humorous_non_tech: ONE sentence, 10-20 words, ZERO technical words

FINAL SAFETY CHECK BEFORE OUTPUT:
- Remove any place-style / lookalike / X-style / X-looking phrases
- Remove any guessed language, script, ethnicity, nationality, culture, or region
- Remove sensitive appearance wording unless essential
- Remove any exact number or text unless clearly CONFIRMED
- Make sure each caption starts with the visible subject

OUTPUT: ONLY a raw JSON object with keys: {keys}
"""
