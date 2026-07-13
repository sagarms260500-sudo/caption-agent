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
Return ONLY the following format.

========================
VERIFIED FACT SHEET
========================

PRIMARY SUBJECT
- Main visible subject with 1–3 distinctive visual attributes.

SECONDARY SUBJECTS
- Other important people, animals, or objects.
- Write "None" if there are none.

ACTION
- Main actions occurring in the video.
- Use short bullet points.

SETTING
- Describe only the visible environment.
- Never guess city, country, landmark, or region.

IMPORTANT OBJECTS
- List only major objects that affect the scene.

DISTINCTIVE VISUAL DETAILS
- Colors
- Lighting
- Weather
- Camera angle
- Motion
- Anything visually important

VISIBLE TEXT
- Only clearly readable text.
- Otherwise write "None".

UNCERTAIN OBSERVATIONS
- Anything that might be incorrect.
- Write "None" if fully confident.
"""

QWEN_PROMPT = """You are an independent visual verifier.

You receive:

1. A VERIFIED FACT SHEET.
2. {n} video frames.

FACT SHEET

{summary}

Your job is NOT to rewrite the fact sheet.

Your job is ONLY to identify mistakes.

Check:

1. WRONG
- incorrect objects
- incorrect colors
- incorrect actions
- incorrect text
- incorrect setting
- incorrect numbers

2. MISSING
- obvious scene-level facts missing from the fact sheet

3. RISKY
Anything that appears to be a guess, including:

- place names
- cities
- countries
- landmarks
- brands
- occupations
- identities
- relationships
- ethnicity
- nationality
- language
- script
- species
- materials
- unreadable text
- exact numbers that are unclear

Also verify consistency:

- Did the subject change?
- Were new objects invented?
- Were important objects omitted?

If yes, report under WRONG.

Rules

• Never rewrite the fact sheet.

• Keep each section under five bullets.

• Ignore insignificant details.

Output ONLY

WRONG:
...

MISSING:
...

RISKY:
...
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

VERIFIED FACT SHEET

{gemini_summary}

QWEN VALIDATION

{qwen_report}

HOW TO USE BOTH SOURCES

The VERIFIED FACT SHEET contains the visual observations.
QWEN independently verified those observations.

If QWEN marks something WRONG,
correct it.

If QWEN marks something MISSING,
include it naturally.

If QWEN marks something RISKY,
remove it completely.

Never invent replacement facts.

Never introduce new objects.

Never introduce new actions.

All four captions must describe exactly the same video.
Only the writing style changes.

{ACCURACY_RULES}

Write ONE caption per style:
{style_lines}

{REGISTER_EXAMPLES}
STYLE CONSISTENCY

Every caption MUST describe the identical video.
Do not add new objects in one style.
Do not remove major objects in another style.
Do not change actions between styles.
Only the writing style changes.

LENGTH:
- formal: 1-2 sentences, 25-35 words, concise
- sarcastic: ONE sentence, 10-20 words
- humorous_tech: 1-2 short lines, 15-25 words, subject first
- humorous_non_tech: ONE sentence, 10-20 words, ZERO technical words

FINAL VALIDATION

Before returning JSON verify every caption:

✓ Starts with the visible subject.
✓ Uses only facts from the VERIFIED FACT SHEET.
✓ Applies every QWEN correction.
✓ Contains no RISKY items.
✓ Contains no guessed places.
✓ Contains no guessed brands.
✓ Contains no guessed occupations.
✓ Contains no guessed identities.
✓ Contains no guessed language or script.
✓ Contains no guessed species.
✓ Contains no metadata.
✓ Contains no audio references.
✓ Uses the requested writing style.
✓ Describes exactly the same scene as the other three captions.

If any check fails,
rewrite ONLY that caption.
Return ONLY the JSON object.

OUTPUT: ONLY a raw JSON object with keys: {keys}
"""
