STYLE_GUIDES = {
    "formal": (
        "Professional, objective, factual. One or two calm sentences. "
        "1-2 sentences, keep concise. START with the real visible subject. "
        "No jokes, no metadata."
    ),
    "sarcastic": (
        "Dry, ironic, lightly mocking but STILL ACCURATE. ONE sentence. "
        "START with the real visible subject. Deadpan understatement. "
        "No invented events, thoughts, locations, or text."
    ),
    "humorous_tech": (
        "Funny tech metaphor mapped onto real video content. ONE or two short lines. "
        "START with the real visible subject, not jargon. Do not start with "
        "'Deployment initiated', 'System', 'Protocol', 'The macro lens', or any abstract process."
    ),
    "humorous_non_tech": (
        "Funny everyday humour, ZERO technical words. ONE sentence. "
        "START with the real visible subject. Observational comedy only."
    ),
}


GEMINI_PROMPT = """
You are a careful video analyst. Watch this video and describe what is visibly there.

Rules:
- Name the main subject with obvious visual attributes, what it does, and where it is.
- Describe the scene directly. Do NOT use place-style or lookalike labels.
  BANNED examples: "Shibuya-style", "Tokyo-looking", "European-style",
  "South Asian-looking", "looks like New York", "looks like X".
- Do NOT guess place names, cities, countries, landmarks, monuments, brands, regions,
  cultures, ethnicities, nationalities, languages, scripts, species, breeds, or materials.
- Generic obvious categories are allowed when visually clear, such as "cat", "dog",
  "bird", "person", "car", "tree", "building", "road", "water", or "food".
- Do NOT say text is from a language, script, ethnicity, culture, country, or region.
  BANNED examples: "South Asian text", "Japanese signage", "Arabic script",
  "Korean street", "Chinese characters", "European storefront".
- Report on-screen text ONLY if clearly legible. Mark exact readable words as CONFIRMED.
  If text is present but not readable, say "unreadable text/signage".
- If something is unclear, say "unclear".
- Do not mention video metadata such as resolution, fps, duration, or clip length.
- Do not mention audio or silence.

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
   Examples: wrong color, wrong object, wrong clothing item, wrong setting, wrong visible text.

2. MISSING - OBVIOUS scene-level things a viewer would immediately notice.

3. RISKY - guesses rather than observations. Flag these for removal:
   - Named places, cities, countries, regions, landmarks, monuments, or brands identified
     from appearance.
     Examples: "Statue of Liberty", "Shibuya", "Tokyo", "New York", "Starbucks",
     "Prugio", "Korea", "India", "South Asia".
   - Place-style, region-style, culture-style, or lookalike labels.
     Examples: "Shibuya-style", "Tokyo-looking", "European-style", "South Asian-looking",
     "looks like Tokyo", "looks like New York", "X-style", "X-looking", "looks like X".
   - Guessed language, script, ethnicity, nationality, culture, or region.
     Examples: "South Asian text", "Japanese signage", "Arabic script",
     "Chinese characters", "Korean street", "Indian market", "European storefront",
     "African clothing", "Latino neighborhood".
   - Specific plant/animal species, breed, or material guesses beyond generic visible categories.
     Risky examples: "ginkgo trees", "oak wood", "marble counter", "tabby breed",
     "stainless steel", "linen shirt".
     Usually safe if visually obvious: "cat", "kitten", "dog", "bird", "tree",
     "flower", "road", "building", "car", "bus", "person".
   - On-screen text that is not exactly and clearly readable.
   - Any language/script/region claim about text, even if text is visible.
   - Inferred emotions, intentions, professions, or relationships not directly visible.
   - Audio, silence, resolution, fps, duration, or clip length.

For text:
- Only accept exact, clearly legible transcription.
- If text is blurry, partial, tiny, or uncertain, flag the text claim as RISKY.
- If the description names a language, script, country, ethnicity, culture, or region for text,
  flag it as RISKY even when signs are visible.

Be conservative. Still frames cannot prove most motion details or sound.
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
   - Start with what is actually visible: "A small orange kitten...", "A busy city street...",
     "A young woman at a computer...", "Ocean waves...", "A person chopping vegetables..."
   - Wrong openers:
     "The macro lens..."
     "Deployment initiated..."
     "System scan..."
     "This video shows..."
     "The footage captures..."
     "In this clip..."
     "A scene of..."
     "A view of..."
   - For humorous_tech, the tech joke must come AFTER the subject is named.
3. ALWAYS name the subject with a key visible attribute when available:
   "a tan dog", "an orange kitten", "a busy city street", "a young woman at a computer".
4. Include a few visual specifics: color, object, action, and setting.
5. BANNED because they lose accuracy points:
   - Guessed place, city, country, region, landmark, monument, or brand names.
   - Place-style, region-style, culture-style, or lookalike qualifiers:
     "Shibuya-style", "Tokyo-looking", "European-style", "South Asian-looking",
     "looks like Tokyo", "looks like New York", "X-style", "X-looking", "looks like X".
   - Guessed language, script, ethnicity, nationality, culture, or region:
     "South Asian text", "Japanese signage", "Arabic script", "Chinese characters",
     "Korean street", "Indian market", "European storefront", "African clothing".
   - Quoted on-screen text unless it is marked CONFIRMED in the FACT SHEET and not flagged RISKY.
   - Specific species, breed, or material guesses unless clearly confirmed.
     Avoid "ginkgo", "tabby", "marble", "linen", "stainless steel" unless confirmed.
     Generic obvious categories like "cat", "dog", "tree", "flower", "road", "building",
     "car", or "bus" are allowed when visible.
   - Resolution, fps, duration, "N-second clip", or video metadata.
   - Audio/silence references: "no audio track", "silent mode", "quiet clip".
   - Clothing inventories. Mention at most ONE distinctive clothing item if useful.
   - Inferred private thoughts, identities, professions, ethnicities, nationalities, or relationships.
6. If SOURCE B says RISKY, drop that claim completely instead of rewriting it.
7. Short correct caption beats a long caption with one error.
"""


REGISTER_EXAMPLES = """
Register to match LENGTH and TONE only. Never copy scene content.

formal: "A young orange cat sits among dense green foliage,
looking directly at the camera with an alert expression."

sarcastic: "A small cat outdoors looks deeply committed to supervising absolutely nothing."

humorous_tech: "A small cat steps through the garden like a tiny quality-control agent
checking every leaf for compliance."

humorous_non_tech: "A tiny cat has gone outside and is now judging everything it sees."
"""


def build_caption_prompt(gemini_summary, qwen_report, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)

    return f"""
You are an expert caption writer. Two sources describe ONE video.

SOURCE A - FACT SHEET FROM VIDEO:
{gemini_summary}

SOURCE B - FRAME CHECKER REPORT:
{qwen_report}

HOW TO RECONCILE:
- If SOURCE B says WRONG, trust SOURCE B for visual facts.
- If SOURCE B says MISSING, add that obvious visual detail if it fits naturally.
- If SOURCE B says RISKY, DROP that claim entirely.
- SOURCE B sees still frames, so it cannot override SOURCE A about clear motion or action
  unless SOURCE B directly contradicts a visible visual fact.
- Never preserve risky wording by softening it. Delete it.

{ACCURACY_RULES}

Write ONE caption per requested style:
{style_lines}

{REGISTER_EXAMPLES}

LENGTH:
- formal: 1-2 sentences, 25-35 words, concise, factual, no clothing lists.
- sarcastic: ONE sentence, 10-20 words, accurate and lightly mocking.
- humorous_tech: 1-2 short lines, 15-25 words, subject first, tech joke second.
- humorous_non_tech: ONE sentence, 10-20 words, ZERO technical words.

SUBJECT-FIRST HARD RULE:
- Every caption must name the real visible subject in the first few words.
- Do NOT start with "The macro lens", "Deployment initiated", "System", "Protocol",
  "This video", "The footage", "In this clip", "A scene", or "A view".
- The subject must be a visible thing, person, animal, place, or action from the fact sheet.

FINAL SAFETY CHECK BEFORE OUTPUT:
- Remove all guessed places, regions, languages, scripts, ethnicities, nationalities,
  cultures, brands, landmarks, and X-style/X-looking phrases.
- Remove all unconfirmed on-screen text.
- Remove all audio, duration, fps, and resolution references.
- Make sure each caption starts with the visible subject.

OUTPUT: ONLY a raw JSON object with keys: {keys}
"""
