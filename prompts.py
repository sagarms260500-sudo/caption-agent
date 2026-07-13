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
STYLE_GUIDES = {
    "formal": (
        "ONE clean sentence (two only if genuinely needed). Subject with "
        "one visual attribute, its action, and the setting, fused into a "
        "single clause. No preamble, no 'the video shows', no listing of "
        "every detail. Neutral, descriptive, documentary tone.\n"
        "Example: 'A wide urban boulevard lined with golden ginkgo trees "
        "in full autumn foliage, with multiple lanes of traffic flowing "
        "through the city below high-rise residential buildings.'\n"
        "Example: 'A young orange tabby kitten sits among dense green "
        "foliage in an outdoor setting, looking directly at the camera "
        "with an alert and curious expression.'"
    ),
    "sarcastic": (
        "ONE sentence. Dry, deadpan, ironic — mock grandeur or mock "
        "understatement about what is actually happening. The joke comes "
        "from the gap between the flat tone and the mundane reality, not "
        "from exaggeration of the facts. It must still be possible to "
        "tell what is in the video from the sentence alone.\n"
        "Example: 'A kitten outdoors, clearly plotting something "
        "elaborate and fully confident it will succeed.'\n"
        "Example: 'A person at a computer, apparently working, which is "
        "exactly what someone would do if they were not working.'"
    ),
    "humorous_tech": (
        "ONE to two short lines. Build a small absurdist scenario from "
        "software/engineering/AI life (a bug, a deploy, an agent, a "
        "stack trace, a rollback, a system update) that maps onto what "
        "is visually happening, rather than literally describing the "
        "subject and then bolting on a metaphor. The subject should be "
        "recognizable through the analogy, not necessarily named first. "
        "Exactly ONE technical frame — do not mix, e.g., networking and "
        "robotics in the same caption.\n"
        "Example: 'She has been staring at this bug for forty minutes. "
        "The bug is a missing comma. The comma is winning.'\n"
        "Example: \"Nature's annual deployment: all leaf nodes updated "
        "to yellow simultaneously, no breaking changes reported.\"\n"
        "Example: 'A small autonomous agent has entered the garden "
        "environment and is scanning for input. Next action: unknown. "
        "Rollback plan: none.'"
    ),
    "humorous_non_tech": (
        "ONE sentence. Zero technical vocabulary of any kind (no code, "
        "server, deploy, render, pipeline, node, agent, system, bug, "
        "etc.). The humor comes from mildly exaggerating the stakes, "
        "effort, judgment, or importance of something mundane and "
        "visible in the frame — observational, relatable, slightly "
        "self-deprecating in spirit.\n"
        "Example: 'A tiny cat has gone outside and is now judging "
        "everything it sees with great authority.'\n"
        "Example: 'A woman at a computer, visibly handling something "
        "extremely important that will be completely forgotten by "
        "Thursday.'"
    ),
}

ACCURACY_RULES = """
ACCURACY RULES:
1. Every claim must come from the VERIFIED FACT SHEET. Never invent anything.
2. ALWAYS name or clearly imply the subject with its key visual attribute
   ("a tan dog", "an orange kitten", "a woman in a red jacket") — the
   reader must be able to identify the subject even through a joke or
   analogy.
3. Include only the 2–4 most visually distinctive details that are clearly
   visible throughout the clip. Prefer omission over uncertain
   specificity. Do not mention clothing, jewelry, text, numbers,
   advertisements, or appearance unless central to understanding the scene.
4. BANNED (these lose points):
   - Guessed place/city/landmark/brand names, including qualifiers
     like "X-style" or "resembling X"
   - Guessed language, script, ethnicity, or region of origin
   - Quoted on-screen text unless clearly legible in the fact sheet
   - Material or species guesses stated as fact
   - Resolution, fps, duration, "N-second clip"
   - Audio/silence references ("no audio track", "silent mode")
5. SUBJECT-IDENTIFIABLE: a reader must be able to tell what is actually in
   the video from the caption alone, even in sarcastic or humorous styles.
   Never open with a camera part, a tech term, or abstract framing that
   obscures the subject.
6. ACCURACY VS. STYLE: you are scored on two separate axes — accuracy
   (faithfulness to the video) and style match (how well the tone lands).
   A caption that nails the tone but misdescribes the scene, or one that
   is accurate but reads flat and generic, both lose points. When style
   and literal precision pull in different directions, keep the joke or
   tone but anchor it to a real, verified detail rather than dropping
   accuracy or dropping the humor — the reference examples all do both
   at once.
"""


def build_caption_prompt(gemini_summary, qwen_report, styles):
    style_lines = "\n".join(f'- "{s}": {STYLE_GUIDES[s]}' for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)

    return f"""
Return ONLY a raw JSON object. No preamble, no markdown fences, no
commentary before or after the JSON.

YOUR JOB

The VERIFIED FACT SHEET contains every verified observation.
It is intentionally more detailed than a caption.
Do NOT try to use every fact.
Your job is to write the BEST caption, not the MOST COMPLETE caption.
Think like a human editor, not a scene-inventory generator.
A good caption identifies the video using only the most important facts.
Leave out minor details that do not help identify the scene.

VERIFIED FACT SHEET (saw the full video):
{gemini_summary}

VISUAL VALIDATION (saw frames only, checked the fact sheet):
{qwen_report}

FACT SELECTION

Every caption MUST include:
1. Primary subject
2. Primary action
3. Setting

Then choose ONLY the 2 or 3 strongest distinguishing details.
Prefer details that uniquely identify the video.

GOOD details to keep (examples):
- bright yellow autumn trees
- rooftop terrace
- zebra crossing
- mountain ridges
- sunset reflections
- kitten emerging from bushes

Usually IGNORE:
- clothing colors, jewelry, nail color
- advertisements, train numbers, building names
- exact text, exact material names
- tiny background objects, small decorative items

unless they are essential for identifying the scene.

HOW TO USE THE FACT SHEET

Use the VERIFIED FACT SHEET as your source of truth.
Use VISUAL VALIDATION only to:
- correct facts
- add major missing facts
- remove risky facts

Never replace a removed fact with another guess.
Do not include every verified fact — choose only what makes the caption
accurate and distinctive.

All four captions MUST describe exactly the same video.
Do not introduce new facts for one style. Do not remove important facts
from another. Only the writing style changes.

{ACCURACY_RULES}

Write ONE caption per style. Study the tone and length of the examples
inside each style guide closely — match that register, not just the rule.
{style_lines}

FINAL EDITOR CHECK

Before returning JSON ask yourself:
✓ Did I write a caption instead of a scene inventory?
✓ Did I choose only the strongest identifying facts?
✓ Could any sentence become shorter without losing meaning?
✓ Did I remove unnecessary details?
✓ Is every remaining fact clearly supported?
✓ Does every style describe the identical video?
✓ Does each caption's tone actually match its style's reference examples,
  not just its rule description?
✓ Is the subject still identifiable through the joke, in the sarcastic
  and humorous styles?

If not, rewrite only that caption.

Return ONLY the JSON. No text before or after it, no markdown fences.
OUTPUT: a raw JSON object with keys: {keys}
"""w JSON object with keys: {keys}
"""
