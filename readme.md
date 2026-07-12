# EyeQ 3 — Video Caption Agent

Three AI models. One verified caption.

**Gemini** watches the video → **Qwen** validates against frames → **Claude** writes four styled captions.

## How it works

1. **Gemini 3.5 Flash** analyzes the full video and produces a scene-level summary — subject, action, setting, visual details, camera work.

2. **Qwen3-VL-235B** sees the actual frames without Gemini's summary. It checks what Gemini got wrong, what's missing, and flags any risky guesses (place names, brands, unreadable text).

3. **Claude Sonnet 4.6** merges both reports — trusting Qwen for visual corrections, dropping anything flagged as risky — then writes one caption per requested style: formal, sarcastic, humorous_tech, humorous_non_tech.

## What makes it different

Every other team trusts a single model. If it says "brown dog" when the dog is cream-colored, all four captions carry that error. EyeQ 3 catches it — Qwen sees the frames independently, flags the mismatch, and Claude uses the corrected version.

## Files

```
main.py        — video download, frame extraction, task loop
prompts.py     — all prompts and style definitions
summarizer.py  — Gemini video analysis
validator.py   — Qwen frame validation
captioner.py   — Claude caption writing
```

## Run

```bash
pip install google-genai opencv-python-headless requests
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
python main.py
```

Reads `/input/tasks.json`, writes `/output/results.json`.

## Docker

```
docker.io/sagar2652/caption-agent:latest
```

## Track 2 — AMD Developer Hackathon
