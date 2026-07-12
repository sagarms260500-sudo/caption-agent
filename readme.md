# iSquare — Video Caption Agent

Three-model pipeline: **Gemini** summarizes → **Qwen** validates → **Claude** writes styled captions.

```
main.py        — orchestration
summarizer.py  — Gemini 3.5 Flash video analysis
validator.py   — Qwen3-VL frame validation
captioner.py   — Claude Sonnet 4.6 caption writing
prompts.py     — all prompts and style definitions
```

**Docker:** `docker.io/sagar2652/caption-agent:latest`

Reads `/input/tasks.json`, writes `/output/results.json`.
