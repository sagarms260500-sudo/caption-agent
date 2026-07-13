import json
import re
import requests
from prompts import build_caption_prompt

MODEL = "claude-sonnet-4-6"
URL = "https://api.anthropic.com/v1/messages"
VERSION = "2023-06-01"

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def _extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(),
                  flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON in response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")


def _call_claude(api_key, prompt):
    payload = {
        "model": MODEL,
        "max_tokens": 1500,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": VERSION,
        "content-type": "application/json"
    }
    r = requests.post(URL, headers=headers, json=payload, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"Claude HTTP {r.status_code}: {r.text[:300]}")
    blocks = r.json().get("content", [])
    text = "".join(b.get("text", "") for b in blocks
                  if b.get("type") == "text").strip()
    if not text:
        raise RuntimeError("empty Claude response")
    return text


def write_captions(api_key, gemini_summary, qwen_report, styles):
    prompt = build_caption_prompt(gemini_summary, qwen_report, styles)
    raw = _call_claude(api_key, prompt)
    obj = _extract_json(raw)
    captions = {}
    for s in styles:
        val = obj.get(s)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"missing '{s}'")
        captions[s] = " ".join(val.split())
    return captions
