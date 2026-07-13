import re
import base64
import requests
from prompts import QWEN_PROMPT

MODEL = "qwen/qwen3-vl-235b-a22b-thinking"
URL = "https://openrouter.ai/api/v1/chat/completions"

THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _clean(text):
    return THINK_RE.sub("", text or "").strip()


def _parse_response(data):
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        print("[warn] Qwen output truncated")
    msg = choice.get("message", {}) or {}
    for field in ("content", "reasoning_content", "reasoning"):
        text = _clean(msg.get(field) or "")
        if text:
            return text
    raise RuntimeError("empty Qwen response")


def validate(api_key, frame_paths, gemini_summary):
    prompt = QWEN_PROMPT.format(n=len(frame_paths), summary=gemini_summary)

    content = [{"type": "text", "text": prompt}]
    for path in frame_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": MODEL,
        "max_tokens": 1500,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(2):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=240)
            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
            return _parse_response(r.json())
        except Exception as e:
            if "HTTP 400" in str(e) or "HTTP 401" in str(e):
                raise
            if attempt == 1:
                raise
            print(f"[retry] Qwen: {e}")

    return "Validation unavailable."
