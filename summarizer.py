import time
from google import genai
from prompts import GEMINI_PROMPT

MODEL = "gemini-3.5-flash"
FALLBACK_MODEL = "gemini-3.1-pro-preview"
MEDIA_RES = "MEDIA_RESOLUTION_HIGH"


def create_client(api_key):
    return genai.Client(api_key=api_key)


def _get_config():
    try:
        from google.genai import types
        enum = getattr(types, "MediaResolution", None)
        if enum and hasattr(enum, MEDIA_RES):
            return types.GenerateContentConfig(
                media_resolution=getattr(enum, MEDIA_RES))
    except Exception:
        pass
    return None


def _generate(client, uploaded, prompt, model, config):
    if config:
        try:
            resp = client.models.generate_content(
                model=model, contents=[uploaded, prompt], config=config)
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception:
            pass
    resp = client.models.generate_content(
        model=model, contents=[uploaded, prompt])
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError(f"{model} returned empty response")
    return text


def summarize(client, video_path):
    uploaded = client.files.upload(file=video_path)
    try:
        wait_start = time.time()
        while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
            if time.time() - wait_start > 180:
                raise RuntimeError("Gemini processing timed out")
            time.sleep(4)
            uploaded = client.files.get(name=uploaded.name)

        if getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError("Gemini failed to process video")

        config = _get_config()

        try:
            return _generate(client, uploaded, GEMINI_PROMPT, MODEL, config)
        except Exception as e:
            print(f"[warn] {MODEL} failed: {e}, trying {FALLBACK_MODEL}")
            return _generate(client, uploaded, GEMINI_PROMPT, FALLBACK_MODEL, config)

    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
