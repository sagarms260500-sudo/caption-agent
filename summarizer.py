import time
from google import genai
from prompts import GEMINI_PROMPT

MODEL = "gemini-3.5-flash"
MEDIA_RES = "MEDIA_RESOLUTION_HIGH"


def create_client(api_key):
    return genai.Client(api_key=api_key)


def summarize(client, video_path):
    uploaded = client.files.upload(file=video_path)
    try:
        while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
            time.sleep(4)
            uploaded = client.files.get(name=uploaded.name)

        if getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError("Gemini failed to process video")

        prompt = GEMINI_PROMPT

        config = None
        try:
            from google.genai import types
            enum = getattr(types, "MediaResolution", None)
            if enum and hasattr(enum, MEDIA_RES):
                config = types.GenerateContentConfig(
                    media_resolution=getattr(enum, MEDIA_RES))
        except Exception:
            pass

        if config:
            try:
                resp = client.models.generate_content(
                    model=MODEL, contents=[uploaded, prompt], config=config)
                return (resp.text or "").strip()
            except Exception:
                pass

        resp = client.models.generate_content(
            model=MODEL, contents=[uploaded, prompt])
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty response")
        return text

    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
