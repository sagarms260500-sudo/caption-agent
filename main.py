import os
import sys
import json
import cv2
import shutil
import signal
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import summarizer
import validator
import captioner

TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

PARALLEL = 8
MAX_FRAMES = 12
FRAME_SIZE = 768
TIMEOUT = int(os.environ.get("HARD_TIMEOUT", "540"))

FALLBACK = "A short video clip."
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

import requests


def check_keys():
    for name, val in [("GEMINI_API_KEY", GEMINI_API_KEY),
                      ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
                      ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)]:
        if not val or set(val.lower()) <= {"x"}:
            raise ValueError(f"{name} is not set")


def download_video(url, dest):
    if not url.startswith("http"):
        if Path(url).is_file():
            return str(url)
        raise FileNotFoundError(f"no video: {url}")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest


def probe_audio(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "UNKNOWN"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=90)
        codecs = [x.strip() for x in r.stdout.splitlines() if x.strip()]
        if codecs:
            return f"AUDIO PRESENT ({', '.join(codecs)})"
        return "NO AUDIO TRACK - silent"
    except Exception:
        return "UNKNOWN"


def get_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps else 0, fps


def pick_timestamps(duration, fps, n=MAX_FRAMES):
    try:
        from scenedetect import detect, ContentDetector
        scenes = detect(str(duration), ContentDetector()) if False else []
    except Exception:
        scenes = []

    last = max(duration - 1.0 / fps, 0.0)
    if n <= 1 or last <= 0:
        return [0.0]
    return [round(i * last / (n - 1), 2) for i in range(n)]


def extract_frames(video_path, timestamps, out_dir):
    cap = cv2.VideoCapture(video_path)
    paths = []
    for i, t in enumerate(timestamps, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = FRAME_SIZE / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        p = os.path.join(out_dir, f"frame_{i}.jpg")
        cv2.imwrite(p, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(p)
    cap.release()
    return paths


def process_task(task, gemini_client):
    task_id = str(task.get("task_id", "unknown"))
    styles = task.get("styles") or STYLES
    print(f"\n===== TASK {task_id} =====")

    try:
        with tempfile.TemporaryDirectory() as td:
            video_path = download_video(task["video_url"],
                                        os.path.join(td, "video.mp4"))
            duration, fps = get_duration(video_path)
            audio = probe_audio(video_path)
            print(f"[{task_id}] {duration:.1f}s | {audio}")

            print(f"[{task_id}] Gemini summarize...")
            summary = summarizer.summarize(gemini_client, video_path, audio)
            print(f"\n--- [{task_id}] GEMINI ---\n{summary}")

            qwen_report = "Validation unavailable."
            try:
                print(f"[{task_id}] Qwen validate...")
                ts = pick_timestamps(duration, fps)
                frames = extract_frames(video_path, ts, td)
                if frames:
                    qwen_report = validator.validate(
                        OPENROUTER_API_KEY, frames, summary)
                    print(f"\n--- [{task_id}] QWEN ---\n{qwen_report}")
            except Exception as e:
                print(f"[warn][{task_id}] Qwen skipped: {e}")

        print(f"[{task_id}] Claude write captions...")
        captions = captioner.write_captions(
            ANTHROPIC_API_KEY, summary, qwen_report, audio, styles)
        print(f"\n--- [{task_id}] CAPTIONS ---")
        print(json.dumps(captions, ensure_ascii=False, indent=2))
        return {"task_id": task_id, "captions": captions}

    except Exception as e:
        print(f"[error][{task_id}] {e}")
        return {"task_id": task_id,
                "captions": {s: FALLBACK for s in styles}}


def write_results(results):
    path = Path(RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def main():
    print("=" * 50)
    print("VIDEO CAPTION AGENT")
    print("Gemini -> Qwen -> Claude")
    print("=" * 50)

    tasks = json.loads(Path(TASKS_PATH).read_text())
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must be a JSON array")
    print(f"Loaded {len(tasks)} task(s) | parallel: {PARALLEL}")

    check_keys()
    gemini_client = summarizer.create_client(GEMINI_API_KEY)
    print("Keys OK")

    results = [None] * len(tasks)
    lock = threading.Lock()
    done = {"n": 0}

    def run(i, task):
        task.setdefault("task_id", f"task_{i + 1}")
        try:
            res = process_task(task, gemini_client)
        except Exception as e:
            print(f"[error] {task.get('task_id')}: {e}")
            styles = task.get("styles") or STYLES
            res = {"task_id": str(task.get("task_id")),
                   "captions": {s: FALLBACK for s in styles}}
        with lock:
            results[i] = res
            done["n"] += 1
            write_results([r for r in results if r is not None])
            print(f"[checkpoint] {done['n']}/{len(tasks)} written")

    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        futures = [pool.submit(run, i, t) for i, t in enumerate(tasks)]
        for f in as_completed(futures):
            if f.exception():
                print(f"[error] worker: {f.exception()}")

    write_results([r for r in results if r is not None])
    print(f"\nDONE - {done['n']}/{len(tasks)} -> {RESULTS_PATH}")


if __name__ == "__main__":
    def _timeout(signum, frame):
        print("\n[TIMEOUT] exiting with current results")
        os._exit(0)

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(TIMEOUT)

    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[fatal] {e}")
        sys.exit(1)
