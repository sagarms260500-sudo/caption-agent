import os
import sys
import json
import time
import cv2
import signal
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import summarizer
import validator
import captioner

TASKS_PATH = os.environ.get("TASKS_PATH", "/input/tasks.json")
RESULTS_PATH = os.environ.get("RESULTS_PATH", "/output/results.json")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MAX_FRAMES = 12
FRAME_SIZE = 768
TIMEOUT = int(os.environ.get("HARD_TIMEOUT", "540"))
FALLBACK = "A short video clip."
STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


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


def get_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps else 0, fps


def extract_frames(video_path, duration, fps, out_dir):
    last = max(duration - 1.0 / fps, 0.0)
    if MAX_FRAMES <= 1 or last <= 0:
        timestamps = [0.0]
    else:
        timestamps = [round(i * last / (MAX_FRAMES - 1), 2)
                      for i in range(MAX_FRAMES)]
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
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        p = os.path.join(out_dir, f"frame_{i}.jpg")
        cv2.imwrite(p, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(p)
    cap.release()
    return paths


def process_task(task, gemini_client):
    task_id = str(task.get("task_id", "unknown"))
    styles = task.get("styles") or STYLES

    try:
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            video_path = download_video(task["video_url"],
                                        os.path.join(td, "video.mp4"))
            duration, fps = get_duration(video_path)
            dl = time.time() - t0

            t1 = time.time()
            summary = summarizer.summarize(gemini_client, video_path)
            gem = time.time() - t1

            t2 = time.time()
            qwen_report = "Validation unavailable."
            try:
                frames = extract_frames(video_path, duration, fps, td)
                if frames:
                    qwen_report = validator.validate(
                        OPENROUTER_API_KEY, frames, summary)
                qw = time.time() - t2
            except Exception as e:
                qw = time.time() - t2
                print(f"[{task_id}] Qwen FAIL: {e}")

        t3 = time.time()
        captions = captioner.write_captions(
            ANTHROPIC_API_KEY, summary, qwen_report, styles)
        cl = time.time() - t3

        total = dl + gem + qw + cl
        print(f"[{task_id}] dl:{dl:.0f}s gem:{gem:.0f}s qw:{qw:.0f}s cl:{cl:.0f}s = {total:.0f}s")
        return {"task_id": task_id, "captions": captions}

    except Exception as e:
        print(f"[{task_id}] FAIL: {e}")
        return {"task_id": task_id,
                "captions": {s: FALLBACK for s in styles}}


def write_results(results):
    path = Path(RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


_state = {"tasks": [], "results": [], "lock": threading.Lock()}


def main():
    print("=" * 50)
    print("DEBUG TIMING RUN")
    print("=" * 50)

    tasks = json.loads(Path(TASKS_PATH).read_text())
    print(f"Loaded {len(tasks)} task(s) | workers: 7")

    check_keys()
    gemini_client = summarizer.create_client(GEMINI_API_KEY)
    start = time.time()
    _state["tasks"] = tasks
    print("Keys OK\n")

    results = [None] * len(tasks)
    _state["results"] = results
    lock = _state["lock"]

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
            done = sum(1 for r in results if r is not None)
            write_results([r for r in results if r is not None])
            elapsed = time.time() - start
            print(f"[checkpoint] {done}/{len(tasks)} @ {elapsed:.0f}s")

    with ThreadPoolExecutor(max_workers=7) as pool:
        futs = [pool.submit(run, i, t) for i, t in enumerate(tasks)]
        for f in as_completed(futs):
            if f.exception():
                print(f"[error] {f.exception()}")

    for i, task in enumerate(tasks):
        if results[i] is None:
            tid = str(task.get("task_id", f"task_{i + 1}"))
            styles = task.get("styles") or STYLES
            results[i] = {"task_id": tid,
                          "captions": {s: FALLBACK for s in styles}}
            print(f"[rescue] {tid}")

    write_results(results)
    total = time.time() - start
    print(f"\nDONE {len(tasks)}/{len(tasks)} in {total:.0f}s")


if __name__ == "__main__":
    def _timeout(signum, frame):
        print("\n[TIMEOUT]")
        tasks = _state["tasks"]
        results = _state["results"]
        for i, task in enumerate(tasks):
            if results[i] is None:
                tid = str(task.get("task_id", f"task_{i + 1}"))
                styles = task.get("styles") or STYLES
                results[i] = {"task_id": tid,
                              "captions": {s: FALLBACK for s in styles}}
        write_results([r for r in results if r is not None])
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
