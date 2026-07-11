FROM python:3.11-slim

# ffmpeg/ffprobe for audio ground truth
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps — pinned for reproducibility, no cache for image size
RUN pip install --no-cache-dir \
    google-genai>=2.11.0 \
    opencv-python-headless>=4.10 \
    requests>=2.31 \
    scenedetect>=0.7

# Create the output directory (input is mounted by the harness)
RUN mkdir -p /output

# Copy the agent
COPY styled_caption_agent_v3.py /app/agent.py

WORKDIR /app

# The harness injects API keys as env vars:
#   GEMINI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, MOONSHOT_API_KEY

ENTRYPOINT ["python", "-u", "agent.py"]
