FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    google-genai>=2.11.0 \
    opencv-python-headless>=4.10 \
    requests>=2.31

RUN mkdir -p /output

COPY main.py prompts.py summarizer.py validator.py captioner.py /app/

WORKDIR /app

ARG GEMINI_API_KEY
ARG OPENROUTER_API_KEY
ARG ANTHROPIC_API_KEY

ENV GEMINI_API_KEY=$GEMINI_API_KEY
ENV OPENROUTER_API_KEY=$OPENROUTER_API_KEY
ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY

ENTRYPOINT ["python", "-u", "main.py"]
