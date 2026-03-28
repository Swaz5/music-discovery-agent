# ── Stage 1: build wheels ─────────────────────────────────────────────────────
# Compile any C-extension wheels (numpy, librosa, chromadb…) in a layer that
# has build tools, then discard the compiler from the final image.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

# libsndfile1  — soundfile backend used by librosa for WAV/FLAC I/O
# ffmpeg       — audioread backend used by librosa for MP3 decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels — no compiler required at runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/*.whl \
 && rm -rf /wheels

# Copy project source and static data
COPY src/        ./src/
COPY data/       ./data/
COPY scripts/    ./scripts/
COPY pytest.ini  .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# These directories are bind-mounted at runtime (see docker-compose.yml).
# Creating them here ensures the paths exist even before the mount takes effect.
RUN mkdir -p .chroma data/previews

# Run as a non-root user.
# chown before switching so the user can write to mounted volumes.
RUN useradd --create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# entrypoint.sh seeds the knowledge base on first start, then launches uvicorn.
ENTRYPOINT ["./entrypoint.sh"]
