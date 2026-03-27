"""
FastAPI application entry point.

Creates the app, registers middleware, and mounts the API router.
Run directly with:  python -m src.api.main
Or via uvicorn:     uvicorn src.api.main:app --reload
"""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Music Discovery Agent API",
    description=(
        "An agentic music discovery service powered by Claude. "
        "Submit a natural-language vibe description and receive curated "
        "artist and track recommendations drawn from Last.fm, Deezer, "
        "audio analysis, and a RAG knowledge base. "
        "Rate artists to build a personal taste profile that shapes future recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow all origins for dev) ─────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────────────────────


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ── Router ────────────────────────────────────────────────────────────────────

app.include_router(router)

# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/", tags=["health"])
async def health_check():
    """Service liveness check."""
    return {"status": "ok", "service": "music-discovery-agent"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
