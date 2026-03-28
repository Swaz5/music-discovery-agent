#!/bin/sh
# entrypoint.sh — seed the RAG knowledge base, then start the API server.
#
# Ingestion is idempotent (doc IDs are content hashes), so re-running after a
# container restart is safe and fast — it only fetches what is missing.
#
# The knowledge base lives in the .chroma/ volume, so data persists across
# container restarts without re-fetching from Last.fm / Deezer.

set -e

echo "[entrypoint] Seeding RAG knowledge base..."
python -m src.rag.ingest || {
    echo "[entrypoint] WARNING: ingestion failed or was skipped — continuing with existing KB."
}

echo "[entrypoint] Starting API server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
