"""
FastAPI route handlers for the music discovery API.

Endpoints
---------
POST /discover          Run the discovery agent (supports SSE streaming)
GET  /artist/{name}     Full artist profile
POST /explore           Genre exploration with knowledge base context
POST /rate              Save a preference rating
GET  /taste-profile     Current taste profile
GET  /history           Session recommendation history
POST /analyze-taste     Taste analysis + blind-spot discovery via Claude
POST /bridge            Genre bridge: BFS similarity graph + Claude explanation
"""

import asyncio
import json
import logging
import math
import re
import time
from typing import AsyncIterator

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agent.discovery_agent import MusicDiscoveryAgent
from src.agent.preference_engine import PreferenceEngine
from src.data import lastfm_client as lastfm
from src.data import music_service
from src.rag.retriever import MusicRetriever

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy singletons ───────────────────────────────────────────────────────────
# Instantiated on first request so startup doesn't block or fail on missing env.

_prefs: PreferenceEngine | None = None
_agent: MusicDiscoveryAgent | None = None
_retriever: MusicRetriever | None = None

_history: list[dict] = []
_MAX_HISTORY = 100


def _get_prefs() -> PreferenceEngine:
    global _prefs
    if _prefs is None:
        _prefs = PreferenceEngine()
    return _prefs


def _get_agent() -> MusicDiscoveryAgent:
    global _agent
    if _agent is None:
        _agent = MusicDiscoveryAgent(preference_engine=_get_prefs())
    return _agent


def _get_retriever() -> MusicRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MusicRetriever()
    return _retriever


# ── Pydantic models ───────────────────────────────────────────────────────────


class DiscoverRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language vibe description")
    include_reasoning: bool = Field(False, description="Include agent reasoning trace")
    stream: bool = Field(False, description="Stream response as Server-Sent Events")


class RecommendationItem(BaseModel):
    artist: str
    track: str
    genre_tags: list[str]
    why: str
    vibe_description: str
    vibe_match_score: float
    deezer_url: str


class DiscoverResponse(BaseModel):
    query: str
    recommendations: list[RecommendationItem]
    reasoning_trace: list[dict] | None = None
    iterations: int
    total_tokens: int


class ExploreRequest(BaseModel):
    genre: str = Field(..., min_length=1, description="Genre name to explore")


class ExploreResponse(BaseModel):
    genre: str
    knowledge_context: str
    top_artists: list[dict]
    related_genres: list[str]


class RateRequest(BaseModel):
    artist: str = Field(..., min_length=1)
    liked: bool
    notes: str = ""


class RateResponse(BaseModel):
    message: str
    taste_profile: dict


class HistoryItem(BaseModel):
    query: str
    recommendations: list[dict]
    iterations: int
    total_tokens: int
    timestamp: float


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class AnalyzeTasteRequest(BaseModel):
    artists: list[str] = Field(..., min_length=1, max_length=20)
    discover_blind_spots: bool = True


class BlindSpot(BaseModel):
    genre: str
    why: str
    try_this: str


class OutlierInfo(BaseModel):
    artist: str
    why_different: str


class AnalyzeTasteResponse(BaseModel):
    taste_identity: str
    analysis: str
    common_threads: list[str]
    audio_profile: dict
    blind_spots: list[BlindSpot]
    outlier: OutlierInfo | None


class BridgeRequest(BaseModel):
    genre_a: str = Field(..., min_length=1)
    genre_b: str = Field(..., min_length=1)
    max_hops: int = Field(3, ge=1, le=5)


class BridgeArtist(BaseModel):
    name: str
    connects_because: str
    genres: list[str]


class TransitionTrack(BaseModel):
    track: str
    artist: str
    position: str


class BridgeResponse(BaseModel):
    genre_a: str
    genre_b: str
    bridge_artists: list[BridgeArtist]
    transition_playlist: list[TransitionTrack]
    explanation: str


# ── SSE helper ────────────────────────────────────────────────────────────────


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


# ── History helper ────────────────────────────────────────────────────────────


def _record_history(query: str, result: dict) -> None:
    global _history
    _history.append({
        "query": query,
        "recommendations": result["recommendations"],
        "iterations": result["iterations"],
        "total_tokens": result["total_tokens"],
        "timestamp": time.time(),
    })
    if len(_history) > _MAX_HISTORY:
        _history = _history[-_MAX_HISTORY:]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/discover", response_model=DiscoverResponse)
async def discover(req: DiscoverRequest):
    """
    Run the music discovery agent for a natural-language vibe query.

    When **stream=true** the response is a `text/event-stream` of JSON objects.
    Each event has a `type` field:

    - `start` — agent has begun processing
    - `reasoning_step` — one tool call from the agent's reasoning trace
    - `recommendations` — final list of recommendations
    - `done` — stream complete with token/iteration counts
    - `error` — unrecoverable failure
    """
    if req.stream:
        return StreamingResponse(
            _stream_discovery(req.query, req.include_reasoning),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await _get_agent().discover(req.query)
    except Exception as exc:
        logger.exception("Discovery failed for query %r", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    _record_history(req.query, result)

    return DiscoverResponse(
        query=req.query,
        recommendations=result["recommendations"],
        reasoning_trace=result["reasoning_trace"] if req.include_reasoning else None,
        iterations=result["iterations"],
        total_tokens=result["total_tokens"],
    )


async def _stream_discovery(
    query: str, include_reasoning: bool
) -> AsyncIterator[str]:
    """
    Async generator that drives the agent and emits SSE events.

    Because the agent's agentic loop runs to completion before returning,
    the reasoning trace is replayed as a stream of events after the run
    finishes — giving callers progressive visibility into what the agent did.
    """
    yield _sse({"type": "start", "query": query})

    try:
        result = await _get_agent().discover(query)
    except Exception as exc:
        logger.exception("Discovery stream failed for query %r", query)
        yield _sse({"type": "error", "message": str(exc)})
        return

    _record_history(query, result)

    if include_reasoning:
        for step in result["reasoning_trace"]:
            yield _sse({"type": "reasoning_step", "data": step})
            await asyncio.sleep(0.02)  # allow the event loop to flush

    yield _sse({"type": "recommendations", "data": result["recommendations"]})
    yield _sse({
        "type": "done",
        "iterations": result["iterations"],
        "total_tokens": result["total_tokens"],
    })


@router.get("/artist/{name}")
async def get_artist(name: str):
    """
    Return full artist profile: bio, tags, similar artists, top tracks,
    averaged audio features, and social stats.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Artist name cannot be empty")

    try:
        profile = await music_service.get_full_artist_profile(name)
    except Exception as exc:
        logger.exception("Failed to fetch artist profile for %r", name)
        raise HTTPException(status_code=500, detail=str(exc))

    # music_service always returns a dict with name set (graceful degradation),
    # so we detect "not found" by checking whether all meaningful fields are
    # empty/zero — the signature of an artist that exists in neither Last.fm
    # nor Deezer.
    not_found = (
        not profile
        or (
            not profile.get("bio")
            and not profile.get("tags")
            and not profile.get("listeners")
            and not profile.get("fan_count")
        )
    )
    if not_found:
        raise HTTPException(status_code=404, detail=f"Artist not found: {name!r}")

    return profile


@router.post("/explore", response_model=ExploreResponse)
async def explore_genre(req: ExploreRequest):
    """
    Explore a genre: returns knowledge base context, Last.fm top artists,
    and related genres inferred from the top artists' own genre tags.
    """
    genre = req.genre.strip()
    if not genre:
        raise HTTPException(status_code=400, detail="Genre cannot be empty")

    try:
        # Fetch KB context and top artists concurrently
        kb_context, top_artists = await asyncio.gather(
            asyncio.to_thread(_get_retriever().retrieve_for_genre, genre),
            lastfm.get_tag_top_artists(genre, limit=10),
        )
    except Exception as exc:
        logger.exception("Genre exploration failed for %r", genre)
        raise HTTPException(status_code=500, detail=str(exc))

    # Fetch tags for the top 5 artists in parallel to surface related genres
    related_genres: list[str] = []
    try:
        tag_lists = await asyncio.gather(
            *[lastfm.get_artist_tags(a["name"]) for a in top_artists[:5]],
            return_exceptions=True,
        )
        tag_counts: dict[str, int] = {}
        for tags in tag_lists:
            if isinstance(tags, Exception):
                continue
            for tag in tags:
                tl = tag.lower()
                if tl != genre.lower():
                    tag_counts[tl] = tag_counts.get(tl, 0) + 1
        related_genres = sorted(tag_counts, key=lambda t: -tag_counts[t])[:8]
    except Exception:
        pass  # related genres are best-effort

    return ExploreResponse(
        genre=genre,
        knowledge_context=kb_context,
        top_artists=top_artists,
        related_genres=related_genres,
    )


@router.post("/rate", response_model=RateResponse)
async def rate_artist(req: RateRequest):
    """
    Save a preference rating for an artist.

    The preference engine fetches audio features automatically if not supplied.
    Returns the updated taste profile summary.
    """
    artist = req.artist.strip()
    if not artist:
        raise HTTPException(status_code=400, detail="Artist name cannot be empty")

    try:
        await _get_prefs().save_preference(artist, liked=req.liked, notes=req.notes)
    except Exception as exc:
        logger.exception("Failed to save preference for %r", artist)
        raise HTTPException(status_code=500, detail=str(exc))

    sentiment = "liked" if req.liked else "disliked"
    return RateResponse(
        message=f"Saved: {artist} → {sentiment}",
        taste_profile=_get_prefs().get_taste_profile(),
    )


@router.get("/taste-profile")
async def get_taste_profile():
    """
    Return the current taste profile derived from all stored ratings.

    Keys: preferred_tags, preferred_audio, disliked_tags, disliked_audio,
    sweet_spots, total_ratings, liked_count, disliked_count, summary.
    """
    return _get_prefs().get_taste_profile()


@router.get("/history", response_model=HistoryResponse)
async def get_history():
    """Return all discovery queries made during the current server session."""
    items = [
        HistoryItem(
            query=h["query"],
            recommendations=h["recommendations"],
            iterations=h["iterations"],
            total_tokens=h["total_tokens"],
            timestamp=h["timestamp"],
        )
        for h in _history
    ]
    return HistoryResponse(items=items, total=len(items))


@router.post("/analyze-taste", response_model=AnalyzeTasteResponse)
async def analyze_taste(req: AnalyzeTasteRequest):
    """
    Analyze a listener's taste from a list of artists they love.

    Fetches full profiles for every artist in parallel, computes aggregate
    audio statistics (averages, variance, outliers, top tags), then asks
    Claude to identify patterns, name the taste identity, and — when
    **discover_blind_spots=true** — suggest unexplored genres the user
    would likely enjoy.
    """
    artists = [a.strip() for a in req.artists if a.strip()]
    if not artists:
        raise HTTPException(status_code=422, detail="artists list must not be empty")

    # 1 ── Fetch all profiles concurrently ────────────────────────────────────
    raw_results = await asyncio.gather(
        *[music_service.get_full_artist_profile(a) for a in artists],
        return_exceptions=True,
    )

    profiles: list[dict] = []
    for artist, result in zip(artists, raw_results):
        if isinstance(result, Exception):
            logger.warning("Could not fetch profile for %r: %s", artist, result)
        elif result and result.get("name"):
            profiles.append(result)

    if not profiles:
        raise HTTPException(
            status_code=422,
            detail="Could not fetch profiles for any of the listed artists",
        )

    # 2 ── Compute aggregate statistics ───────────────────────────────────────
    stats = _compute_taste_stats(profiles)

    # 3 ── Ask Claude to interpret the data ───────────────────────────────────
    try:
        return await _claude_taste_analysis(profiles, stats, req.discover_blind_spots)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Taste analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Taste analysis helpers ────────────────────────────────────────────────────

# The audio_profile dict from music_service uses "avg_" prefixes; map them to
# bare names for consistent downstream use.
_AUDIO_KEYS = ("energy", "danceability", "valence", "tempo", "acousticness")


def _compute_taste_stats(profiles: list[dict]) -> dict:
    """
    Compute aggregate statistics across artist profiles.

    Returns:
        avg_features   — mean value per audio feature
        variance       — population variance per feature
        top_tags       — most-frequent tags across all artists
        tag_counts     — raw {tag: count} mapping
        outlier_name   — artist whose audio profile deviates most from the mean
        outlier_dist   — Euclidean distance of the outlier from the group centroid
    """
    # ── Collect feature vectors ───────────────────────────────────────────────
    vectors: dict[str, list[float]] = {k: [] for k in _AUDIO_KEYS}
    artist_vectors: dict[str, dict[str, float]] = {}

    for p in profiles:
        audio = p.get("audio_profile", {})
        name = p.get("name", "")
        av: dict[str, float] = {}
        for key in _AUDIO_KEYS:
            v = audio.get(f"avg_{key}") or audio.get(key)
            if v is not None:
                fv = float(v)
                vectors[key].append(fv)
                av[key] = fv
        if av:
            artist_vectors[name] = av

    # ── Averages ──────────────────────────────────────────────────────────────
    avg_features: dict[str, float] = {
        k: round(sum(v) / len(v), 3)
        for k, v in vectors.items()
        if v
    }

    # ── Variance ──────────────────────────────────────────────────────────────
    variance: dict[str, float] = {}
    for k, v in vectors.items():
        if len(v) >= 2:
            mean = avg_features[k]
            variance[k] = round(sum((x - mean) ** 2 for x in v) / len(v), 4)

    # ── Tag frequency ─────────────────────────────────────────────────────────
    tag_counts: dict[str, int] = {}
    for p in profiles:
        for tag in p.get("tags", []):
            tl = tag.lower()
            tag_counts[tl] = tag_counts.get(tl, 0) + 1
    top_tags = sorted(tag_counts, key=lambda t: -tag_counts[t])[:15]

    # ── Outlier (max Euclidean distance from centroid) ────────────────────────
    outlier_name: str = ""
    outlier_dist = 0.0
    shared_keys = list(avg_features.keys())

    for name, av in artist_vectors.items():
        dist = math.sqrt(
            sum((av[k] - avg_features[k]) ** 2 for k in shared_keys if k in av)
        )
        if dist > outlier_dist:
            outlier_dist = dist
            outlier_name = name

    return {
        "avg_features": avg_features,
        "variance": variance,
        "top_tags": top_tags,
        "tag_counts": tag_counts,
        "outlier_name": outlier_name,
        "outlier_dist": round(outlier_dist, 3),
    }


_CLAUDE_MODEL = "claude-sonnet-4-20250514"


async def _claude_taste_analysis(
    profiles: list[dict],
    stats: dict,
    discover_blind_spots: bool,
) -> AnalyzeTasteResponse:
    """Call Claude with the aggregated taste data and parse its JSON response."""

    # ── Build artist summaries for the prompt ─────────────────────────────────
    summaries: list[str] = []
    for p in profiles:
        audio = p.get("audio_profile", {})
        tags = ", ".join(p.get("tags", [])[:6]) or "—"
        feature_parts = []
        for k in _AUDIO_KEYS:
            v = audio.get(f"avg_{k}") or audio.get(k)
            if v is not None:
                fmt = f"{float(v):.0f} BPM" if k == "tempo" else f"{float(v):.2f}"
                feature_parts.append(f"{k}={fmt}")
        features_str = "  ".join(feature_parts)
        summaries.append(f"• {p['name']}: [{tags}]  {features_str}")

    # ── Variance interpretation (high = eclectic) ─────────────────────────────
    eclectic_note = ""
    if stats["variance"]:
        high_var = [
            f"{k} (σ²={v})"
            for k, v in stats["variance"].items()
            if v > 0.02
        ]
        if high_var:
            eclectic_note = f"High variance in: {', '.join(high_var)} — suggests eclectic taste."

    blind_spot_instruction = (
        "Suggest 3–4 blind spots: genres/artists the listener would likely love "
        "but probably hasn't discovered, given the patterns above. Each needs a "
        "specific artist/track recommendation (try_this field)."
        if discover_blind_spots
        else "Set blind_spots to an empty array []."
    )

    prompt = f"""\
You are an expert music analyst. A listener loves these artists:

{chr(10).join(summaries)}

Aggregate data across their taste:
  Average audio features: {json.dumps(stats['avg_features'])}
  Feature variance:       {json.dumps(stats['variance'])}
  Most common tags:       {', '.join(stats['top_tags'][:10])}
  Outlier artist:         {stats['outlier_name'] or 'none identified'}
  {eclectic_note}

Tasks:
1. Identify the connective thread — what unites these seemingly different artists?
2. Give this listener a "taste identity" label (2–4 evocative words, e.g. "Melancholic Texturalist").
3. Write a 2–3 paragraph analysis of their patterns: sonic tendencies, emotional terrain, what the audio numbers reveal.
4. List 3–5 concise common threads as short phrases.
5. {blind_spot_instruction}
6. Explain why the outlier artist is different from the rest.

Respond with ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "taste_identity": "...",
  "analysis": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3",
  "common_threads": ["...", "..."],
  "blind_spots": [
    {{"genre": "...", "why": "...", "try_this": "Artist — Track"}}
  ],
  "outlier": {{"artist": "...", "why_different": "..."}}
}}"""

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON: %s\n%.400s", exc, raw)
        raise HTTPException(status_code=500, detail="Failed to parse analysis from Claude")

    blind_spots = [
        BlindSpot(
            genre=bs.get("genre", ""),
            why=bs.get("why", ""),
            try_this=bs.get("try_this", ""),
        )
        for bs in data.get("blind_spots", [])
    ] if discover_blind_spots else []

    outlier_raw = data.get("outlier")
    outlier = OutlierInfo(
        artist=outlier_raw.get("artist", stats["outlier_name"]),
        why_different=outlier_raw.get("why_different", ""),
    ) if outlier_raw else None

    return AnalyzeTasteResponse(
        taste_identity=data.get("taste_identity", ""),
        analysis=data.get("analysis", ""),
        common_threads=data.get("common_threads", []),
        audio_profile=stats["avg_features"],
        blind_spots=blind_spots,
        outlier=outlier,
    )


# ── /bridge endpoint ──────────────────────────────────────────────────────────


@router.post("/bridge", response_model=BridgeResponse)
async def find_genre_bridge(req: BridgeRequest):
    """
    Discover the musical bridge between two genres.

    Runs a BFS through the Last.fm artist-similarity graph starting from the
    top artists of **genre_a** and searching for paths that reach the top artists
    of **genre_b**. Each similarity edge is one "hop"; **max_hops** controls how
    deep the search goes (1 = direct neighbours only, 3 = three steps deep).

    The found paths, seed artists, and RAG knowledge base context for both genres
    are passed to Claude, which:
    - Selects the most musically meaningful bridge artists and explains *why* each
      one sits at the intersection
    - Builds a transition playlist that moves smoothly from one genre to the other
    - Writes a paragraph explaining the historical/sonic connection

    The endpoint is useful even when the graph finds no paths — Claude draws on
    its training knowledge grounded by the knowledge base context.
    """
    genre_a = req.genre_a.strip().lower()
    genre_b = req.genre_b.strip().lower()

    if genre_a == genre_b:
        raise HTTPException(status_code=422, detail="genre_a and genre_b must be different")

    try:
        graph_data = await _bfs_genre_bridge(genre_a, genre_b, req.max_hops)
    except Exception as exc:
        logger.exception("Bridge graph search failed (%r → %r)", genre_a, genre_b)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        return await _claude_bridge_analysis(genre_a, genre_b, graph_data)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Bridge Claude analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Bridge helpers ────────────────────────────────────────────────────────────


async def _bfs_genre_bridge(genre_a: str, genre_b: str, max_hops: int) -> dict:
    """
    BFS through the Last.fm similarity graph from genre_a seeds to genre_b seeds.

    Algorithm
    ---------
    1. Fetch top-8 artists for both genres and KB context for both — all in one
       asyncio.gather to minimise latency.
    2. Build ``names_b``: the lowercase name set of genre_b seeds.  This is our
       "target" set — reaching any artist in it counts as finding a bridge.
    3. Check hop-0: artists that appear in *both* seed sets are immediate bridges.
    4. BFS loop (max_hops iterations):
       a. Expand the current frontier (capped to avoid combinatorial explosion).
       b. Fetch similar artists for all frontier artists in parallel.
       c. Any similar artist whose lowercase name is in ``names_b`` → new path found.
       d. Unseen similar artists become the next frontier.
    5. After BFS, fetch tags for all artists that appear in found paths (excluding
       genre_a seeds whose genre is already known) so Claude can explain their role.

    Returns a dict with seeds, found paths, artist tags, and KB context.
    """
    # ── Stage 1: seeds + KB context ───────────────────────────────────────────
    raw = await asyncio.gather(
        lastfm.get_tag_top_artists(genre_a, limit=8),
        lastfm.get_tag_top_artists(genre_b, limit=8),
        asyncio.to_thread(_get_retriever().retrieve_for_genre, genre_a),
        asyncio.to_thread(_get_retriever().retrieve_for_genre, genre_b),
        return_exceptions=True,
    )

    # Seeds are required; KB context degrades gracefully
    if isinstance(raw[0], Exception):
        raise raw[0]
    if isinstance(raw[1], Exception):
        raise raw[1]

    seeds_a: list[dict] = raw[0]
    seeds_b: list[dict] = raw[1]
    kb_a: str = raw[2] if not isinstance(raw[2], Exception) else ""
    kb_b: str = raw[3] if not isinstance(raw[3], Exception) else ""

    names_a_lower: set[str] = {s["name"].lower() for s in seeds_a}
    names_b_lower: dict[str, str] = {s["name"].lower(): s["name"] for s in seeds_b}

    # path_map: artist_name_lower → full path of canonical names leading here
    path_map: dict[str, list[str]] = {
        s["name"].lower(): [s["name"]] for s in seeds_a
    }

    # ── Hop 0: direct overlap ─────────────────────────────────────────────────
    found_paths: list[list[str]] = []
    for lower in names_a_lower & set(names_b_lower):
        found_paths.append(path_map[lower])

    # ── BFS ───────────────────────────────────────────────────────────────────
    # frontier: list of (canonical_name, name_lower)
    frontier: list[tuple[str, str]] = [
        (s["name"], s["name"].lower()) for s in seeds_a
    ]
    visited: set[str] = set(names_a_lower)

    # Shrink frontier at each hop to bound API calls:
    # hop 1 → expand up to 8 artists (8×6 = 48 similar fetches)
    # hop 2 → expand up to 6 artists
    # hop 3+ → expand up to 4 artists
    _MAX_EXPAND = [8, 6, 4]

    for hop in range(min(max_hops, len(_MAX_EXPAND))):
        if not frontier:
            break

        batch = frontier[:_MAX_EXPAND[hop]]

        sim_results = await asyncio.gather(
            *[lastfm.get_similar_artists(canon, limit=6) for canon, _ in batch],
            return_exceptions=True,
        )

        new_frontier: list[tuple[str, str]] = []
        seen_bridges: set[str] = {p[-1].lower() for p in found_paths}

        for (canon_src, lower_src), similar in zip(batch, sim_results):
            if isinstance(similar, Exception):
                logger.debug("get_similar_artists(%r) failed: %s", canon_src, similar)
                continue

            src_path = path_map[lower_src]

            for sim in similar:
                canon = sim["name"]
                lower = canon.lower()
                new_path = src_path + [canon]

                # Found a bridge — record it (deduplicate by endpoint)
                if lower in names_b_lower and lower not in seen_bridges:
                    found_paths.append(new_path)
                    seen_bridges.add(lower)

                # Add to next frontier if unseen
                if lower not in visited:
                    visited.add(lower)
                    path_map[lower] = new_path
                    new_frontier.append((canon, lower))

        frontier = new_frontier

        if len(found_paths) >= 5:
            break  # enough paths collected

    # ── Stage 3: fetch tags for bridge/intermediate artists ───────────────────
    # Include every artist in found paths except genre_a seeds (their genre
    # is already known from the query).
    to_tag: list[str] = []
    seen_for_tag: set[str] = set(names_a_lower)

    for path in found_paths:
        for artist in path:  # include the genre_b endpoint too
            if artist.lower() not in seen_for_tag:
                to_tag.append(artist)
                seen_for_tag.add(artist.lower())

    # If no paths found, still tag some hop-1 candidates so Claude has context
    # about the similarity neighbourhood surrounding the genre_a seeds.
    if not found_paths:
        for lower, path in path_map.items():
            if lower not in names_a_lower and len(path) == 2:
                a = path[-1]
                if a.lower() not in seen_for_tag:
                    to_tag.append(a)
                    seen_for_tag.add(a.lower())
                if len(to_tag) >= 8:
                    break

    to_tag = to_tag[:12]  # hard cap

    tag_results = await asyncio.gather(
        *[lastfm.get_artist_tags(a) for a in to_tag],
        return_exceptions=True,
    )

    artist_tags: dict[str, list[str]] = {}
    for artist, result in zip(to_tag, tag_results):
        if not isinstance(result, Exception):
            artist_tags[artist] = [t for t in result[:8] if t]

    return {
        "seeds_a": [s["name"] for s in seeds_a],
        "seeds_b": [s["name"] for s in seeds_b],
        "found_paths": found_paths[:5],
        "artist_tags": artist_tags,
        "kb_a": kb_a[:800] if kb_a else "",
        "kb_b": kb_b[:800] if kb_b else "",
        "hops_searched": min(max_hops, len(_MAX_EXPAND)),
    }


async def _claude_bridge_analysis(
    genre_a: str,
    genre_b: str,
    graph: dict,
) -> BridgeResponse:
    """
    Ask Claude to interpret the BFS graph data and produce a structured bridge analysis.

    Even when the graph found no paths, Claude draws on its training knowledge
    (grounded by the KB context) to surface musically meaningful connections.
    """
    # ── Format found paths ────────────────────────────────────────────────────
    if graph["found_paths"]:
        path_lines = "\n".join(
            f"  • {' → '.join(path)}"
            for path in graph["found_paths"]
        )
        paths_section = f"Similarity graph paths found ({len(graph['found_paths'])}):\n{path_lines}"
    else:
        paths_section = (
            f"No direct similarity-graph paths found within "
            f"{graph['hops_searched']} hop(s). Use your musical knowledge."
        )

    # ── Format artist tags ────────────────────────────────────────────────────
    if graph["artist_tags"]:
        tag_lines = "\n".join(
            f"  • {artist}: {', '.join(tags)}"
            for artist, tags in graph["artist_tags"].items()
        )
        tags_section = f"Tags for bridge/intermediate artists:\n{tag_lines}"
    else:
        tags_section = "No tag data available for intermediate artists."

    prompt = f"""\
You are a music historian and genre expert. Explain how {genre_a} connects to {genre_b}.

Seed artists for {genre_a}: {', '.join(graph['seeds_a'])}
Seed artists for {genre_b}: {', '.join(graph['seeds_b'])}

{paths_section}

{tags_section}

Knowledge base context — {genre_a}:
{graph['kb_a'] or '(not available)'}

Knowledge base context — {genre_b}:
{graph['kb_b'] or '(not available)'}

Tasks:
1. Select 4–6 artists that best serve as musical stepping stones from {genre_a} to \
{genre_b}. Prioritise artists from the similarity paths above when they exist; \
supplement with your own knowledge. For each, explain concisely WHY they bridge \
the gap — reference specific musical elements (rhythm, timbre, harmony, production).
2. Build a 8–10 track transition playlist. Order it so it moves the listener \
smoothly from deep {genre_a} through the hybrid territory into {genre_b}. For each \
track write a short "position" string describing where in the journey it sits \
(e.g. "pure {genre_a}" / "{genre_a}-leaning fusion" / "right at the intersection" \
/ "{genre_b}-leaning" / "pure {genre_b}").
3. Write a 2–3 paragraph explanation of the musical and historical connection between \
the two genres: shared techniques, mutual influences, key crossover moments.

Respond with ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "bridge_artists": [
    {{"name": "...", "connects_because": "...", "genres": ["...", "..."]}}
  ],
  "transition_playlist": [
    {{"track": "...", "artist": "...", "position": "..."}}
  ],
  "explanation": "paragraph 1\\n\\nparagraph 2\\n\\nparagraph 3"
}}"""

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=_CLAUDE_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Claude returned non-JSON for bridge: %s\n%.400s", exc, raw)
        raise HTTPException(status_code=500, detail="Failed to parse bridge analysis from Claude")

    bridge_artists = [
        BridgeArtist(
            name=ba.get("name", ""),
            connects_because=ba.get("connects_because", ""),
            genres=ba.get("genres", []),
        )
        for ba in data.get("bridge_artists", [])
    ]

    transition_playlist = [
        TransitionTrack(
            track=t.get("track", ""),
            artist=t.get("artist", ""),
            position=t.get("position", ""),
        )
        for t in data.get("transition_playlist", [])
    ]

    return BridgeResponse(
        genre_a=genre_a,
        genre_b=genre_b,
        bridge_artists=bridge_artists,
        transition_playlist=transition_playlist,
        explanation=data.get("explanation", ""),
    )
