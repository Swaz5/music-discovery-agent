"""
Tests for the FastAPI REST API.

Uses httpx.AsyncClient with ASGITransport to hit the app in-process.
All heavy dependencies (agent, music_service, lastfm, retriever, prefs)
are patched so tests run without network access or API keys.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from httpx import AsyncClient, ASGITransport

from src.api.main import app
import src.api.routes as routes_module

# ── Shared fixture data ───────────────────────────────────────────────────────

SAMPLE_RECOMMENDATIONS = [
    {
        "artist": "Portishead",
        "track": "Glory Box",
        "genre_tags": ["trip-hop", "electronic", "dark"],
        "why": "Haunting vocals over downtempo beats with deep bass presence.",
        "vibe_description": "Like wandering through a foggy harbour at 3 AM.",
        "vibe_match_score": 0.92,
        "deezer_url": "https://www.deezer.com/track/123456",
    }
]

SAMPLE_DISCOVER_RESULT = {
    "recommendations": SAMPLE_RECOMMENDATIONS,
    "reasoning_trace": [
        {
            "iteration": 1,
            "tool": "search_knowledge_base",
            "arguments": {"query": "dark atmospheric trip-hop"},
            "result_preview": "Trip-hop originated in Bristol in the early 1990s...",
        }
    ],
    "iterations": 2,
    "total_tokens": 1500,
}

SAMPLE_ARTIST_PROFILE = {
    "name": "Portishead",
    "bio": "Portishead are a British band from Bristol...",
    "tags": ["trip-hop", "electronic", "dark", "atmospheric"],
    "similar_artists": [{"name": "Massive Attack", "url": "https://www.last.fm/music/Massive+Attack"}],
    "top_tracks": [],
    "audio_profile": {"avg_energy": 0.38, "avg_valence": 0.22},
    "fan_count": 1200000,
    "listeners": 2100000,
    "album_art_url": "https://example.com/portishead.jpg",
}

SAMPLE_TASTE_PROFILE = {
    "preferred_tags": ["trip-hop", "electronic"],
    "preferred_audio": {"energy": 0.4, "valence": 0.25},
    "disliked_tags": [],
    "disliked_audio": {},
    "sweet_spots": {},
    "total_ratings": 1,
    "liked_count": 1,
    "disliked_count": 0,
    "summary": "This user prefers low-energy, atmospheric music.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_history():
    """Clear in-memory history between tests."""
    routes_module._history.clear()
    yield
    routes_module._history.clear()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset lazy singletons so patches take effect cleanly."""
    routes_module._agent = None
    routes_module._prefs = None
    routes_module._retriever = None
    yield
    routes_module._agent = None
    routes_module._prefs = None
    routes_module._retriever = None


@pytest.fixture
def async_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── Health check ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check(async_client):
    async with async_client as client:
        response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "music-discovery-agent" in body["service"]


# ── POST /discover (JSON) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_returns_recommendations(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            response = await client.post(
                "/discover", json={"query": "dark atmospheric trip-hop"}
            )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "dark atmospheric trip-hop"
    assert len(body["recommendations"]) == 1
    assert body["recommendations"][0]["artist"] == "Portishead"
    assert body["iterations"] == 2
    assert body["total_tokens"] == 1500
    assert body["reasoning_trace"] is None  # include_reasoning defaults to false


@pytest.mark.asyncio
async def test_discover_with_reasoning(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            response = await client.post(
                "/discover",
                json={"query": "dark atmospheric trip-hop", "include_reasoning": True},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["reasoning_trace"] is not None
    assert len(body["reasoning_trace"]) == 1
    assert body["reasoning_trace"][0]["tool"] == "search_knowledge_base"


@pytest.mark.asyncio
async def test_discover_records_history(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            await client.post("/discover", json={"query": "cozy jazz"})
            history_response = await client.get("/history")

    assert history_response.status_code == 200
    history = history_response.json()
    assert history["total"] == 1
    assert history["items"][0]["query"] == "cozy jazz"


@pytest.mark.asyncio
async def test_discover_empty_query_rejected(async_client):
    async with async_client as client:
        response = await client.post("/discover", json={"query": ""})
    assert response.status_code == 422  # Pydantic min_length validation


@pytest.mark.asyncio
async def test_discover_agent_error_returns_500(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(side_effect=RuntimeError("API key missing"))

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            response = await client.post("/discover", json={"query": "something"})

    assert response.status_code == 500
    assert "API key missing" in response.json()["detail"]


# ── POST /discover (SSE streaming) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_stream_events(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            response = await client.post(
                "/discover",
                json={"query": "dark trip-hop", "stream": True, "include_reasoning": True},
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Parse SSE events from the response body
    events = _parse_sse(response.text)
    types = [e["type"] for e in events]

    assert "start" in types
    assert "reasoning_step" in types
    assert "recommendations" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_discover_stream_error_event(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(side_effect=ValueError("service down"))

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            response = await client.post(
                "/discover", json={"query": "test", "stream": True}
            )

    events = _parse_sse(response.text)
    types = [e["type"] for e in events]
    assert "start" in types
    assert "error" in types
    error_event = next(e for e in events if e["type"] == "error")
    assert "service down" in error_event["message"]


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into a list of data dicts."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ── GET /artist/{name} ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_artist_returns_profile(async_client):
    with patch(
        "src.api.routes.music_service.get_full_artist_profile",
        new_callable=AsyncMock,
        return_value=SAMPLE_ARTIST_PROFILE,
    ):
        async with async_client as client:
            response = await client.get("/artist/Portishead")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Portishead"
    assert "trip-hop" in body["tags"]


@pytest.mark.asyncio
async def test_get_artist_not_found(async_client):
    with patch(
        "src.api.routes.music_service.get_full_artist_profile",
        new_callable=AsyncMock,
        return_value={},
    ):
        async with async_client as client:
            response = await client.get("/artist/NonExistentArtistXYZ123")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_artist_service_error(async_client):
    with patch(
        "src.api.routes.music_service.get_full_artist_profile",
        new_callable=AsyncMock,
        side_effect=ConnectionError("Last.fm unreachable"),
    ):
        async with async_client as client:
            response = await client.get("/artist/Portishead")

    assert response.status_code == 500


# ── POST /explore ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_explore_genre(async_client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(
        return_value="Trip-hop originated in Bristol in the early 1990s..."
    )
    top_artists = [
        {"name": "Portishead", "rank": "1", "url": "https://www.last.fm/music/Portishead"},
        {"name": "Massive Attack", "rank": "2", "url": "https://www.last.fm/music/Massive+Attack"},
    ]
    artist_tags = ["trip-hop", "electronic", "ambient", "dark"]

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            return_value=top_artists,
        ),
        patch(
            "src.api.routes.lastfm.get_artist_tags",
            new_callable=AsyncMock,
            return_value=artist_tags,
        ),
    ):
        async with async_client as client:
            response = await client.post("/explore", json={"genre": "trip-hop"})

    assert response.status_code == 200
    body = response.json()
    assert body["genre"] == "trip-hop"
    assert "Trip-hop" in body["knowledge_context"]
    assert len(body["top_artists"]) == 2
    assert isinstance(body["related_genres"], list)
    # "trip-hop" should be excluded from related genres
    assert "trip-hop" not in body["related_genres"]


@pytest.mark.asyncio
async def test_explore_empty_genre_rejected(async_client):
    async with async_client as client:
        response = await client.post("/explore", json={"genre": ""})
    assert response.status_code == 422


# ── POST /rate ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_liked(async_client):
    mock_prefs = MagicMock()
    mock_prefs.save_preference = AsyncMock()
    mock_prefs.get_taste_profile = MagicMock(return_value=SAMPLE_TASTE_PROFILE)

    with patch.object(routes_module, "_get_prefs", return_value=mock_prefs):
        async with async_client as client:
            response = await client.post(
                "/rate",
                json={"artist": "Portishead", "liked": True, "notes": "love the dark atmosphere"},
            )

    assert response.status_code == 200
    body = response.json()
    assert "Portishead" in body["message"]
    assert "liked" in body["message"]
    assert body["taste_profile"]["liked_count"] == 1
    mock_prefs.save_preference.assert_awaited_once_with(
        "Portishead", liked=True, notes="love the dark atmosphere"
    )


@pytest.mark.asyncio
async def test_rate_disliked(async_client):
    mock_prefs = MagicMock()
    mock_prefs.save_preference = AsyncMock()
    mock_prefs.get_taste_profile = MagicMock(return_value={})

    with patch.object(routes_module, "_get_prefs", return_value=mock_prefs):
        async with async_client as client:
            response = await client.post(
                "/rate", json={"artist": "Some Artist", "liked": False}
            )

    assert response.status_code == 200
    assert "disliked" in response.json()["message"]


@pytest.mark.asyncio
async def test_rate_empty_artist_rejected(async_client):
    async with async_client as client:
        response = await client.post("/rate", json={"artist": "", "liked": True})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_rate_save_error_returns_500(async_client):
    mock_prefs = MagicMock()
    mock_prefs.save_preference = AsyncMock(side_effect=IOError("disk full"))

    with patch.object(routes_module, "_get_prefs", return_value=mock_prefs):
        async with async_client as client:
            response = await client.post(
                "/rate", json={"artist": "Portishead", "liked": True}
            )

    assert response.status_code == 500


# ── GET /taste-profile ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_taste_profile(async_client):
    mock_prefs = MagicMock()
    mock_prefs.get_taste_profile = MagicMock(return_value=SAMPLE_TASTE_PROFILE)

    with patch.object(routes_module, "_get_prefs", return_value=mock_prefs):
        async with async_client as client:
            response = await client.get("/taste-profile")

    assert response.status_code == 200
    body = response.json()
    assert body["liked_count"] == 1
    assert "trip-hop" in body["preferred_tags"]


@pytest.mark.asyncio
async def test_get_taste_profile_empty(async_client):
    mock_prefs = MagicMock()
    mock_prefs.get_taste_profile = MagicMock(return_value={})

    with patch.object(routes_module, "_get_prefs", return_value=mock_prefs):
        async with async_client as client:
            response = await client.get("/taste-profile")

    assert response.status_code == 200
    assert response.json() == {}


# ── GET /history ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_history_empty_on_start(async_client):
    async with async_client as client:
        response = await client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []


@pytest.mark.asyncio
async def test_history_accumulates_across_calls(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            await client.post("/discover", json={"query": "rainy day jazz"})
            await client.post("/discover", json={"query": "energetic workout music"})
            response = await client.get("/history")

    body = response.json()
    assert body["total"] == 2
    queries = [item["query"] for item in body["items"]]
    assert "rainy day jazz" in queries
    assert "energetic workout music" in queries


@pytest.mark.asyncio
async def test_history_item_has_expected_fields(async_client):
    mock_agent = MagicMock()
    mock_agent.discover = AsyncMock(return_value=SAMPLE_DISCOVER_RESULT)

    with patch.object(routes_module, "_get_agent", return_value=mock_agent):
        async with async_client as client:
            await client.post("/discover", json={"query": "test query"})
            response = await client.get("/history")

    item = response.json()["items"][0]
    assert "query" in item
    assert "recommendations" in item
    assert "iterations" in item
    assert "total_tokens" in item
    assert "timestamp" in item
    assert item["timestamp"] <= time.time()


# ── POST /analyze-taste ───────────────────────────────────────────────────────

SAMPLE_PROFILES = [
    {
        "name": "Radiohead",
        "tags": ["alternative rock", "art rock", "experimental"],
        "audio_profile": {
            "avg_energy": 0.55,
            "avg_danceability": 0.38,
            "avg_valence": 0.28,
            "avg_tempo": 108.0,
            "avg_acousticness": 0.20,
        },
    },
    {
        "name": "Tame Impala",
        "tags": ["psychedelic rock", "indie pop", "dream pop"],
        "audio_profile": {
            "avg_energy": 0.62,
            "avg_danceability": 0.55,
            "avg_valence": 0.45,
            "avg_tempo": 115.0,
            "avg_acousticness": 0.15,
        },
    },
    {
        "name": "Beach House",
        "tags": ["dream pop", "shoegaze", "indie pop"],
        "audio_profile": {
            "avg_energy": 0.40,
            "avg_danceability": 0.42,
            "avg_valence": 0.35,
            "avg_tempo": 95.0,
            "avg_acousticness": 0.38,
        },
    },
]

SAMPLE_CLAUDE_ANALYSIS = {
    "taste_identity": "Melancholic Texturalist",
    "analysis": (
        "This listener gravitates toward music that prioritizes texture and atmosphere "
        "over conventional song structure.\n\n"
        "The audio data reveals a consistent preference for low-to-mid energy music "
        "with dark valence, suggesting an appreciation for emotional complexity.\n\n"
        "All three artists share a willingness to use sonic space as a compositional tool."
    ),
    "common_threads": [
        "Atmospheric layering over conventional structure",
        "Dark or ambiguous emotional tone",
        "Psychedelic influences",
        "Indie/alternative sensibility",
    ],
    "blind_spots": [
        {
            "genre": "ambient",
            "why": "The low energy and textural focus suggest they'd love pure ambient music.",
            "try_this": "Brian Eno — Music For Airports",
        },
        {
            "genre": "post-rock",
            "why": "Builds the same emotional intensity without vocals.",
            "try_this": "Sigur Rós — Ára bátur",
        },
    ],
    "outlier": {
        "artist": "Tame Impala",
        "why_different": "Higher energy and danceability than the other two; more pop-leaning.",
    },
}


@pytest.mark.asyncio
async def test_analyze_taste_full_response(async_client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(SAMPLE_CLAUDE_ANALYSIS))]

    with (
        patch(
            "src.api.routes.music_service.get_full_artist_profile",
            new_callable=AsyncMock,
            side_effect=SAMPLE_PROFILES,
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(
                    create=AsyncMock(return_value=mock_response)
                )
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={
                    "artists": ["Radiohead", "Tame Impala", "Beach House"],
                    "discover_blind_spots": True,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["taste_identity"] == "Melancholic Texturalist"
    assert len(body["analysis"]) > 50
    assert len(body["common_threads"]) >= 2
    assert len(body["blind_spots"]) == 2
    assert body["blind_spots"][0]["genre"] == "ambient"
    assert body["blind_spots"][0]["try_this"] == "Brian Eno — Music For Airports"
    assert body["outlier"]["artist"] == "Tame Impala"
    assert "energy" in body["audio_profile"]


@pytest.mark.asyncio
async def test_analyze_taste_no_blind_spots(async_client):
    claude_response = {**SAMPLE_CLAUDE_ANALYSIS, "blind_spots": []}
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(claude_response))]

    with (
        patch(
            "src.api.routes.music_service.get_full_artist_profile",
            new_callable=AsyncMock,
            side_effect=SAMPLE_PROFILES,
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={
                    "artists": ["Radiohead", "Tame Impala", "Beach House"],
                    "discover_blind_spots": False,
                },
            )

    assert response.status_code == 200
    assert response.json()["blind_spots"] == []


@pytest.mark.asyncio
async def test_analyze_taste_partial_profile_failures(async_client):
    """If some artist lookups fail, analysis proceeds with those that succeed."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(SAMPLE_CLAUDE_ANALYSIS))]

    # First call succeeds, second raises, third succeeds
    side_effects = [SAMPLE_PROFILES[0], ConnectionError("timeout"), SAMPLE_PROFILES[2]]

    with (
        patch(
            "src.api.routes.music_service.get_full_artist_profile",
            new_callable=AsyncMock,
            side_effect=side_effects,
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={"artists": ["Radiohead", "Tame Impala", "Beach House"]},
            )

    assert response.status_code == 200  # still returns despite one failure


@pytest.mark.asyncio
async def test_analyze_taste_all_profiles_fail(async_client):
    with patch(
        "src.api.routes.music_service.get_full_artist_profile",
        new_callable=AsyncMock,
        side_effect=ConnectionError("service down"),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={"artists": ["Radiohead", "Tame Impala"]},
            )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_taste_empty_artists_rejected(async_client):
    async with async_client as client:
        response = await client.post("/analyze-taste", json={"artists": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_analyze_taste_claude_bad_json(async_client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is not JSON at all")]

    with (
        patch(
            "src.api.routes.music_service.get_full_artist_profile",
            new_callable=AsyncMock,
            side_effect=SAMPLE_PROFILES,
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={"artists": ["Radiohead", "Tame Impala", "Beach House"]},
            )

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_analyze_taste_strips_markdown_fences(async_client):
    """Claude sometimes wraps JSON in ```json fences — ensure we strip them."""
    fenced = f"```json\n{json.dumps(SAMPLE_CLAUDE_ANALYSIS)}\n```"
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=fenced)]

    with (
        patch(
            "src.api.routes.music_service.get_full_artist_profile",
            new_callable=AsyncMock,
            side_effect=SAMPLE_PROFILES,
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/analyze-taste",
                json={"artists": ["Radiohead", "Tame Impala", "Beach House"]},
            )

    assert response.status_code == 200
    assert response.json()["taste_identity"] == "Melancholic Texturalist"


# ── POST /bridge ──────────────────────────────────────────────────────────────

# Last.fm mock data
JAZZ_SEEDS = [
    {"name": "Miles Davis", "rank": "1", "url": ""},
    {"name": "John Coltrane", "rank": "2", "url": ""},
    {"name": "Herbie Hancock", "rank": "3", "url": ""},
    {"name": "Charles Mingus", "rank": "4", "url": ""},
]

ELECTRONIC_SEEDS = [
    {"name": "Aphex Twin", "rank": "1", "url": ""},
    {"name": "Flying Lotus", "rank": "2", "url": ""},
    {"name": "Boards of Canada", "rank": "3", "url": ""},
    {"name": "Autechre", "rank": "4", "url": ""},
]

# Herbie Hancock's similar artists include Flying Lotus → 1-hop bridge
HANCOCK_SIMILAR = [
    {"name": "Flying Lotus", "match": 0.85, "url": ""},
    {"name": "Thundercat", "match": 0.72, "url": ""},
    {"name": "Kamasi Washington", "match": 0.68, "url": ""},
]

MILES_SIMILAR = [
    {"name": "John Coltrane", "match": 0.95, "url": ""},
    {"name": "Bill Evans", "match": 0.88, "url": ""},
    {"name": "Wayne Shorter", "match": 0.80, "url": ""},
]

COLTRANE_SIMILAR = [
    {"name": "Pharoah Sanders", "match": 0.90, "url": ""},
    {"name": "Alice Coltrane", "match": 0.85, "url": ""},
    {"name": "Archie Shepp", "match": 0.75, "url": ""},
]

MINGUS_SIMILAR = [
    {"name": "Charles Mingus Jr.", "match": 0.60, "url": ""},
]

SAMPLE_BRIDGE_ANALYSIS = {
    "bridge_artists": [
        {
            "name": "Herbie Hancock",
            "connects_because": "Hancock pioneered electronic jazz fusion...",
            "genres": ["jazz", "funk", "electronic"],
        },
        {
            "name": "Flying Lotus",
            "connects_because": "Flying Lotus blends electronic production with jazz harmony...",
            "genres": ["electronic", "jazz", "hip-hop"],
        },
    ],
    "transition_playlist": [
        {"track": "So What", "artist": "Miles Davis", "position": "pure jazz foundation"},
        {"track": "Rockit", "artist": "Herbie Hancock", "position": "jazz meets electronic"},
        {"track": "Zodiac Shit", "artist": "Flying Lotus", "position": "right at the intersection"},
        {"track": "Windowlicker", "artist": "Aphex Twin", "position": "pure electronic"},
    ],
    "explanation": "Jazz and electronic music share a deep connection...\n\nBoth genres prize improvisation...\n\nThe bridge runs through funk.",
}


def _make_similar_side_effect(seeds_a):
    """Return get_similar_artists responses keyed by artist name."""
    sim_map = {
        "Miles Davis": MILES_SIMILAR,
        "John Coltrane": COLTRANE_SIMILAR,
        "Herbie Hancock": HANCOCK_SIMILAR,
        "Charles Mingus": MINGUS_SIMILAR,
    }

    async def _impl(artist, limit=10):
        return sim_map.get(artist, [])

    return _impl


@pytest.mark.asyncio
async def test_bridge_finds_one_hop_path(async_client):
    """Herbie Hancock's similar artists include Flying Lotus → 1-hop bridge."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(SAMPLE_BRIDGE_ANALYSIS))]

    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(return_value="genre context")

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            side_effect=[JAZZ_SEEDS, ELECTRONIC_SEEDS],
        ),
        patch(
            "src.api.routes.lastfm.get_similar_artists",
            side_effect=_make_similar_side_effect(JAZZ_SEEDS),
        ),
        patch(
            "src.api.routes.lastfm.get_artist_tags",
            new_callable=AsyncMock,
            return_value=["jazz", "electronic", "fusion"],
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic", "max_hops": 1},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["genre_a"] == "jazz"
    assert body["genre_b"] == "electronic"
    assert len(body["bridge_artists"]) >= 1
    assert len(body["transition_playlist"]) >= 2
    assert body["explanation"]


@pytest.mark.asyncio
async def test_bridge_response_structure(async_client):
    """All required fields are present and correctly typed."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(SAMPLE_BRIDGE_ANALYSIS))]
    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(return_value="")

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            side_effect=[JAZZ_SEEDS, ELECTRONIC_SEEDS],
        ),
        patch(
            "src.api.routes.lastfm.get_similar_artists",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes.lastfm.get_artist_tags",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic"},
            )

    assert response.status_code == 200
    body = response.json()

    # Top-level structure
    for key in ("genre_a", "genre_b", "bridge_artists", "transition_playlist", "explanation"):
        assert key in body, f"Missing field: {key}"

    # bridge_artists items
    for artist in body["bridge_artists"]:
        assert "name" in artist
        assert "connects_because" in artist
        assert "genres" in artist
        assert isinstance(artist["genres"], list)

    # transition_playlist items
    for track in body["transition_playlist"]:
        assert "track" in track
        assert "artist" in track
        assert "position" in track


@pytest.mark.asyncio
async def test_bridge_same_genre_rejected(async_client):
    async with async_client as client:
        response = await client.post(
            "/bridge",
            json={"genre_a": "jazz", "genre_b": "jazz"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bridge_empty_genre_rejected(async_client):
    async with async_client as client:
        response = await client.post(
            "/bridge",
            json={"genre_a": "", "genre_b": "electronic"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bridge_max_hops_out_of_range(async_client):
    async with async_client as client:
        response = await client.post(
            "/bridge",
            json={"genre_a": "jazz", "genre_b": "electronic", "max_hops": 0},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bridge_no_paths_still_calls_claude(async_client):
    """When BFS finds no paths, Claude still runs and returns an analysis."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(SAMPLE_BRIDGE_ANALYSIS))]
    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(return_value="")

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            side_effect=[JAZZ_SEEDS, ELECTRONIC_SEEDS],
        ),
        patch(
            "src.api.routes.lastfm.get_similar_artists",
            new_callable=AsyncMock,
            return_value=[],  # no similarity → no paths found
        ),
        patch(
            "src.api.routes.lastfm.get_artist_tags",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic", "max_hops": 2},
            )

    assert response.status_code == 200
    assert response.json()["explanation"]


@pytest.mark.asyncio
async def test_bridge_seed_fetch_failure_returns_500(async_client):
    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(return_value="")

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            side_effect=ConnectionError("Last.fm down"),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic"},
            )

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_bridge_claude_bad_json_returns_500(async_client):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="not json")]
    mock_retriever = MagicMock()
    mock_retriever.retrieve_for_genre = MagicMock(return_value="")

    with (
        patch.object(routes_module, "_get_retriever", return_value=mock_retriever),
        patch(
            "src.api.routes.lastfm.get_tag_top_artists",
            new_callable=AsyncMock,
            side_effect=[JAZZ_SEEDS, ELECTRONIC_SEEDS],
        ),
        patch(
            "src.api.routes.lastfm.get_similar_artists",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes.lastfm.get_artist_tags",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.api.routes.anthropic.AsyncAnthropic",
            return_value=MagicMock(
                messages=MagicMock(create=AsyncMock(return_value=mock_response))
            ),
        ),
    ):
        async with async_client as client:
            response = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic"},
            )

    assert response.status_code == 500
