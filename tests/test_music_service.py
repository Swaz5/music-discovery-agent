"""
Tests for MusicService — caching behaviour and parallel execution.

We mock all three underlying sources (Last.fm, Deezer, AudioAnalyzer) so
tests run fast and offline. Caching and parallel-call assertions are the
primary concerns; correctness of the underlying clients is tested separately.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.data.music_service as svc
from src.data.music_service import (
    get_full_artist_profile,
    get_similar_with_features,
    search_artists_by_vibe,
    cache_clear,
    _cache_key,
    _cache_get,
    _cache_set,
    TTL_SECONDS,
)

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

LASTFM_INFO = {
    "name": "Radiohead",
    "bio_summary": "Radiohead are an English rock band.",
    "tags": ["alternative rock", "art rock", "electronic"],
    "similar_artists": [
        {"name": "Thom Yorke", "url": "https://www.last.fm/music/Thom+Yorke"},
        {"name": "Portishead", "url": "https://www.last.fm/music/Portishead"},
    ],
    "listeners": "4800000",
    "playcount": "650000000",
}

LASTFM_TRACKS = [
    {"name": "Creep", "playcount": "90000000", "listeners": "5000000", "url": "https://..."},
    {"name": "Karma Police", "playcount": "60000000", "listeners": "4000000", "url": "https://..."},
]

DEEZER_ARTIST = {
    "id": 399,
    "name": "Radiohead",
    "picture_url": "https://cdn.deezer.com/artist399.jpg",
    "nb_fan": 2500000,
}

DEEZER_TRACKS = [
    {
        "title": "Creep",
        "duration": 238,
        "preview_url": "https://cdn.deezer.com/preview/creep.mp3",
        "album": {"title": "Pablo Honey", "cover_url": "https://cdn.deezer.com/cover1.jpg"},
    },
    {
        "title": "Karma Police",
        "duration": 264,
        "preview_url": "https://cdn.deezer.com/preview/karma.mp3",
        "album": {"title": "OK Computer", "cover_url": "https://cdn.deezer.com/cover2.jpg"},
    },
]

AUDIO_FEATURES = {
    "file_path": "/tmp/Creep.mp3",
    "energy": 0.65,
    "energy_label": "moderate energy",
    "tempo": 92.0,
    "tempo_label": "moderate",
    "danceability": 0.82,
    "valence": 0.55,
    "valence_label": "positive",
    "acousticness": 0.70,
    "instrumentalness": 0.48,
    "loudness": -15.5,
}

SIMILAR_ARTISTS = [
    {"name": "Thom Yorke", "match": 1.0, "url": "https://..."},
    {"name": "Portishead", "match": 0.72, "url": "https://..."},
]

DEEZER_SEARCH_TRACKS = [
    {
        "title": "Creep",
        "artist": "Radiohead",
        "album": "Pablo Honey",
        "duration": 238,
        "preview_url": "https://cdn.deezer.com/preview/creep.mp3",
        "cover_url": "https://cdn.deezer.com/cover1.jpg",
        "link": "https://www.deezer.com/track/123",
    }
]

TAG_ARTISTS = [
    {"name": "Radiohead", "rank": "1", "url": "https://..."},
    {"name": "Portishead", "rank": "2", "url": "https://..."},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Wipe the cache before every test so tests are independent."""
    cache_clear()
    yield
    cache_clear()


def _mock_analyzer():
    analyzer = MagicMock()
    analyzer.analyze_track = MagicMock(return_value=AUDIO_FEATURES)
    return analyzer


def _patch_all(
    lastfm_info=LASTFM_INFO,
    lastfm_tracks=LASTFM_TRACKS,
    lastfm_similar=SIMILAR_ARTISTS,
    lastfm_tag_artists=TAG_ARTISTS,
    deezer_artist=DEEZER_ARTIST,
    deezer_top_tracks=DEEZER_TRACKS,
    deezer_search=DEEZER_SEARCH_TRACKS,
    deezer_download="data/previews/Creep.mp3",
):
    """Return a list of patches covering all external calls."""
    return [
        patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(return_value=lastfm_info)),
        patch("src.data.music_service.lastfm.get_top_tracks", AsyncMock(return_value=lastfm_tracks)),
        patch("src.data.music_service.lastfm.get_similar_artists", AsyncMock(return_value=lastfm_similar)),
        patch("src.data.music_service.lastfm.get_tag_top_artists", AsyncMock(return_value=lastfm_tag_artists)),
        patch("src.data.music_service.deezer.search_artist", AsyncMock(return_value=deezer_artist)),
        patch("src.data.music_service.deezer.get_artist_top_tracks", AsyncMock(return_value=deezer_top_tracks)),
        patch("src.data.music_service.deezer.search_tracks", AsyncMock(return_value=deezer_search)),
        patch("src.data.music_service.deezer.download_preview", AsyncMock(return_value=deezer_download)),
        patch("src.data.music_service._analyzer", _mock_analyzer()),
    ]


# ---------------------------------------------------------------------------
# Tests: in-memory cache primitives
# ---------------------------------------------------------------------------

def test_cache_miss_returns_none():
    assert _cache_get("nonexistent_key") is None


def test_cache_set_and_get():
    _cache_set("k1", {"data": 42})
    assert _cache_get("k1") == {"data": 42}


def test_cache_hit_returns_same_object():
    obj = [1, 2, 3]
    _cache_set("obj_key", obj)
    assert _cache_get("obj_key") is obj


def test_cache_expires_after_ttl(monkeypatch):
    _cache_set("exp_key", "value")
    # Simulate time advancing past TTL
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + TTL_SECONDS + 1)
    assert _cache_get("exp_key") is None


def test_cache_not_expired_before_ttl(monkeypatch):
    _cache_set("fresh_key", "still good")
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + TTL_SECONDS - 1)
    assert _cache_get("fresh_key") == "still good"


def test_cache_clear_removes_all():
    _cache_set("a", 1)
    _cache_set("b", 2)
    cache_clear()
    assert _cache_get("a") is None
    assert _cache_get("b") is None


def test_cache_key_is_deterministic():
    k1 = _cache_key("fn", "arg1", "arg2")
    k2 = _cache_key("fn", "arg1", "arg2")
    assert k1 == k2


def test_cache_key_differs_for_different_args():
    assert _cache_key("fn", "a") != _cache_key("fn", "b")
    assert _cache_key("fn1", "a") != _cache_key("fn2", "a")


# ---------------------------------------------------------------------------
# Tests: get_full_artist_profile — structure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_full_artist_profile_returns_expected_keys():
    with _patch_all()[0], _patch_all()[1], _patch_all()[2], _patch_all()[3], \
         _patch_all()[4], _patch_all()[5], _patch_all()[6], _patch_all()[7], \
         _patch_all()[8]:
        profile = await get_full_artist_profile("Radiohead")

    expected = {"name", "bio", "tags", "similar_artists", "top_tracks",
                "audio_profile", "fan_count", "listeners", "album_art_url"}
    assert expected.issubset(profile.keys())


@pytest.mark.asyncio
async def test_get_full_artist_profile_fields():
    patches = _patch_all()
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        profile = await get_full_artist_profile("Radiohead")

    assert profile["name"] == "Radiohead"
    assert profile["bio"] == LASTFM_INFO["bio_summary"]
    assert profile["tags"] == LASTFM_INFO["tags"]
    assert profile["fan_count"] == 2500000
    assert profile["listeners"] == 4800000
    assert profile["album_art_url"] == DEEZER_ARTIST["picture_url"]


@pytest.mark.asyncio
async def test_get_full_artist_profile_audio_profile_keys():
    patches = _patch_all()
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        profile = await get_full_artist_profile("Radiohead")

    ap = profile["audio_profile"]
    assert "avg_energy" in ap
    assert "avg_danceability" in ap
    assert "avg_valence" in ap
    assert "avg_tempo" in ap
    assert "avg_acousticness" in ap


# ---------------------------------------------------------------------------
# Tests: get_full_artist_profile — caching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_full_artist_profile_caches_result():
    call_count = 0
    original = svc.lastfm.get_artist_info

    async def counting_info(name):
        nonlocal call_count
        call_count += 1
        return LASTFM_INFO

    patches = _patch_all(lastfm_info=LASTFM_INFO)
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        # Replace the already-patched mock with our counting version
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=counting_info)):
            await get_full_artist_profile("Radiohead")
            await get_full_artist_profile("Radiohead")  # second call — should hit cache

    assert call_count == 1


@pytest.mark.asyncio
async def test_get_full_artist_profile_cache_is_case_insensitive_key():
    """Two calls with different casing share the same cache entry."""
    call_count = 0

    async def counting_info(name):
        nonlocal call_count
        call_count += 1
        return LASTFM_INFO

    patches = _patch_all()
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=counting_info)):
            await get_full_artist_profile("Radiohead")
            await get_full_artist_profile("radiohead")

    assert call_count == 1


@pytest.mark.asyncio
async def test_get_full_artist_profile_cache_expires(monkeypatch):
    """After TTL elapses, a fresh fetch should occur."""
    call_count = 0

    async def counting_info(name):
        nonlocal call_count
        call_count += 1
        return LASTFM_INFO

    patches = _patch_all()
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=counting_info)):
            await get_full_artist_profile("Radiohead")

            # Advance clock past TTL
            original_monotonic = time.monotonic
            monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + TTL_SECONDS + 1)

            await get_full_artist_profile("Radiohead")

    assert call_count == 2


@pytest.mark.asyncio
async def test_different_artists_use_different_cache_entries():
    call_count = 0

    async def counting_info(name):
        nonlocal call_count
        call_count += 1
        return LASTFM_INFO

    patches = _patch_all()
    with patches[0], patches[1], patches[2], patches[3], patches[4], \
         patches[5], patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=counting_info)):
            await get_full_artist_profile("Radiohead")
            await get_full_artist_profile("Portishead")

    assert call_count == 2


# ---------------------------------------------------------------------------
# Tests: get_full_artist_profile — parallel execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lastfm_and_deezer_called_in_parallel():
    """
    Verify that Last.fm and Deezer artist calls are fired concurrently.
    We simulate latency (0.1 s each) and assert total wall time is < 0.18 s
    (sequential would be ≥ 0.2 s).
    """
    async def slow_lastfm_info(name):
        await asyncio.sleep(0.1)
        return LASTFM_INFO

    async def slow_lastfm_tracks(name, limit=5):
        await asyncio.sleep(0.1)
        return LASTFM_TRACKS

    async def slow_deezer_artist(name):
        await asyncio.sleep(0.1)
        return DEEZER_ARTIST

    patches = _patch_all()
    with patches[5], patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=slow_lastfm_info)), \
             patch("src.data.music_service.lastfm.get_top_tracks", AsyncMock(side_effect=slow_lastfm_tracks)), \
             patch("src.data.music_service.deezer.search_artist", AsyncMock(side_effect=slow_deezer_artist)):
            t0 = time.monotonic()
            await get_full_artist_profile("Radiohead")
            elapsed = time.monotonic() - t0

    assert elapsed < 0.18, f"Expected parallel execution, but took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Tests: graceful degradation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lastfm_failure_returns_partial_profile():
    """If Last.fm fails, profile should still return with Deezer data."""
    patches = _patch_all()
    with patches[4], patches[5], patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_artist_info", AsyncMock(side_effect=Exception("Last.fm down"))), \
             patch("src.data.music_service.lastfm.get_top_tracks", AsyncMock(side_effect=Exception("Last.fm down"))):
            profile = await get_full_artist_profile("Radiohead")

    assert profile["name"] == "Radiohead"
    assert profile["bio"] == ""
    assert profile["tags"] == []
    assert profile["fan_count"] == 2500000  # from Deezer


@pytest.mark.asyncio
async def test_deezer_failure_returns_partial_profile():
    """If Deezer fails, profile should still return with Last.fm data."""
    patches = _patch_all()
    with patches[0], patches[1], patches[8]:
        with patch("src.data.music_service.deezer.search_artist", AsyncMock(side_effect=Exception("Deezer down"))), \
             patch("src.data.music_service.deezer.get_artist_top_tracks", AsyncMock(side_effect=Exception("Deezer down"))), \
             patch("src.data.music_service.deezer.search_tracks", AsyncMock(side_effect=Exception("Deezer down"))), \
             patch("src.data.music_service.deezer.download_preview", AsyncMock(side_effect=Exception("Deezer down"))):
            profile = await get_full_artist_profile("Radiohead")

    assert profile["name"] == "Radiohead"
    assert profile["bio"] == LASTFM_INFO["bio_summary"]
    assert profile["tags"] == LASTFM_INFO["tags"]
    assert profile["fan_count"] == 0   # no Deezer
    assert profile["audio_profile"] == {}  # no previews → no analysis


# ---------------------------------------------------------------------------
# Tests: get_similar_with_features
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_similar_with_features_returns_list():
    patches = _patch_all()
    with patches[2], patches[6], patches[7], patches[8]:
        result = await get_similar_with_features("Radiohead", limit=2)

    assert isinstance(result, list)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_get_similar_with_features_sorted_by_match():
    patches = _patch_all()
    with patches[2], patches[6], patches[7], patches[8]:
        result = await get_similar_with_features("Radiohead", limit=2)

    matches = [r["match"] for r in result]
    assert matches == sorted(matches, reverse=True)


@pytest.mark.asyncio
async def test_get_similar_with_features_has_audio():
    patches = _patch_all()
    with patches[2], patches[6], patches[7], patches[8]:
        result = await get_similar_with_features("Radiohead", limit=2)

    for entry in result:
        assert "audio_features" in entry


@pytest.mark.asyncio
async def test_get_similar_with_features_cached():
    call_count = 0

    async def counting_similar(name, limit=10):
        nonlocal call_count
        call_count += 1
        return SIMILAR_ARTISTS

    patches = _patch_all()
    with patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_similar_artists", AsyncMock(side_effect=counting_similar)):
            await get_similar_with_features("Radiohead")
            await get_similar_with_features("Radiohead")

    assert call_count == 1


@pytest.mark.asyncio
async def test_get_similar_lastfm_failure_returns_empty():
    with patch("src.data.music_service.lastfm.get_similar_artists", AsyncMock(side_effect=Exception("err"))):
        result = await get_similar_with_features("Radiohead")
    assert result == []


# ---------------------------------------------------------------------------
# Tests: search_artists_by_vibe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_artists_by_vibe_returns_list():
    patches = _patch_all()
    with patches[3], patches[6], patches[7], patches[8]:
        result = await search_artists_by_vibe(["shoegaze"], energy_range=(0.0, 1.0), valence_range=(0.0, 1.0))

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_search_artists_by_vibe_energy_filter():
    """Artists with energy outside the range should be excluded."""
    patches = _patch_all()

    # Audio features report energy=0.65; set range that excludes it
    with patches[3], patches[6], patches[7], patches[8]:
        result = await search_artists_by_vibe(
            ["shoegaze"],
            energy_range=(0.80, 1.0),  # 0.65 is below this range
            valence_range=(0.0, 1.0),
        )

    assert result == []


@pytest.mark.asyncio
async def test_search_artists_by_vibe_valence_filter():
    """Artists with valence outside the range should be excluded."""
    patches = _patch_all()

    with patches[3], patches[6], patches[7], patches[8]:
        result = await search_artists_by_vibe(
            ["shoegaze"],
            energy_range=(0.0, 1.0),
            valence_range=(0.90, 1.0),  # 0.55 is below this range
        )

    assert result == []


@pytest.mark.asyncio
async def test_search_artists_by_vibe_passes_filter():
    """Artists within range should be included."""
    patches = _patch_all()

    with patches[3], patches[6], patches[7], patches[8]:
        result = await search_artists_by_vibe(
            ["shoegaze"],
            energy_range=(0.5, 0.8),   # 0.65 in range
            valence_range=(0.4, 0.7),  # 0.55 in range
        )

    assert len(result) >= 1


@pytest.mark.asyncio
async def test_search_artists_by_vibe_cached():
    call_count = 0

    async def counting_tag(tag, limit=10):
        nonlocal call_count
        call_count += 1
        return TAG_ARTISTS

    patches = _patch_all()
    with patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_tag_top_artists", AsyncMock(side_effect=counting_tag)):
            await search_artists_by_vibe(["shoegaze"])
            await search_artists_by_vibe(["shoegaze"])

    assert call_count == 1


@pytest.mark.asyncio
async def test_search_artists_by_vibe_multiple_tags_parallel():
    """All tag requests should fire in parallel — total time < sum of individual."""
    call_times = []

    async def slow_tag(tag, limit=10):
        call_times.append(time.monotonic())
        await asyncio.sleep(0.1)
        return TAG_ARTISTS

    patches = _patch_all()
    with patches[6], patches[7], patches[8]:
        with patch("src.data.music_service.lastfm.get_tag_top_artists", AsyncMock(side_effect=slow_tag)):
            t0 = time.monotonic()
            await search_artists_by_vibe(["shoegaze", "dream pop", "post-rock"])
            elapsed = time.monotonic() - t0

    # 3 × 0.1 s sequential = 0.3 s; parallel should be < 0.18 s
    assert elapsed < 0.18, f"Tag fetches not parallel: {elapsed:.3f}s"
