"""
Tests for PreferenceEngine.

Uses tmp_path for all file I/O — never touches data/preferences.json.
Music service is mocked to avoid network calls.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.preference_engine import PreferenceEngine, _SWEET_SPOT_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def prefs(tmp_path: Path) -> PreferenceEngine:
    """PreferenceEngine backed by a temporary file."""
    return PreferenceEngine(path=tmp_path / "prefs.json")


def _rating(artist: str, liked: bool, tags=None, features=None) -> dict:
    """Build a minimal rating entry for _compute_profile unit tests."""
    return {
        "artist": artist,
        "liked": liked,
        "notes": "",
        "timestamp": "2026-01-01T00:00:00",
        "tags": tags or [],
        "audio_features": features or {},
    }


# ── Persistence ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_writes_file(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    await engine.save_preference("Portishead", liked=True, audio_features={"energy": 0.4})

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["ratings"]) == 1
    assert data["ratings"][0]["artist"] == "Portishead"
    assert data["ratings"][0]["liked"] is True


@pytest.mark.asyncio
async def test_save_includes_audio_features_in_entry(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    await engine.save_preference(
        "Artist", liked=True,
        audio_features={"energy": 0.6, "valence": 0.3},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ratings"][0]["audio_features"]["energy"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_load_reads_existing_file(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps({
        "ratings": [{"artist": "Boards of Canada", "liked": True, "notes": "",
                     "timestamp": "x", "tags": ["idm"], "audio_features": {}}],
        "taste_profile": {},
    }), encoding="utf-8")

    engine = PreferenceEngine(path=path)
    history = engine.get_preference_history()

    assert len(history) == 1
    assert history[0]["artist"] == "Boards of Canada"


@pytest.mark.asyncio
async def test_missing_file_returns_empty_state(tmp_path):
    engine = PreferenceEngine(path=tmp_path / "nonexistent.json")
    assert engine.get_preference_history() == []
    assert engine.get_taste_profile() == {}


@pytest.mark.asyncio
async def test_corrupt_file_returns_empty_state(tmp_path):
    path = tmp_path / "prefs.json"
    path.write_text("NOT VALID JSON", encoding="utf-8")
    engine = PreferenceEngine(path=path)
    assert engine.get_preference_history() == []


# ── Rating deduplication ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_overwrites_existing_rating_for_same_artist(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    await engine.save_preference("Radiohead", liked=True, audio_features={})
    await engine.save_preference("Radiohead", liked=False, audio_features={})

    history = engine.get_preference_history()
    assert len(history) == 1
    assert history[0]["liked"] is False


@pytest.mark.asyncio
async def test_overwrite_is_case_insensitive(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    await engine.save_preference("radiohead", liked=True, audio_features={})
    await engine.save_preference("Radiohead", liked=False, audio_features={})

    assert len(engine.get_preference_history()) == 1


@pytest.mark.asyncio
async def test_multiple_different_artists_accumulate(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    await engine.save_preference("Artist A", liked=True, audio_features={})
    await engine.save_preference("Artist B", liked=False, audio_features={})
    await engine.save_preference("Artist C", liked=True, audio_features={})

    assert len(engine.get_preference_history()) == 3


# ── Music service integration ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_fetches_profile_when_audio_features_omitted(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    mock_profile = {
        "tags": ["trip-hop", "electronic"],
        "audio_profile": {"energy": 0.4, "valence": 0.25},
    }
    with patch(
        "src.data.music_service.get_full_artist_profile",
        new=AsyncMock(return_value=mock_profile),
    ):
        await engine.save_preference("Portishead", liked=True)

    history = engine.get_preference_history()
    assert history[0]["tags"] == ["trip-hop", "electronic"]


@pytest.mark.asyncio
async def test_save_graceful_when_music_service_fails(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)

    with patch(
        "src.data.music_service.get_full_artist_profile",
        new=AsyncMock(side_effect=RuntimeError("network error")),
    ):
        await engine.save_preference("Portishead", liked=True)  # must not raise

    history = engine.get_preference_history()
    assert len(history) == 1
    assert history[0]["tags"] == []
    assert history[0]["audio_features"] == {}


# ── _compute_profile ──────────────────────────────────────────────────────────


def test_compute_profile_counts(prefs):
    ratings = [
        _rating("A", liked=True),
        _rating("B", liked=True),
        _rating("C", liked=False),
    ]
    profile = prefs._compute_profile(ratings)

    assert profile["total_ratings"] == 3
    assert profile["liked_count"] == 2
    assert profile["disliked_count"] == 1


def test_compute_profile_empty_ratings(prefs):
    profile = prefs._compute_profile([])

    assert profile["total_ratings"] == 0
    assert profile["liked_count"] == 0
    assert profile["preferred_audio"] == {}
    assert profile["disliked_audio"] == {}


def test_compute_profile_averages_liked_audio(prefs):
    ratings = [
        _rating("A", liked=True, features={"energy": 0.4, "valence": 0.2}),
        _rating("B", liked=True, features={"energy": 0.6, "valence": 0.4}),
    ]
    profile = prefs._compute_profile(ratings)

    assert profile["preferred_audio"]["energy"] == pytest.approx(0.5, abs=0.001)
    assert profile["preferred_audio"]["valence"] == pytest.approx(0.3, abs=0.001)


def test_compute_profile_averages_disliked_audio(prefs):
    ratings = [
        _rating("A", liked=False, features={"energy": 0.9}),
        _rating("B", liked=False, features={"energy": 0.7}),
    ]
    profile = prefs._compute_profile(ratings)

    assert profile["disliked_audio"]["energy"] == pytest.approx(0.8, abs=0.001)


def test_compute_profile_preferred_tags_ranked_by_frequency(prefs):
    ratings = [
        _rating("A", liked=True, tags=["trip-hop", "electronic"]),
        _rating("B", liked=True, tags=["trip-hop", "dark"]),
        _rating("C", liked=True, tags=["electronic"]),
    ]
    profile = prefs._compute_profile(ratings)

    pref_tags = profile["preferred_tags"]
    # trip-hop (2) and electronic (2) both rank above dark (1)
    assert "trip-hop" in pref_tags[:2]
    assert "electronic" in pref_tags[:2]
    assert "dark" in pref_tags


def test_compute_profile_disliked_tags_separate_from_preferred(prefs):
    ratings = [
        _rating("A", liked=True,  tags=["trip-hop"]),
        _rating("B", liked=False, tags=["pop", "mainstream"]),
    ]
    profile = prefs._compute_profile(ratings)

    assert "trip-hop" in profile["preferred_tags"]
    assert "pop" in profile["disliked_tags"]
    assert "pop" not in profile["preferred_tags"]


def test_compute_profile_sweet_spot_when_liked_disliked_diverge(prefs):
    """Sweet spot is flagged when liked vs disliked differ by > threshold."""
    ratings = [
        _rating("A", liked=True,  features={"energy": 0.1}),
        _rating("B", liked=False, features={"energy": 0.9}),
    ]
    profile = prefs._compute_profile(ratings)

    assert "energy" in profile["sweet_spots"]
    # User prefers lower energy → negative delta
    assert profile["sweet_spots"]["energy"] < 0


def test_compute_profile_no_sweet_spot_when_similar(prefs):
    """No sweet spot when liked/disliked audio are within the threshold."""
    delta_below = _SWEET_SPOT_THRESHOLD * 0.4
    ratings = [
        _rating("A", liked=True,  features={"energy": 0.5}),
        _rating("B", liked=False, features={"energy": 0.5 + delta_below}),
    ]
    profile = prefs._compute_profile(ratings)

    assert "energy" not in profile.get("sweet_spots", {})


def test_compute_profile_no_sweet_spot_without_both_liked_and_disliked(prefs):
    """Sweet spots require both liked and disliked entries with the feature."""
    ratings = [_rating("A", liked=True, features={"energy": 0.3})]
    profile = prefs._compute_profile(ratings)

    assert profile["sweet_spots"] == {}


# ── _build_summary ────────────────────────────────────────────────────────────


def test_summary_describes_high_energy(prefs):
    ratings = [_rating("A", liked=True, features={"energy": 0.8, "valence": 0.6, "tempo": 140.0})]
    profile = prefs._compute_profile(ratings)
    assert "high-energy" in profile["summary"]


def test_summary_describes_low_energy(prefs):
    ratings = [_rating("A", liked=True, features={"energy": 0.3})]
    profile = prefs._compute_profile(ratings)
    assert "low-energy" in profile["summary"] or "atmospheric" in profile["summary"]


def test_summary_describes_melancholic_valence(prefs):
    ratings = [_rating("A", liked=True, features={"energy": 0.5, "valence": 0.2})]
    profile = prefs._compute_profile(ratings)
    assert "dark" in profile["summary"] or "melancholic" in profile["summary"]


def test_summary_describes_uplifting_valence(prefs):
    ratings = [_rating("A", liked=True, features={"valence": 0.8})]
    profile = prefs._compute_profile(ratings)
    assert "uplifting" in profile["summary"] or "positive" in profile["summary"]


def test_summary_no_liked_artists(prefs):
    profile = prefs._compute_profile([])
    assert "No liked" in profile["summary"]


def test_summary_includes_preferred_tags(prefs):
    ratings = [_rating("A", liked=True, tags=["shoegaze", "dreamy"])]
    profile = prefs._compute_profile(ratings)
    assert "shoegaze" in profile["summary"]


# ── get_taste_profile ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_taste_profile_has_all_expected_keys(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("Artist", liked=True, audio_features={"energy": 0.5})

    profile = engine.get_taste_profile()
    expected_keys = {
        "preferred_tags", "preferred_audio", "disliked_tags", "disliked_audio",
        "sweet_spots", "total_ratings", "liked_count", "disliked_count", "summary",
    }
    assert expected_keys.issubset(profile.keys())


@pytest.mark.asyncio
async def test_get_taste_profile_empty_before_any_ratings(tmp_path):
    engine = PreferenceEngine(path=tmp_path / "prefs.json")
    assert engine.get_taste_profile() == {}


# ── get_recommendation_context ────────────────────────────────────────────────


def test_recommendation_context_empty_with_no_ratings(prefs):
    assert prefs.get_recommendation_context() == ""


@pytest.mark.asyncio
async def test_recommendation_context_contains_header(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("Artist", liked=True, audio_features={"energy": 0.4})

    ctx = engine.get_recommendation_context()
    assert "## User taste profile" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_contains_preferred_tags(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("Artist", liked=True, audio_features={})
    # Inject tags directly so we don't need the music service
    engine._data["ratings"][0]["tags"] = ["trip-hop", "electronic"]
    engine._data["taste_profile"] = engine._compute_profile(engine._data["ratings"])

    ctx = engine.get_recommendation_context()
    assert "trip-hop" in ctx
    assert "Gravitates toward" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_contains_disliked_tags(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("Pop Star", liked=False, audio_features={})
    engine._data["ratings"][0]["tags"] = ["pop", "mainstream"]
    engine._data["taste_profile"] = engine._compute_profile(engine._data["ratings"])

    ctx = engine.get_recommendation_context()
    assert "pop" in ctx
    assert "Tends to dislike" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_contains_audio_features(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference(
        "Artist", liked=True,
        audio_features={"energy": 0.4, "valence": 0.2, "danceability": 0.5,
                        "tempo": 80.0, "acousticness": 0.3},
    )

    ctx = engine.get_recommendation_context()
    assert "energy" in ctx
    assert "0.40" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_contains_sweet_spots(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    # Big divergence in energy ensures a sweet spot
    await engine.save_preference("Liked", liked=True, audio_features={"energy": 0.1})
    await engine.save_preference("Disliked", liked=False, audio_features={"energy": 0.9})

    ctx = engine.get_recommendation_context()
    assert "Strongest preferences" in ctx
    assert "energy" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_contains_rating_counts(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("A", liked=True, audio_features={})
    await engine.save_preference("B", liked=False, audio_features={})

    ctx = engine.get_recommendation_context()
    assert "1 liked" in ctx
    assert "1 disliked" in ctx


@pytest.mark.asyncio
async def test_recommendation_context_reminds_to_surprise(tmp_path):
    """Context always ends with the instruction not to just confirm existing taste."""
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("A", liked=True, audio_features={})

    ctx = engine.get_recommendation_context()
    assert "discover" in ctx.lower()


# ── get_preference_history ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_preference_history_returns_all_entries(tmp_path):
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("A", liked=True, audio_features={})
    await engine.save_preference("B", liked=True, audio_features={})
    await engine.save_preference("C", liked=False, audio_features={})

    history = engine.get_preference_history()
    assert len(history) == 3
    assert {h["artist"] for h in history} == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_get_preference_history_returns_copy(tmp_path):
    """Mutating the returned list does not affect internal state."""
    path = tmp_path / "prefs.json"
    engine = PreferenceEngine(path=path)
    await engine.save_preference("A", liked=True, audio_features={})

    history = engine.get_preference_history()
    history.clear()

    assert len(engine.get_preference_history()) == 1
