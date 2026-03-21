"""Tests for the Last.fm API client using mocked httpx responses."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from src.data.lastfm_client import (
    search_artist,
    get_artist_info,
    get_similar_artists,
    get_artist_tags,
    get_top_tracks,
    get_tag_top_artists,
)

# ---------------------------------------------------------------------------
# Fixtures / mock data
# ---------------------------------------------------------------------------

SEARCH_ARTIST_RESPONSE = {
    "results": {
        "artistmatches": {
            "artist": [
                {"name": "Radiohead", "listeners": "4800000", "url": "https://www.last.fm/music/Radiohead"},
                {"name": "Radiohead Tribute", "listeners": "1200", "url": "https://www.last.fm/music/Radiohead+Tribute"},
                {"name": "Radioheadz", "listeners": "300", "url": "https://www.last.fm/music/Radioheadz"},
                {"name": "Kid Radiohead", "listeners": "50", "url": "https://www.last.fm/music/Kid+Radiohead"},
                {"name": "Radiohead Cover Band", "listeners": "20", "url": "https://www.last.fm/music/Radiohead+Cover+Band"},
            ]
        }
    }
}

ARTIST_INFO_RESPONSE = {
    "artist": {
        "name": "Radiohead",
        "stats": {"listeners": "4800000", "playcount": "650000000"},
        "bio": {
            "summary": "Radiohead are an English rock band from Abingdon, Oxfordshire. <a href='https://www.last.fm/music/Radiohead'>Read more</a>"
        },
        "tags": {
            "tag": [
                {"name": "alternative rock", "url": "https://www.last.fm/tag/alternative+rock"},
                {"name": "art rock", "url": "https://www.last.fm/tag/art+rock"},
                {"name": "electronic", "url": "https://www.last.fm/tag/electronic"},
            ]
        },
        "similar": {
            "artist": [
                {"name": "Thom Yorke", "url": "https://www.last.fm/music/Thom+Yorke"},
                {"name": "Portishead", "url": "https://www.last.fm/music/Portishead"},
            ]
        },
    }
}

SIMILAR_ARTISTS_RESPONSE = {
    "similarartists": {
        "artist": [
            {"name": "Thom Yorke", "match": "1.000000", "url": "https://www.last.fm/music/Thom+Yorke"},
            {"name": "Portishead", "match": "0.721345", "url": "https://www.last.fm/music/Portishead"},
            {"name": "Massive Attack", "match": "0.698231", "url": "https://www.last.fm/music/Massive+Attack"},
        ]
    }
}

TOP_TAGS_RESPONSE = {
    "toptags": {
        "tag": [
            {"name": "alternative rock", "count": "100", "url": "..."},
            {"name": "art rock", "count": "85", "url": "..."},
            {"name": "electronic", "count": "72", "url": "..."},
            {"name": "indie", "count": "60", "url": "..."},
        ]
    }
}

TOP_TRACKS_RESPONSE = {
    "toptracks": {
        "track": [
            {"name": "Creep", "playcount": "90000000", "listeners": "5000000", "url": "https://www.last.fm/music/Radiohead/_/Creep"},
            {"name": "Karma Police", "playcount": "60000000", "listeners": "4000000", "url": "https://www.last.fm/music/Radiohead/_/Karma+Police"},
            {"name": "No Surprises", "playcount": "50000000", "listeners": "3500000", "url": "https://www.last.fm/music/Radiohead/_/No+Surprises"},
        ]
    }
}

TAG_TOP_ARTISTS_RESPONSE = {
    "topartists": {
        "artist": [
            {"name": "My Bloody Valentine", "@attr": {"rank": "1"}, "url": "https://www.last.fm/music/My+Bloody+Valentine"},
            {"name": "Slowdive", "@attr": {"rank": "2"}, "url": "https://www.last.fm/music/Slowdive"},
            {"name": "Ride", "@attr": {"rank": "3"}, "url": "https://www.last.fm/music/Ride"},
        ]
    }
}

ERROR_RESPONSE = {"error": 6, "message": "Artist not found"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _mock_client(json_data: dict, status_code: int = 200):
    """Context manager that patches httpx.AsyncClient.get."""
    mock_resp = _make_mock_response(json_data, status_code)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return patch("src.data.lastfm_client.httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# Tests: search_artist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_artist_returns_top_5():
    with _mock_client(SEARCH_ARTIST_RESPONSE):
        results = await search_artist("Radiohead")

    assert len(results) == 5
    assert results[0]["name"] == "Radiohead"
    assert results[0]["listeners"] == "4800000"
    assert "last.fm" in results[0]["url"]


@pytest.mark.asyncio
async def test_search_artist_fields():
    with _mock_client(SEARCH_ARTIST_RESPONSE):
        results = await search_artist("Radiohead")

    for r in results:
        assert "name" in r
        assert "listeners" in r
        assert "url" in r


@pytest.mark.asyncio
async def test_search_artist_empty_results():
    empty = {"results": {"artistmatches": {"artist": []}}}
    with _mock_client(empty):
        results = await search_artist("xyznotanartist")

    assert results == []


# ---------------------------------------------------------------------------
# Tests: get_artist_info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_artist_info_basic():
    with _mock_client(ARTIST_INFO_RESPONSE):
        info = await get_artist_info("Radiohead")

    assert info["name"] == "Radiohead"
    assert info["listeners"] == "4800000"
    assert info["playcount"] == "650000000"
    assert "Radiohead are an English rock band" in info["bio_summary"]


@pytest.mark.asyncio
async def test_get_artist_info_tags():
    with _mock_client(ARTIST_INFO_RESPONSE):
        info = await get_artist_info("Radiohead")

    assert "alternative rock" in info["tags"]
    assert "art rock" in info["tags"]


@pytest.mark.asyncio
async def test_get_artist_info_similar():
    with _mock_client(ARTIST_INFO_RESPONSE):
        info = await get_artist_info("Radiohead")

    names = [a["name"] for a in info["similar_artists"]]
    assert "Thom Yorke" in names
    assert "Portishead" in names


@pytest.mark.asyncio
async def test_get_artist_info_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError, match="Artist not found"):
            await get_artist_info("xyznotanartist")


# ---------------------------------------------------------------------------
# Tests: get_similar_artists
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_similar_artists_returns_list():
    with _mock_client(SIMILAR_ARTISTS_RESPONSE):
        similar = await get_similar_artists("Radiohead")

    assert isinstance(similar, list)
    assert len(similar) == 3


@pytest.mark.asyncio
async def test_get_similar_artists_match_score():
    with _mock_client(SIMILAR_ARTISTS_RESPONSE):
        similar = await get_similar_artists("Radiohead")

    assert similar[0]["name"] == "Thom Yorke"
    assert similar[0]["match"] == pytest.approx(1.0)
    assert similar[1]["match"] == pytest.approx(0.721345)


@pytest.mark.asyncio
async def test_get_similar_artists_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_similar_artists("xyznotanartist")


# ---------------------------------------------------------------------------
# Tests: get_artist_tags
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_artist_tags_returns_names():
    with _mock_client(TOP_TAGS_RESPONSE):
        tags = await get_artist_tags("Radiohead")

    assert isinstance(tags, list)
    assert tags[0] == "alternative rock"
    assert tags[1] == "art rock"


@pytest.mark.asyncio
async def test_get_artist_tags_all_strings():
    with _mock_client(TOP_TAGS_RESPONSE):
        tags = await get_artist_tags("Radiohead")

    assert all(isinstance(t, str) for t in tags)


@pytest.mark.asyncio
async def test_get_artist_tags_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_artist_tags("xyznotanartist")


# ---------------------------------------------------------------------------
# Tests: get_top_tracks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_top_tracks_returns_tracks():
    with _mock_client(TOP_TRACKS_RESPONSE):
        tracks = await get_top_tracks("Radiohead")

    assert len(tracks) == 3
    assert tracks[0]["name"] == "Creep"
    assert tracks[0]["playcount"] == "90000000"
    assert tracks[0]["listeners"] == "5000000"


@pytest.mark.asyncio
async def test_get_top_tracks_fields():
    with _mock_client(TOP_TRACKS_RESPONSE):
        tracks = await get_top_tracks("Radiohead")

    for t in tracks:
        assert "name" in t
        assert "playcount" in t
        assert "listeners" in t
        assert "url" in t


@pytest.mark.asyncio
async def test_get_top_tracks_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_top_tracks("xyznotanartist")


# ---------------------------------------------------------------------------
# Tests: get_tag_top_artists
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_tag_top_artists_returns_artists():
    with _mock_client(TAG_TOP_ARTISTS_RESPONSE):
        artists = await get_tag_top_artists("shoegaze")

    assert len(artists) == 3
    assert artists[0]["name"] == "My Bloody Valentine"
    assert artists[0]["rank"] == "1"


@pytest.mark.asyncio
async def test_get_tag_top_artists_fields():
    with _mock_client(TAG_TOP_ARTISTS_RESPONSE):
        artists = await get_tag_top_artists("shoegaze")

    for a in artists:
        assert "name" in a
        assert "rank" in a
        assert "url" in a


@pytest.mark.asyncio
async def test_get_tag_top_artists_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_tag_top_artists("notarealtag")
