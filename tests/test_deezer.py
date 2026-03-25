"""Tests for the Deezer API client using mocked httpx responses."""

import pytest
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock, mock_open

from src.data.deezer_client import (
    search_tracks,
    search_artist,
    get_artist,
    get_artist_top_tracks,
    get_track,
    download_preview,
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

SEARCH_TRACKS_RESPONSE = {
    "data": [
        {
            "title": "Creep",
            "artist": {"name": "Radiohead"},
            "album": {"title": "Pablo Honey", "cover_medium": "https://cdn.deezer.com/cover1.jpg"},
            "duration": 238,
            "preview": "https://cdns-preview-d.dzcdn.net/creep.mp3",
            "link": "https://www.deezer.com/track/123",
        },
        {
            "title": "Karma Police",
            "artist": {"name": "Radiohead"},
            "album": {"title": "OK Computer", "cover_medium": "https://cdn.deezer.com/cover2.jpg"},
            "duration": 264,
            "preview": "https://cdns-preview-d.dzcdn.net/karma.mp3",
            "link": "https://www.deezer.com/track/456",
        },
    ]
}

SEARCH_ARTIST_RESPONSE = {
    "data": [
        {
            "id": 71,
            "name": "Radiohead",
            "picture_medium": "https://cdn.deezer.com/artist71.jpg",
            "nb_fan": 2500000,
        }
    ]
}

GET_ARTIST_RESPONSE = {
    "id": 71,
    "name": "Radiohead",
    "picture_medium": "https://cdn.deezer.com/artist71.jpg",
    "nb_album": 9,
    "nb_fan": 2500000,
    "link": "https://www.deezer.com/artist/71",
}

ARTIST_TOP_TRACKS_RESPONSE = {
    "data": [
        {
            "title": "Creep",
            "duration": 238,
            "preview": "https://cdns-preview-d.dzcdn.net/creep.mp3",
            "album": {"title": "Pablo Honey", "cover_medium": "https://cdn.deezer.com/cover1.jpg"},
        },
        {
            "title": "Karma Police",
            "duration": 264,
            "preview": "https://cdns-preview-d.dzcdn.net/karma.mp3",
            "album": {"title": "OK Computer", "cover_medium": "https://cdn.deezer.com/cover2.jpg"},
        },
    ]
}

GET_TRACK_RESPONSE = {
    "id": 123,
    "title": "Creep",
    "artist": {"name": "Radiohead"},
    "album": {"title": "Pablo Honey"},
    "duration": 238,
    "preview": "https://cdns-preview-d.dzcdn.net/creep.mp3",
    "bpm": 92.0,
    "link": "https://www.deezer.com/track/123",
}

ERROR_RESPONSE = {
    "error": {"code": 800, "message": "Artist not found", "type": "DataException"}
}


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
    """Context manager patching httpx.AsyncClient.get."""
    mock_resp = _make_mock_response(json_data, status_code)
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return patch("src.data.deezer_client.httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# Tests: search_tracks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_tracks_returns_results():
    with _mock_client(SEARCH_TRACKS_RESPONSE):
        results = await search_tracks("Radiohead")

    assert len(results) == 2
    assert results[0]["title"] == "Creep"
    assert results[0]["artist"] == "Radiohead"


@pytest.mark.asyncio
async def test_search_tracks_fields():
    with _mock_client(SEARCH_TRACKS_RESPONSE):
        results = await search_tracks("Radiohead")

    for t in results:
        assert "title" in t
        assert "artist" in t
        assert "album" in t
        assert "duration" in t
        assert "preview_url" in t
        assert "cover_url" in t
        assert "link" in t


@pytest.mark.asyncio
async def test_search_tracks_limit():
    with _mock_client(SEARCH_TRACKS_RESPONSE):
        results = await search_tracks("Radiohead", limit=1)

    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_tracks_empty():
    with _mock_client({"data": []}):
        results = await search_tracks("xyznotatrack")

    assert results == []


@pytest.mark.asyncio
async def test_search_tracks_api_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError, match="Artist not found"):
            await search_tracks("bad query")


# ---------------------------------------------------------------------------
# Tests: search_artist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_artist_returns_top_match():
    with _mock_client(SEARCH_ARTIST_RESPONSE):
        artist = await search_artist("Radiohead")

    assert artist["id"] == 71
    assert artist["name"] == "Radiohead"
    assert artist["nb_fan"] == 2500000
    assert "picture_url" in artist


@pytest.mark.asyncio
async def test_search_artist_empty_returns_empty_dict():
    with _mock_client({"data": []}):
        result = await search_artist("xyznotanartist")

    assert result == {}


# ---------------------------------------------------------------------------
# Tests: get_artist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_artist_fields():
    with _mock_client(GET_ARTIST_RESPONSE):
        artist = await get_artist(71)

    assert artist["id"] == 71
    assert artist["name"] == "Radiohead"
    assert artist["nb_album"] == 9
    assert artist["nb_fan"] == 2500000
    assert "deezer.com" in artist["link"]


@pytest.mark.asyncio
async def test_get_artist_api_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_artist(99999)


# ---------------------------------------------------------------------------
# Tests: get_artist_top_tracks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_artist_top_tracks_returns_list():
    with _mock_client(ARTIST_TOP_TRACKS_RESPONSE):
        tracks = await get_artist_top_tracks(71)

    assert isinstance(tracks, list)
    assert len(tracks) == 2


@pytest.mark.asyncio
async def test_get_artist_top_tracks_fields():
    with _mock_client(ARTIST_TOP_TRACKS_RESPONSE):
        tracks = await get_artist_top_tracks(71)

    for t in tracks:
        assert "title" in t
        assert "duration" in t
        assert "preview_url" in t
        assert "album" in t
        assert "title" in t["album"]
        assert "cover_url" in t["album"]


@pytest.mark.asyncio
async def test_get_artist_top_tracks_limit():
    with _mock_client(ARTIST_TOP_TRACKS_RESPONSE):
        tracks = await get_artist_top_tracks(71, limit=1)

    assert len(tracks) == 1


# ---------------------------------------------------------------------------
# Tests: get_track
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_track_fields():
    with _mock_client(GET_TRACK_RESPONSE):
        track = await get_track(123)

    assert track["id"] == 123
    assert track["title"] == "Creep"
    assert track["artist"] == "Radiohead"
    assert track["album"] == "Pablo Honey"
    assert track["duration"] == 238
    assert track["bpm"] == 92.0
    assert "deezer.com" in track["link"]


@pytest.mark.asyncio
async def test_get_track_preview_url():
    with _mock_client(GET_TRACK_RESPONSE):
        track = await get_track(123)

    assert track["preview_url"].endswith(".mp3")


@pytest.mark.asyncio
async def test_get_track_api_error_raises():
    with _mock_client(ERROR_RESPONSE):
        with pytest.raises(ValueError):
            await get_track(99999)


# ---------------------------------------------------------------------------
# Tests: download_preview
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_download_preview_saves_file(tmp_path):
    save_path = str(tmp_path / "test_preview.mp3")
    mp3_bytes = b"ID3" + b"\x00" * 100

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = mp3_bytes

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("src.data.deezer_client.httpx.AsyncClient", return_value=mock_client):
        result = await download_preview("https://cdns-preview-d.dzcdn.net/creep.mp3", save_path)

    assert result == save_path
    assert Path(save_path).exists()
    assert Path(save_path).read_bytes() == mp3_bytes


@pytest.mark.asyncio
async def test_download_preview_skips_if_cached(tmp_path):
    save_path = tmp_path / "cached.mp3"
    save_path.write_bytes(b"existing content")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock()

    with patch("src.data.deezer_client.httpx.AsyncClient", return_value=mock_client):
        result = await download_preview("https://example.com/track.mp3", str(save_path))

    mock_client.get.assert_not_called()
    assert result == str(save_path)


@pytest.mark.asyncio
async def test_download_preview_creates_parent_dirs(tmp_path):
    save_path = str(tmp_path / "nested" / "dirs" / "track.mp3")
    mp3_bytes = b"\xff\xfb" + b"\x00" * 50

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = mp3_bytes

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("src.data.deezer_client.httpx.AsyncClient", return_value=mock_client):
        result = await download_preview("https://example.com/track.mp3", save_path)

    assert Path(result).exists()
