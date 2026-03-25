"""
Deezer API client.

Handles requests to the Deezer public API, providing methods for searching
tracks, fetching album/artist details, and retrieving 30-second audio preview URLs.
No authentication required — Deezer's catalog API is fully open.
"""

import asyncio
import os
from pathlib import Path

import httpx

BASE_URL = "https://api.deezer.com/"
PREVIEWS_DIR = Path("data/previews")


async def _get(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
    """Make a GET request with retry on 429 rate limiting."""
    for attempt in range(5):
        response = await client.get(url, params=params)
        if response.status_code == 429:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            raise ValueError(f"Deezer API error {err.get('code', '')}: {err.get('message', '')}")
        return data
    raise httpx.HTTPStatusError(
        "Rate limited after retries", request=response.request, response=response
    )


async def search_tracks(query: str, limit: int = 5) -> list[dict]:
    """Search tracks by query string. Returns up to `limit` results."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"{BASE_URL}search", params={"q": query, "limit": limit})

    tracks = data.get("data", [])[:limit]
    return [
        {
            "title": t.get("title", ""),
            "artist": t.get("artist", {}).get("name", ""),
            "album": t.get("album", {}).get("title", ""),
            "duration": t.get("duration", 0),
            "preview_url": t.get("preview", ""),
            "cover_url": t.get("album", {}).get("cover_medium", ""),
            "link": t.get("link", ""),
        }
        for t in tracks
    ]


async def search_artist(name: str) -> dict:
    """Search for an artist by name. Returns the top match."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"{BASE_URL}search/artist", params={"q": name})

    artists = data.get("data", [])
    if not artists:
        return {}
    a = artists[0]
    return {
        "id": a.get("id"),
        "name": a.get("name", ""),
        "picture_url": a.get("picture_medium", ""),
        "nb_fan": a.get("nb_fan", 0),
    }


async def get_artist(artist_id: int) -> dict:
    """Get full artist details by Deezer artist ID."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"{BASE_URL}artist/{artist_id}")

    return {
        "id": data.get("id"),
        "name": data.get("name", ""),
        "picture_url": data.get("picture_medium", ""),
        "nb_album": data.get("nb_album", 0),
        "nb_fan": data.get("nb_fan", 0),
        "link": data.get("link", ""),
    }


async def get_artist_top_tracks(artist_id: int, limit: int = 5) -> list[dict]:
    """Get top tracks for an artist by Deezer artist ID."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"{BASE_URL}artist/{artist_id}/top", params={"limit": limit})

    tracks = data.get("data", [])[:limit]
    return [
        {
            "title": t.get("title", ""),
            "duration": t.get("duration", 0),
            "preview_url": t.get("preview", ""),
            "album": {
                "title": t.get("album", {}).get("title", ""),
                "cover_url": t.get("album", {}).get("cover_medium", ""),
            },
        }
        for t in tracks
    ]


async def get_track(track_id: int) -> dict:
    """Get full track details by Deezer track ID."""
    async with httpx.AsyncClient() as client:
        data = await _get(client, f"{BASE_URL}track/{track_id}")

    return {
        "id": data.get("id"),
        "title": data.get("title", ""),
        "artist": data.get("artist", {}).get("name", ""),
        "album": data.get("album", {}).get("title", ""),
        "duration": data.get("duration", 0),
        "preview_url": data.get("preview", ""),
        "bpm": data.get("bpm"),
        "link": data.get("link", ""),
    }


async def download_preview(preview_url: str, save_path: str) -> str:
    """
    Download a 30-second MP3 preview to disk.

    Saves to data/previews/ by default. Returns the local file path.
    Skips download if the file already exists (cache).
    """
    dest = Path(save_path)
    if dest.exists():
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(preview_url)
        response.raise_for_status()
        dest.write_bytes(response.content)

    return str(dest)


if __name__ == "__main__":
    async def main():
        print("=== search_artist('Radiohead') ===")
        artist = await search_artist("Radiohead")
        print(f"  {artist['name']} — {artist['nb_fan']:,} fans — id: {artist['id']}")

        artist_id = artist["id"]

        print(f"\n=== get_artist({artist_id}) ===")
        details = await get_artist(artist_id)
        print(f"  Albums: {details['nb_album']}, Fans: {details['nb_fan']:,}")
        print(f"  Link: {details['link']}")

        print(f"\n=== get_artist_top_tracks({artist_id}) ===")
        top_tracks = await get_artist_top_tracks(artist_id, limit=5)
        for t in top_tracks:
            mins, secs = divmod(t["duration"], 60)
            print(f"  {t['title']} ({mins}:{secs:02d}) — {t['album']['title']}")
            print(f"    Preview: {t['preview_url']}")

        if top_tracks and top_tracks[0]["preview_url"]:
            first = top_tracks[0]
            filename = first["title"].replace(" ", "_").replace("/", "-") + ".mp3"
            save_path = str(PREVIEWS_DIR / filename)
            print(f"\n=== download_preview -> {save_path} ===")
            path = await download_preview(first["preview_url"], save_path)
            size_kb = Path(path).stat().st_size // 1024
            print(f"  Saved to: {path} ({size_kb} KB)")

    asyncio.run(main())
