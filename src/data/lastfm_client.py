"""
Last.fm API client.

Handles authenticated requests to the Last.fm REST API, providing methods
for fetching similar artists, top tracks, tag-based search, and artist metadata.
"""

import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://ws.audioscrobbler.com/2.0/"
API_KEY = os.getenv("LASTFM_API_KEY")

# Hard timeout for every Last.fm HTTP call.  Without this a single slow/hung
# response blocks the entire FastAPI asyncio event loop, making every other
# in-flight request also time out.
_TIMEOUT = httpx.Timeout(10.0)


def _base_params(method: str) -> dict:
    return {
        "method": method,
        "api_key": API_KEY,
        "format": "json",
    }


async def search_artist(name: str) -> dict:
    """Search for artists by name. Returns top 5 matches."""
    params = {
        **_base_params("artist.search"),
        "artist": name,
        "limit": 5,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    matches = data.get("results", {}).get("artistmatches", {}).get("artist", [])
    return [
        {
            "name": a.get("name", ""),
            "listeners": a.get("listeners", "0"),
            "url": a.get("url", ""),
        }
        for a in matches
    ]


async def get_artist_info(artist: str) -> dict:
    """Get detailed info for an artist including bio, tags, similar artists, and stats."""
    params = {
        **_base_params("artist.getInfo"),
        "artist": artist,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")

    info = data.get("artist", {})
    bio = info.get("bio", {})
    stats = info.get("stats", {})
    tags = [t.get("name", "") for t in info.get("tags", {}).get("tag", [])]
    similar = [
        {"name": a.get("name", ""), "url": a.get("url", "")}
        for a in info.get("similar", {}).get("artist", [])
    ]

    return {
        "name": info.get("name", ""),
        "bio_summary": bio.get("summary", ""),
        "tags": tags,
        "similar_artists": similar,
        "listeners": stats.get("listeners", "0"),
        "playcount": stats.get("playcount", "0"),
    }


async def get_similar_artists(artist: str, limit: int = 10) -> list[dict]:
    """Get artists similar to a given artist."""
    params = {
        **_base_params("artist.getSimilar"),
        "artist": artist,
        "limit": limit,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")

    artists = data.get("similarartists", {}).get("artist", [])
    return [
        {
            "name": a.get("name", ""),
            "match": float(a.get("match", 0)),
            "url": a.get("url", ""),
        }
        for a in artists
    ]


async def get_artist_tags(artist: str) -> list[str]:
    """Get top tags for an artist, sorted by count."""
    params = {
        **_base_params("artist.getTopTags"),
        "artist": artist,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")

    tags = data.get("toptags", {}).get("tag", [])
    # Tags already come sorted by count descending from the API
    return [t.get("name", "") for t in tags]


async def get_top_tracks(artist: str, limit: int = 5) -> list[dict]:
    """Get top tracks for an artist."""
    params = {
        **_base_params("artist.getTopTracks"),
        "artist": artist,
        "limit": limit,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")

    tracks = data.get("toptracks", {}).get("track", [])
    return [
        {
            "name": t.get("name", ""),
            "playcount": t.get("playcount", "0"),
            "listeners": t.get("listeners", "0"),
            "url": t.get("url", ""),
        }
        for t in tracks
    ]


async def get_tag_top_artists(tag: str, limit: int = 10) -> list[dict]:
    """Get top artists for a given tag/genre."""
    params = {
        **_base_params("tag.getTopArtists"),
        "tag": tag,
        "limit": limit,
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")

    artists = data.get("topartists", {}).get("artist", [])
    return [
        {
            "name": a.get("name", ""),
            "rank": a.get("@attr", {}).get("rank", ""),
            "url": a.get("url", ""),
        }
        for a in artists
    ]


if __name__ == "__main__":
    async def main():
        artist = "Radiohead"
        tag = "shoegaze"

        print(f"=== search_artist('{artist}') ===")
        results = await search_artist(artist)
        for r in results:
            print(f"  {r['name']} — {r['listeners']} listeners — {r['url']}")

        print(f"\n=== get_artist_info('{artist}') ===")
        info = await get_artist_info(artist)
        print(f"  Listeners: {info['listeners']}, Playcount: {info['playcount']}")
        print(f"  Tags: {info['tags'][:5]}")
        print(f"  Similar: {[a['name'] for a in info['similar_artists'][:3]]}")
        print(f"  Bio: {info['bio_summary'][:200]}...")

        print(f"\n=== get_similar_artists('{artist}') ===")
        similar = await get_similar_artists(artist, limit=5)
        for a in similar:
            print(f"  {a['name']} (match: {a['match']:.3f})")

        print(f"\n=== get_artist_tags('{artist}') ===")
        tags = await get_artist_tags(artist)
        print(f"  {tags[:10]}")

        print(f"\n=== get_top_tracks('{artist}') ===")
        tracks = await get_top_tracks(artist)
        for t in tracks:
            print(f"  {t['name']} — {t['playcount']} plays")

        print(f"\n=== get_tag_top_artists('{tag}') ===")
        artists = await get_tag_top_artists(tag, limit=5)
        for a in artists:
            print(f"  #{a['rank']} {a['name']}")

    asyncio.run(main())
