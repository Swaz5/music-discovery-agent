"""
Unified music service.

Aggregates data from Last.fm, Deezer, and the audio analyzer into a single
consistent interface. Responsible for deduplication, cross-source enrichment,
and presenting a normalized track/artist data model to the rest of the system.

Caching
-------
A simple in-memory dict keyed by (function_name, args) stores (timestamp, result)
pairs. Entries expire after TTL_SECONDS (default 3600 = 1 hour). No external
dependency required.

Graceful degradation
--------------------
Each data source is fetched independently. If Deezer is unavailable, the
profile is returned with Last.fm data only and audio features omitted.
If Last.fm is unavailable, Deezer metadata is returned without bio/tags.
Failures are logged to stderr so they are visible without crashing callers.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from src.data import lastfm_client as lastfm
from src.data import deezer_client as deezer
from src.data.audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600
PREVIEWS_DIR = Path("data/previews")

# Cap concurrent preview downloads so Deezer's CDN doesn't throttle us.
# 30 simultaneous requests (6 artists × 5 tracks) reliably triggers rate limits.
_DOWNLOAD_SEMAPHORE = asyncio.Semaphore(6)

_cache: dict[str, tuple[float, Any]] = {}
_analyzer = AudioAnalyzer()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.monotonic() - ts > TTL_SECONDS:
        del _cache[key]
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.monotonic(), value)


def cache_clear() -> None:
    """Wipe the entire in-memory cache (useful in tests)."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _preview_path(title: str, artist: str = "") -> str:
    """
    Build a collision-safe local path for a preview file.
    Include artist name so two songs called e.g. 'Creep' from different
    artists don't overwrite each other when fetched in parallel.
    """
    def _safe(s: str) -> str:
        return s.replace(" ", "_").replace("/", "-").replace("\\", "-")
    artist_part = (_safe(artist)[:20] + "_") if artist else ""
    title_part = _safe(title)[:40]
    return str(PREVIEWS_DIR / f"{artist_part}{title_part}.mp3")


async def _download_and_analyze(tracks: list[dict]) -> list[dict]:
    """
    Download preview MP3s for a list of track dicts (must have 'preview_url'
    and 'title' keys), run LibROSA analysis on each, and return the tracks
    with an 'audio_features' key added.

    Tracks without a preview URL are returned as-is.
    """
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    async def _enrich(track: dict) -> dict:
        url = track.get("preview_url", "")
        if not url:
            return track
        path = _preview_path(
            track.get("title", "unknown"),
            track.get("artist", track.get("album", {}).get("title", "")),
        )
        try:
            async with _DOWNLOAD_SEMAPHORE:
                await deezer.download_preview(url, path)
            features = _analyzer.analyze_track(path)
            return {**track, "audio_features": features}
        except Exception as exc:
            logger.warning(
                "Audio analysis failed for %r: %s: %s",
                track.get("title"), type(exc).__name__, exc,
            )
            return track

    return list(await asyncio.gather(*[_enrich(t) for t in tracks]))


def _audio_profile(enriched_tracks: list[dict]) -> dict:
    """Average Librosa features across tracks that have 'audio_features'."""
    feats = [t["audio_features"] for t in enriched_tracks if "audio_features" in t]
    if not feats:
        return {}
    keys = ("energy", "danceability", "valence", "tempo", "acousticness")
    return {
        f"avg_{k}": round(sum(f[k] for f in feats) / len(feats), 3)
        for k in keys
        if all(k in f for f in feats)
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_full_artist_profile(name: str) -> dict:
    """
    Build a unified artist profile by combining Last.fm, Deezer, and Librosa.

    Last.fm and Deezer calls are issued in PARALLEL with asyncio.gather().
    Audio analysis runs after downloads complete (sequential dependency).

    Graceful degradation:
      - If Last.fm fails  → profile has no bio/tags/similar/lastfm_listeners.
      - If Deezer fails   → profile has no fan_count/album_art/audio features.
    """
    key = _cache_key("artist_profile", name.lower())
    cached = _cache_get(key)
    if cached is not None:
        logger.debug("Cache hit: %s", key)
        return cached

    # --- Parallel fetch: Last.fm + Deezer artist search ---
    lastfm_info_task = lastfm.get_artist_info(name)
    lastfm_tracks_task = lastfm.get_top_tracks(name, limit=5)
    deezer_artist_task = deezer.search_artist(name)

    (lastfm_info, lastfm_tracks, deezer_artist) = await asyncio.gather(
        lastfm_info_task,
        lastfm_tracks_task,
        deezer_artist_task,
        return_exceptions=True,
    )

    # --- Handle Last.fm results ---
    bio = ""
    tags: list[str] = []
    similar_artists: list[dict] = []
    lastfm_listeners = 0

    if isinstance(lastfm_info, Exception):
        logger.warning("Last.fm artist info failed for %r: %s", name, lastfm_info)
    else:
        bio = lastfm_info.get("bio_summary", "")
        tags = lastfm_info.get("tags", [])
        similar_artists = lastfm_info.get("similar_artists", [])
        lastfm_listeners = _safe_int(lastfm_info.get("listeners", 0))

    if isinstance(lastfm_tracks, Exception):
        logger.warning("Last.fm top tracks failed for %r: %s", name, lastfm_tracks)
        lastfm_tracks = []

    # --- Handle Deezer results ---
    fan_count = 0
    album_art_url = ""
    deezer_artist_id = None

    if isinstance(deezer_artist, Exception):
        logger.warning("Deezer artist search failed for %r: %s", name, deezer_artist)
        deezer_artist = {}

    if deezer_artist:
        fan_count = _safe_int(deezer_artist.get("nb_fan", 0))
        album_art_url = deezer_artist.get("picture_url", "")
        deezer_artist_id = deezer_artist.get("id")

    # --- Deezer top tracks (needs artist id from above) ---
    deezer_tracks: list[dict] = []
    if deezer_artist_id is not None:
        try:
            deezer_tracks = await deezer.get_artist_top_tracks(deezer_artist_id, limit=5)
        except Exception as exc:
            logger.warning("Deezer top tracks failed for %r: %s", name, exc)

    # --- Merge track lists: prefer Deezer (has preview URLs), fill gaps from Last.fm ---
    deezer_titles = {t["title"].lower() for t in deezer_tracks}
    merged_tracks = list(deezer_tracks)
    for t in lastfm_tracks:
        if t["name"].lower() not in deezer_titles:
            merged_tracks.append({
                "title": t["name"],
                "preview_url": "",
                "album": {"title": ""},
                "duration": 0,
                "lastfm_url": t.get("url", ""),
                "playcount": t.get("playcount", "0"),
            })

    # --- Download previews & run audio analysis ---
    enriched_tracks = await _download_and_analyze(merged_tracks)
    profile_audio = _audio_profile(enriched_tracks)

    result = {
        "name": name,
        "bio": bio,
        "tags": tags,
        "similar_artists": similar_artists,
        "top_tracks": enriched_tracks,
        "audio_profile": profile_audio,
        "fan_count": fan_count,
        "listeners": lastfm_listeners,
        "album_art_url": album_art_url,
    }

    _cache_set(key, result)
    return result


async def search_artists_by_vibe(
    tags: list[str],
    energy_range: tuple[float, float] = (0.0, 1.0),
    valence_range: tuple[float, float] = (0.0, 1.0),
) -> list[dict]:
    """
    Find artists matching a "vibe" defined by genre tags and audio feature ranges.

    1. Uses Last.fm to get candidate artists for each tag (in parallel).
    2. Deduplicates candidates and fetches/analyzes one Deezer preview each.
    3. Filters by energy_range and valence_range.

    Returns artists sorted by relevance (appearance across multiple tags).
    """
    key = _cache_key("vibe_search", *sorted(tags), *energy_range, *valence_range)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    # --- Fetch candidate artists for all tags in parallel ---
    tag_results = await asyncio.gather(
        *[lastfm.get_tag_top_artists(tag, limit=10) for tag in tags],
        return_exceptions=True,
    )

    # Tally how many tags each artist appears in (relevance score)
    relevance: dict[str, int] = {}
    for result in tag_results:
        if isinstance(result, Exception):
            logger.warning("Last.fm tag fetch failed: %s", result)
            continue
        for artist in result:
            relevance[artist["name"]] = relevance.get(artist["name"], 0) + 1

    if not relevance:
        return []

    # Sort by relevance and take top 20 to analyze
    candidates = sorted(relevance, key=lambda n: relevance[n], reverse=True)[:20]

    # --- Fetch one preview per candidate artist in parallel ---
    async def _fetch_preview(artist_name: str) -> dict | None:
        try:
            tracks = await deezer.search_tracks(artist_name, limit=1)
            if not tracks or not tracks[0].get("preview_url"):
                return None
            track = tracks[0]
            path = _preview_path(track["title"], track.get("artist", artist_name))
            async with _DOWNLOAD_SEMAPHORE:
                await deezer.download_preview(track["preview_url"], path)
            features = _analyzer.analyze_track(path)
            return {
                "name": artist_name,
                "relevance": relevance[artist_name],
                "preview_title": track["title"],
                "audio_features": features,
            }
        except Exception as exc:
            logger.warning("Vibe analysis failed for %r: %s", artist_name, exc)
            return None

    analyses = await asyncio.gather(*[_fetch_preview(n) for n in candidates])

    # --- Filter by audio feature ranges ---
    filtered = []
    for entry in analyses:
        if entry is None:
            continue
        f = entry["audio_features"]
        energy = f.get("energy", -1)
        valence = f.get("valence", -1)
        if energy_range[0] <= energy <= energy_range[1] and valence_range[0] <= valence <= valence_range[1]:
            filtered.append(entry)

    filtered.sort(key=lambda e: e["relevance"], reverse=True)
    _cache_set(key, filtered)
    return filtered


async def get_similar_with_features(artist: str, limit: int = 10) -> list[dict]:
    """
    Get artists similar to `artist` from Last.fm, then enrich each with
    Librosa-computed audio features via a Deezer preview.

    Returns list sorted by Last.fm similarity score (descending).
    """
    key = _cache_key("similar_features", artist.lower(), limit)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        similar = await lastfm.get_similar_artists(artist, limit=limit)
    except Exception as exc:
        logger.warning("Last.fm similar artists failed for %r: %s", artist, exc)
        return []

    async def _enrich_similar(entry: dict) -> dict:
        name = entry["name"]
        try:
            tracks = await deezer.search_tracks(name, limit=1)
            if tracks and tracks[0].get("preview_url"):
                track = tracks[0]
                path = _preview_path(track["title"], track.get("artist", name))
                async with _DOWNLOAD_SEMAPHORE:
                    await deezer.download_preview(track["preview_url"], path)
                features = _analyzer.analyze_track(path)
                return {**entry, "audio_features": features}
        except Exception as exc:
            logger.warning("Feature enrichment failed for similar artist %r: %s", name, exc)
        return entry

    enriched = await asyncio.gather(*[_enrich_similar(e) for e in similar])
    result = sorted(enriched, key=lambda e: e.get("match", 0), reverse=True)
    _cache_set(key, result)
    return result


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    ARTISTS = ["Radiohead", "Daft Punk", "Nick Drake"]

    async def main():
        for artist_name in ARTISTS:
            print(f"\n{'=' * 60}")
            print(f"  {artist_name}")
            print(f"{'=' * 60}")

            profile = await get_full_artist_profile(artist_name)

            print(f"  Listeners (Last.fm): {profile['listeners']:,}")
            print(f"  Fans (Deezer):       {profile['fan_count']:,}")
            print(f"  Tags:                {profile['tags'][:5]}")
            print(f"  Similar:             {[a['name'] for a in profile['similar_artists'][:3]]}")
            print(f"  Bio:                 {profile['bio'][:120].strip()}...")
            print(f"  Album art:           {profile['album_art_url']}")

            print(f"\n  Top tracks:")
            for t in profile["top_tracks"][:5]:
                af = t.get("audio_features", {})
                energy = af.get("energy", "—")
                valence = af.get("valence", "—")
                tempo = af.get("tempo", "—")
                title = t.get("title", t.get("name", "?"))
                print(f"    {title:<35} energy={energy}  valence={valence}  tempo={tempo}")

            ap = profile["audio_profile"]
            if ap:
                print(f"\n  Audio profile (avg across tracks):")
                for k, v in ap.items():
                    print(f"    {k:<22} {v}")
            else:
                print("\n  [no audio profile — previews unavailable]")

        # Cache hit demo
        print(f"\n{'=' * 60}")
        print("  Cache hit demo — fetching Radiohead again...")
        print(f"{'=' * 60}")
        t0 = time.monotonic()
        await get_full_artist_profile("Radiohead")
        elapsed = time.monotonic() - t0
        print(f"  Returned in {elapsed*1000:.1f} ms (from cache)")

    _asyncio.run(main())
