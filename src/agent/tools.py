"""
Agent tool definitions.

Defines the tool schemas and handler functions exposed to the Claude agent,
including tools for searching tracks, fetching similar artists, retrieving
audio features, and querying the RAG knowledge base.

Anthropic tool_use format
--------------------------
Each tool is represented to the API as a dict with this shape::

    {
        "name": "tool_name",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": {
                "param_name": {"type": "string", "description": "..."},
                ...
            },
            "required": ["param_name", ...]
        }
    }

When the model wants to call a tool, it returns a response with
``stop_reason == "tool_use"`` and a content block of type ``"tool_use"``
containing ``id``, ``name``, and ``input`` (the parsed arguments dict).

The caller must then invoke the tool, wrap the result in a ``"tool_result"``
user message block keyed by the original ``id``, and send it back to continue
the conversation.

TOOLS_REGISTRY maps each tool name to::

    {
        "schema": <tool dict for the API>,
        "function": <async callable(arguments: dict) -> Any>
    }

Use ``get_tool_schemas()`` to extract all schemas for the API ``tools``
parameter, and ``execute_tool(name, arguments)`` to dispatch a call.
"""

import logging
from pathlib import Path
from typing import Any

from src.data import deezer_client
from src.data import lastfm_client
from src.data import music_service
from src.data.audio_analyzer import AudioAnalyzer
from src.rag.retriever import MusicRetriever

logger = logging.getLogger(__name__)

_analyzer = AudioAnalyzer()
_retriever = MusicRetriever()

_PREVIEWS_DIR = Path("data/previews")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _search_artist(name: str) -> dict:
    """
    Search for an artist by name and return their full profile.

    Combines Last.fm bio/tags/similar-artists data with Deezer fan counts,
    album art, top tracks, and Librosa-derived audio features.
    """
    return await music_service.get_full_artist_profile(name)


async def _get_similar_artists(artist: str, limit: int = 10) -> list[dict]:
    """
    Return artists similar to the given artist, each enriched with
    Librosa audio features computed from a Deezer 30-second preview.
    """
    return await music_service.get_similar_with_features(artist, limit=limit)


async def _search_by_tag(tag: str, limit: int = 10) -> list[dict]:
    """
    Return the top Last.fm artists for a genre tag such as 'shoegaze'
    or 'trip-hop', ordered by Last.fm rank.
    """
    return await lastfm_client.get_tag_top_artists(tag, limit=limit)


async def _get_audio_features(track_name: str, artist_name: str) -> dict:
    """
    Fetch audio features for a specific track.

    1. Searches Deezer for ``"<track_name> <artist_name>"``.
    2. Downloads the 30-second MP3 preview.
    3. Runs Librosa analysis to extract energy, danceability, valence,
       tempo, and acousticness.

    Returns the raw features dict, or an error key if no preview is found.
    """
    query = f"{track_name} {artist_name}"
    tracks = await deezer_client.search_tracks(query, limit=1)

    if not tracks:
        return {"error": f"No Deezer results found for {query!r}"}

    track = tracks[0]
    preview_url = track.get("preview_url", "")
    if not preview_url:
        return {"error": f"No preview URL available for {track.get('title', query)!r}"}

    safe_artist = artist_name.replace(" ", "_").replace("/", "-")[:20]
    safe_title = track_name.replace(" ", "_").replace("/", "-")[:40]
    save_path = str(_PREVIEWS_DIR / f"{safe_artist}_{safe_title}.mp3")

    _PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    await deezer_client.download_preview(preview_url, save_path)

    features = _analyzer.analyze_track(save_path)
    return {
        "track": track.get("title", track_name),
        "artist": track.get("artist", artist_name),
        "album": track.get("album", ""),
        "audio_features": features,
    }


async def _search_knowledge_base(query: str, n_results: int = 5) -> str:
    """
    Query the RAG knowledge base for genre guides, artist bios, and
    music history relevant to the query.

    Returns a formatted context block (string) ready for inclusion in a
    prompt, produced by MusicRetriever.retrieve_mixed().
    """
    return _retriever.retrieve_mixed(query, n=n_results)


async def _search_tracks(query: str, limit: int = 5) -> list[dict]:
    """
    Search Deezer for tracks matching the query.

    Each result includes title, artist, album, preview URL, and a
    direct Deezer link.
    """
    return await deezer_client.search_tracks(query, limit=limit)


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

_SCHEMA_SEARCH_ARTIST = {
    "name": "search_artist",
    "description": (
        "Search for an artist by name and get their profile including tags, "
        "genres, similar artists, and audio characteristics"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Artist name to search for",
            }
        },
        "required": ["name"],
    },
}

_SCHEMA_GET_SIMILAR_ARTISTS = {
    "name": "get_similar_artists",
    "description": (
        "Find artists similar to a given artist, enriched with audio "
        "features for comparison"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "artist": {
                "type": "string",
                "description": "Name of the seed artist",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of similar artists to return (default 10)",
                "default": 10,
            },
        },
        "required": ["artist"],
    },
}

_SCHEMA_SEARCH_BY_TAG = {
    "name": "search_by_tag",
    "description": (
        "Find top artists for a specific genre tag like 'shoegaze', "
        "'trip-hop', or 'dream-pop'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tag": {
                "type": "string",
                "description": "Genre or mood tag to search, e.g. 'shoegaze', 'trip-hop'",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of artists to return (default 10)",
                "default": 10,
            },
        },
        "required": ["tag"],
    },
}

_SCHEMA_GET_AUDIO_FEATURES = {
    "name": "get_audio_features",
    "description": (
        "Get detailed audio features (energy, danceability, valence, tempo) "
        "for a specific track, computed via Librosa audio analysis"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "track_name": {
                "type": "string",
                "description": "Name of the track",
            },
            "artist_name": {
                "type": "string",
                "description": "Name of the artist who recorded the track",
            },
        },
        "required": ["track_name", "artist_name"],
    },
}

_SCHEMA_SEARCH_KNOWLEDGE_BASE = {
    "name": "search_knowledge_base",
    "description": (
        "Search the music knowledge base for genre guides, artist bios, "
        "and music history relevant to a query"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query",
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

_SCHEMA_SEARCH_TRACKS = {
    "name": "search_tracks",
    "description": (
        "Search for specific tracks on Deezer, returns track name, artist, "
        "album, preview URL, and Deezer link"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'Creep Radiohead' or 'Blue in Green Miles Davis'",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of tracks to return (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS_REGISTRY: dict[str, dict[str, Any]] = {
    "search_artist": {
        "schema": _SCHEMA_SEARCH_ARTIST,
        "function": lambda args: _search_artist(**args),
    },
    "get_similar_artists": {
        "schema": _SCHEMA_GET_SIMILAR_ARTISTS,
        "function": lambda args: _get_similar_artists(**args),
    },
    "search_by_tag": {
        "schema": _SCHEMA_SEARCH_BY_TAG,
        "function": lambda args: _search_by_tag(**args),
    },
    "get_audio_features": {
        "schema": _SCHEMA_GET_AUDIO_FEATURES,
        "function": lambda args: _get_audio_features(**args),
    },
    "search_knowledge_base": {
        "schema": _SCHEMA_SEARCH_KNOWLEDGE_BASE,
        "function": lambda args: _search_knowledge_base(**args),
    },
    "search_tracks": {
        "schema": _SCHEMA_SEARCH_TRACKS,
        "function": lambda args: _search_tracks(**args),
    },
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tool_schemas() -> list[dict]:
    """
    Return all tool schemas in the format expected by the Anthropic API's
    ``tools`` parameter.

    Each element is a dict with ``name``, ``description``, and
    ``input_schema`` keys, ready to be passed directly to
    ``anthropic.Anthropic().messages.create(tools=get_tool_schemas(), ...)``.
    """
    return [entry["schema"] for entry in TOOLS_REGISTRY.values()]


async def execute_tool(name: str, arguments: dict) -> Any:
    """
    Dispatch a tool call by name and return its result.

    Parameters
    ----------
    name : str
        Tool name as it appears in the API response ``tool_use`` block.
    arguments : dict
        Parsed ``input`` dict from the ``tool_use`` content block.

    Returns
    -------
    Any
        The tool's return value (dict, list, or str depending on the tool).

    Raises
    ------
    KeyError
        If ``name`` is not registered in TOOLS_REGISTRY.
    """
    if name not in TOOLS_REGISTRY:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list(TOOLS_REGISTRY)}")

    logger.debug("Executing tool %r with arguments %r", name, arguments)
    result = await TOOLS_REGISTRY[name]["function"](arguments)
    logger.debug("Tool %r returned %d-item result", name, len(result) if hasattr(result, "__len__") else 1)
    return result
