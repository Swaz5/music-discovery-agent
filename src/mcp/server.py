"""
MCP server for the music discovery agent.

Exposes music discovery capabilities as MCP tools and resources so that
any MCP-compatible client (Claude Desktop, Cursor, custom agents) can call
them without extra integration code.

Transport: stdio (the client spawns this process and pipes JSON-RPC over
stdin/stdout). See __main__ block and Claude Desktop config below.

─────────────────────────────────────────────────────────────────────────
Claude Desktop configuration
─────────────────────────────────────────────────────────────────────────

  Windows config file location:
    %APPDATA%\\Claude\\claude_desktop_config.json
    (typically C:\\Users\\<you>\\AppData\\Roaming\\Claude\\claude_desktop_config.json)

  macOS config file location:
    ~/Library/Application Support/Claude/claude_desktop_config.json

  Add (or merge) this JSON block, replacing the cwd path:

    {
      "mcpServers": {
        "music-discovery": {
          "command": "python",
          "args": ["-m", "src.mcp.server"],
          "cwd": "C:\\\\Users\\\\jakea\\\\Documents\\\\Coding Projects\\\\music-discovery-agent"
        }
      }
    }

  Then restart Claude Desktop — the tools will appear automatically.
─────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.data import music_service
from src.data import lastfm_client
from src.agent.discovery_agent import MusicDiscoveryAgent
from src.rag.retriever import MusicRetriever

logger = logging.getLogger(__name__)

# ── Singletons (created once, reused across tool calls) ──────────────────────

_agent = MusicDiscoveryAgent()
_retriever = MusicRetriever()

PREFERENCES_PATH = Path("data/preferences.json")

# ── FastMCP server instance ───────────────────────────────────────────────────

mcp = FastMCP(
    "music-discovery",
    instructions=(
        "A music discovery server. Use discover_music for open-ended vibe-based "
        "recommendations, get_artist_deep_dive for artist profiles, "
        "compare_artists to contrast two artists, explore_genre for genre deep "
        "dives, and save_preference to remember what the user likes or dislikes."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_preferences() -> dict:
    """Load preferences from disk, returning an empty structure if missing."""
    if PREFERENCES_PATH.exists():
        try:
            return json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"liked": [], "disliked": [], "notes": {}}


def _save_preferences(prefs: dict) -> None:
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(
        json.dumps(prefs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fmt_audio(features: dict) -> str:
    """Render audio feature dict as a compact readable line."""
    if not features:
        return "  (audio analysis unavailable)"
    keys = ("energy", "danceability", "valence", "tempo", "acousticness")
    parts = []
    for k in keys:
        if k in features:
            v = features[k]
            parts.append(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    return "  " + "  |  ".join(parts)


def _fmt_artist_profile(profile: dict) -> str:
    """Render a full artist profile as a readable multi-section string."""
    lines: list[str] = []
    name = profile.get("name", "Unknown")

    lines.append(f"# {name}")
    lines.append("")

    # Stats
    listeners = profile.get("listeners", 0)
    fans = profile.get("fan_count", 0)
    if listeners or fans:
        lines.append(
            f"Last.fm listeners: {int(listeners):,}   "
            f"Deezer fans: {int(fans):,}"
        )

    # Tags
    tags = profile.get("tags", [])
    if tags:
        lines.append(f"Tags: {', '.join(tags[:8])}")

    # Bio
    bio = profile.get("bio", "").strip()
    if bio:
        lines.append("")
        lines.append("## Bio")
        # Strip the Last.fm "Read more on Last.fm" suffix if present
        bio = bio.split("<a href")[0].strip()
        lines.append(bio[:800] + ("…" if len(bio) > 800 else ""))

    # Audio profile
    audio = profile.get("audio_profile", {})
    if audio:
        lines.append("")
        lines.append("## Average audio profile (across top tracks)")
        lines.append(_fmt_audio(audio))

    # Top tracks
    tracks = profile.get("top_tracks", [])
    if tracks:
        lines.append("")
        lines.append("## Top tracks")
        for t in tracks[:6]:
            title = t.get("title") or t.get("name", "?")
            af = t.get("audio_features", {})
            af_str = _fmt_audio(af) if af else ""
            preview = t.get("preview_url", "")
            link = t.get("lastfm_url", t.get("link", ""))
            row = f"  • {title}"
            if af_str:
                row += f"\n  {af_str.strip()}"
            if preview:
                row += f"\n    Preview: {preview}"
            elif link:
                row += f"\n    Link: {link}"
            lines.append(row)

    # Similar artists
    similar = profile.get("similar_artists", [])
    if similar:
        lines.append("")
        lines.append("## Similar artists")
        names = [a.get("name", "") for a in similar[:8] if a.get("name")]
        lines.append("  " + ", ".join(names))

    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Discover music based on a mood, vibe, or reference description. "
        "Runs the full discovery agent — searches the knowledge base, finds "
        "artists by genre tag, verifies audio features, and returns "
        "personalised recommendations with reasoning for each pick."
    )
)
async def discover_music(vibe_description: str) -> str:
    """
    Parameters
    ----------
    vibe_description : str
        Natural-language description of what the user wants to hear.
        Examples: "melancholic indie for a rainy day",
                  "something like Portishead but more upbeat",
                  "late-night jazz with a dark electronic edge"
    """
    result = await _agent.discover(vibe_description)
    recs = result.get("recommendations", [])

    if not recs:
        return (
            "The agent completed its search but did not produce structured "
            "recommendations. Try rephrasing your request with more specific "
            "mood or genre details."
        )

    lines: list[str] = [
        f'# Music recommendations for: "{vibe_description}"',
        f"_(found via {result['iterations']} research steps, "
        f"{result['total_tokens']:,} tokens)_",
        "",
    ]

    for i, rec in enumerate(recs, 1):
        artist = rec.get("artist", "?")
        track = rec.get("track", "?")
        score = rec.get("vibe_match_score", 0.0)
        tags = ", ".join(rec.get("genre_tags", [])) or "—"
        why = rec.get("why", "")
        vibe_desc = rec.get("vibe_description", "")
        url = rec.get("deezer_url", "")

        lines.append(f"## {i}. {artist} — {track}")
        lines.append(f"**Tags:** {tags}   **Vibe match:** {score:.2f}")
        if why:
            lines.append(f"**Why:** {why}")
        if vibe_desc:
            lines.append(f"_{vibe_desc}_")
        if url:
            lines.append(f"[Listen on Deezer]({url})")
        lines.append("")

    # Append a brief reasoning trace so the user can see how it was found
    trace = result.get("reasoning_trace", [])
    if trace:
        lines.append("---")
        lines.append("**Research steps taken:**")
        for step in trace:
            args_preview = json.dumps(step.get("arguments", {}), ensure_ascii=False)[:70]
            status = "✗ error" if "error" in step else "✓"
            lines.append(f"  {status} `{step['tool']}({args_preview})`")

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Get a comprehensive profile of an artist including their sound "
        "characteristics, similar artists, top tracks with audio analysis, "
        "and genre context from the knowledge base."
    )
)
async def get_artist_deep_dive(artist_name: str) -> str:
    """
    Parameters
    ----------
    artist_name : str
        The artist's name, e.g. "Massive Attack", "Björk", "Floating Points"
    """
    # Fetch live profile and KB context in parallel
    profile_task = asyncio.create_task(
        music_service.get_full_artist_profile(artist_name)
    )
    kb_context = _retriever.retrieve_for_artist(artist_name, n=3)
    profile = await profile_task

    lines: list[str] = [_fmt_artist_profile(profile)]

    if kb_context and "[No relevant" not in kb_context:
        lines.append("")
        lines.append("## Knowledge base context")
        lines.append(kb_context)

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Compare two artists across audio features, genres, tags, and "
        "popularity. Shows what they have in common and how they differ "
        "so you can understand the sonic distance between them."
    )
)
async def compare_artists(artist1: str, artist2: str) -> str:
    """
    Parameters
    ----------
    artist1 : str
        First artist name
    artist2 : str
        Second artist name
    """
    # Fetch both profiles concurrently
    p1, p2 = await asyncio.gather(
        music_service.get_full_artist_profile(artist1),
        music_service.get_full_artist_profile(artist2),
    )

    n1 = p1.get("name", artist1)
    n2 = p2.get("name", artist2)

    lines: list[str] = [f"# {n1}  vs  {n2}", ""]

    # ── Stats comparison ──────────────────────────────────────────────
    lines.append("## Popularity")
    lines.append(
        f"| Metric | {n1} | {n2} |"
        "\n|---|---|---|"
        f"\n| Last.fm listeners | {int(p1.get('listeners', 0)):,} "
        f"| {int(p2.get('listeners', 0)):,} |"
        f"\n| Deezer fans | {int(p1.get('fan_count', 0)):,} "
        f"| {int(p2.get('fan_count', 0)):,} |"
    )

    # ── Tags comparison ───────────────────────────────────────────────
    tags1 = set(p1.get("tags", []))
    tags2 = set(p2.get("tags", []))
    shared = sorted(tags1 & tags2)
    only1 = sorted(tags1 - tags2)
    only2 = sorted(tags2 - tags1)

    lines.append("")
    lines.append("## Genre / tags")
    if shared:
        lines.append(f"**In common:** {', '.join(shared)}")
    if only1:
        lines.append(f"**Only {n1}:** {', '.join(only1[:6])}")
    if only2:
        lines.append(f"**Only {n2}:** {', '.join(only2[:6])}")

    # ── Audio feature comparison ──────────────────────────────────────
    a1 = p1.get("audio_profile", {})
    a2 = p2.get("audio_profile", {})
    audio_keys = ("energy", "danceability", "valence", "tempo", "acousticness")
    common_keys = [k for k in audio_keys if k in a1 and k in a2]

    if common_keys:
        lines.append("")
        lines.append("## Audio features (averaged across top tracks)")
        header = "| Feature | " + n1 + " | " + n2 + " | Δ |"
        sep = "|---|---|---|---|"
        rows = [header, sep]
        for k in common_keys:
            v1 = a1[k]
            v2 = a2[k]
            delta = v2 - v1
            arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "≈")
            rows.append(
                f"| {k} | {v1:.3f} | {v2:.3f} | {arrow} {abs(delta):.3f} |"
            )
        lines.append("\n".join(rows))
    elif a1 or a2:
        lines.append("")
        lines.append("_Audio profiles could not be compared "
                     "(one or both artists lack preview data)._")

    # ── Similar artists overlap ───────────────────────────────────────
    sim1 = {a["name"] for a in p1.get("similar_artists", [])}
    sim2 = {a["name"] for a in p2.get("similar_artists", [])}
    mutual = sorted(sim1 & sim2)
    if mutual:
        lines.append("")
        lines.append("## Artists that appear in both 'similar' lists")
        lines.append("  " + ", ".join(mutual))

    # ── Summary judgement ─────────────────────────────────────────────
    lines.append("")
    lines.append("## Summary")
    if shared:
        lines.append(
            f"{n1} and {n2} share the '{shared[0]}' tag space, suggesting "
            "a common sonic ancestry."
        )
    if common_keys:
        energy_diff = abs(a1.get("energy", 0) - a2.get("energy", 0))
        valence_diff = abs(a1.get("valence", 0) - a2.get("valence", 0))
        if energy_diff > 0.2:
            more_energetic = n1 if a1.get("energy", 0) > a2.get("energy", 0) else n2
            lines.append(f"{more_energetic} is notably more energetic (Δ {energy_diff:.2f}).")
        if valence_diff > 0.2:
            more_positive = n1 if a1.get("valence", 0) > a2.get("valence", 0) else n2
            lines.append(
                f"{more_positive} skews more emotionally positive (Δ {valence_diff:.2f})."
            )

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Explore a music genre — its characteristics, key artists, subgenres, "
        "and what it sounds like. Combines the curated knowledge base with "
        "live Last.fm tag data for a complete picture."
    )
)
async def explore_genre(genre_name: str) -> str:
    """
    Parameters
    ----------
    genre_name : str
        Genre or tag name, e.g. "shoegaze", "trip-hop", "afrobeat", "ambient"
    """
    # Fetch KB context and Last.fm top artists in parallel
    kb_context = _retriever.retrieve_for_genre(genre_name, n=4)
    top_artists = await lastfm_client.get_tag_top_artists(genre_name, limit=12)

    lines: list[str] = [f"# Genre: {genre_name}", ""]

    if kb_context and "[No relevant" not in kb_context:
        lines.append("## From the knowledge base")
        lines.append(kb_context)
        lines.append("")

    if top_artists:
        lines.append("## Top artists on Last.fm")
        for a in top_artists:
            rank = a.get("rank", "")
            name = a.get("name", "?")
            url = a.get("url", "")
            row = f"  {rank}. {name}" if rank else f"  • {name}"
            if url:
                row += f"  ({url})"
            lines.append(row)
    else:
        lines.append("_No Last.fm data found for this tag._")

    return "\n".join(lines)


@mcp.tool(
    description=(
        "Save a music preference — whether you liked or disliked an artist, "
        "with optional notes about why. Preferences are stored locally and "
        "can be read back via the music://preferences resource."
    )
)
def save_preference(artist: str, liked: bool, notes: str = "") -> str:
    """
    Parameters
    ----------
    artist : str
        Artist name
    liked : bool
        True if the user liked the artist, False if they disliked them
    notes : str, optional
        Free-text note, e.g. "too slow" or "loved the drum production"
    """
    prefs = _load_preferences()

    # Move artist between liked/disliked lists if they changed their mind
    liked_list: list[str] = prefs.setdefault("liked", [])
    disliked_list: list[str] = prefs.setdefault("disliked", [])

    if liked:
        if artist not in liked_list:
            liked_list.append(artist)
        if artist in disliked_list:
            disliked_list.remove(artist)
    else:
        if artist not in disliked_list:
            disliked_list.append(artist)
        if artist in liked_list:
            liked_list.remove(artist)

    if notes:
        prefs.setdefault("notes", {})[artist] = notes

    _save_preferences(prefs)

    sentiment = "liked ✓" if liked else "disliked ✗"
    msg = f"Saved: {artist} → {sentiment}"
    if notes:
        msg += f' (note: "{notes}")'
    return msg


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource(
    "music://preferences",
    name="User music preferences",
    description=(
        "Your saved music preferences — artists you liked and disliked, "
        "with any notes. Updated by the save_preference tool."
    ),
    mime_type="text/plain",
)
def resource_preferences() -> str:
    """Return the user's taste profile as readable text."""
    prefs = _load_preferences()

    liked = prefs.get("liked", [])
    disliked = prefs.get("disliked", [])
    notes_map = prefs.get("notes", {})

    lines: list[str] = ["# Your music preferences", ""]

    if liked:
        lines.append("## Artists you like")
        for a in liked:
            note = notes_map.get(a, "")
            lines.append(f"  ✓ {a}" + (f"  — {note}" if note else ""))
    else:
        lines.append("## Artists you like\n  (none saved yet)")

    lines.append("")

    if disliked:
        lines.append("## Artists you dislike")
        for a in disliked:
            note = notes_map.get(a, "")
            lines.append(f"  ✗ {a}" + (f"  — {note}" if note else ""))
    else:
        lines.append("## Artists you dislike\n  (none saved yet)")

    return "\n".join(lines)


@mcp.resource(
    "music://genres",
    name="Available genres in knowledge base",
    description=(
        "List of genre tags and topics covered by the curated knowledge base. "
        "Pass any of these to explore_genre for a deep dive."
    ),
    mime_type="text/plain",
)
def resource_genres() -> str:
    """Return the genres present in the knowledge base."""
    # Query the KB with broad terms to surface available genre topics
    results = _retriever.retrieve_for_vibe("genre", n=20)
    lines: list[str] = [
        "# Genres in the knowledge base",
        "",
        "Use any of these with the explore_genre tool:",
        "",
        results,
    ]
    return "\n".join(lines)


# ── Server entry point ────────────────────────────────────────────────────────

def run_server() -> None:
    """Start the MCP server on stdio transport."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
