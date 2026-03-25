"""
High-level retrieval interface for the music discovery agent.

MusicRetriever wraps MusicKnowledgeBase with smarter retrieval strategies
and formats results as ready-to-paste context blocks for LLM prompts.

Retrieval strategies
--------------------
retrieve_for_vibe      Pure semantic search on the vibe description.
retrieve_for_artist    Metadata-filtered exact match, semantic fallback.
retrieve_for_genre     Two-pass: tag filter + semantic search, merged.
retrieve_mixed         Multi-type search with per-artist deduplication.

All methods return a formatted string — not raw dicts — so the agent can
pass results directly into a prompt without further processing.
"""

import logging
import re
from difflib import SequenceMatcher

from src.rag.knowledge_base import MusicKnowledgeBase

logger = logging.getLogger(__name__)

# Maximum L2 distance before a result is considered too far to include.
# Empirically calibrated on the current corpus: most good matches sit
# below 1.6; beyond 1.8 results are usually tangential.
_MAX_DISTANCE = 1.8

# Similarity ratio above which two text chunks are treated as near-duplicates
_DEDUP_THRESHOLD = 0.85


def _format_result(r: dict) -> str:
    """Render a single retrieval result as a formatted prompt block."""
    meta = r["metadata"]
    source = meta.get("source", "unknown")
    doc_type = meta.get("type", "unknown")
    artist = meta.get("artist", "")
    tags = meta.get("tags", "")

    header_parts = [f"Source: {source}", f"Type: {doc_type}"]
    if artist:
        header_parts.append(f"Artist: {artist}")
    if tags:
        header_parts.append(f"Tags: {tags}")

    header = " | ".join(header_parts)
    return f"{header}\n{r['text']}\n---"


def _format_block(results: list[dict], query: str = "") -> str:
    """Render a list of results as a labelled context block."""
    if not results:
        return "[No relevant knowledge base entries found.]"
    label = f"[Relevant knowledge for: {query!r}]\n\n" if query else ""
    return label + "\n\n".join(_format_result(r) for r in results)


def _filter_by_distance(results: list[dict], max_dist: float = _MAX_DISTANCE) -> list[dict]:
    """Drop results whose L2 distance exceeds the threshold."""
    return [r for r in results if r["distance_score"] <= max_dist]


def _deduplicate(results: list[dict], threshold: float = _DEDUP_THRESHOLD) -> list[dict]:
    """
    Remove near-duplicate text entries.

    Uses SequenceMatcher ratio to compare each candidate against all
    already-accepted results. Runs in O(n²) but n is always small (≤20).
    """
    accepted: list[dict] = []
    for candidate in results:
        text_c = candidate["text"]
        is_dup = any(
            SequenceMatcher(None, text_c, a["text"]).ratio() >= threshold
            for a in accepted
        )
        if not is_dup:
            accepted.append(candidate)
    return accepted


def _cap_per_artist(results: list[dict], max_per_artist: int = 2) -> list[dict]:
    """
    Limit how many chunks come from the same artist.

    Ensures diversity when a single artist has many bio chunks in the KB.
    Non-artist documents (artist == "") are not capped.
    """
    counts: dict[str, int] = {}
    out: list[dict] = []
    for r in results:
        artist = r["metadata"].get("artist", "")
        if not artist:
            out.append(r)
            continue
        if counts.get(artist, 0) < max_per_artist:
            counts[artist] = counts.get(artist, 0) + 1
            out.append(r)
    return out


class MusicRetriever:
    """
    Retrieval interface for the music discovery agent.

    All public methods return a formatted string ready for inclusion in
    an LLM prompt. The KB is queried but results are filtered, deduplicated,
    and rendered before returning.
    """

    def __init__(self, kb: MusicKnowledgeBase | None = None) -> None:
        self._kb = kb or MusicKnowledgeBase()

    # ------------------------------------------------------------------
    # Public retrieval methods
    # ------------------------------------------------------------------

    def retrieve_for_vibe(self, vibe_description: str, n: int = 5) -> str:
        """
        Retrieve knowledge relevant to a natural-language vibe description.

        Performs pure semantic search — no metadata filtering. This is the
        most general entry point and the one the agent should reach for first
        when a user describes a feeling or mood rather than a genre or artist.

        Parameters
        ----------
        vibe_description : str
            e.g. "rainy day melancholic indie", "euphoric summer dance music"
        n : int
            Number of results to return (after filtering and deduplication).

        Returns
        -------
        str — formatted context block for inclusion in an LLM prompt.
        """
        raw = self._kb.retrieve(vibe_description, n_results=min(n * 2, 20))
        results = _filter_by_distance(raw)
        results = _deduplicate(results)
        results = results[:n]
        logger.debug("retrieve_for_vibe(%r): %d raw → %d after filter", vibe_description, len(raw), len(results))
        return _format_block(results, query=vibe_description)

    def retrieve_for_artist(self, artist_name: str, n: int = 3) -> str:
        """
        Retrieve knowledge about a specific artist.

        First attempts an exact metadata match on the ``artist`` field.
        If that returns fewer than ``n`` results, falls back to a semantic
        search using the artist name, which may surface related genre guides
        and tag descriptions that contextualise the artist.

        Parameters
        ----------
        artist_name : str  — exact artist name as stored in KB metadata
        n : int

        Returns
        -------
        str — formatted context block.
        """
        # Pass 1: exact metadata match
        exact = self._kb.retrieve(
            artist_name,
            n_results=n,
            where_filter={"artist": artist_name},
        )
        exact = _filter_by_distance(exact)

        if len(exact) >= n:
            return _format_block(exact[:n], query=artist_name)

        # Pass 2: semantic fallback — enriches with genre context
        needed = n - len(exact)
        semantic = self._kb.retrieve(artist_name, n_results=needed * 3)
        semantic = _filter_by_distance(semantic)

        # Exclude IDs already in exact results so we don't double up
        exact_texts = {r["text"] for r in exact}
        semantic = [r for r in semantic if r["text"] not in exact_texts]
        semantic = _deduplicate(semantic)[:needed]

        combined = exact + semantic
        logger.debug(
            "retrieve_for_artist(%r): %d exact + %d semantic",
            artist_name, len(exact), len(semantic),
        )
        return _format_block(combined, query=artist_name)

    def retrieve_for_genre(self, genre: str, n: int = 3) -> str:
        """
        Retrieve genre guides and related content for a genre/tag name.

        Two-pass strategy:
          1. Filter by type=genre_guide for curated genre guides.
          2. Semantic search for artist bios and tag descriptions that
             mention the genre.

        Results from both passes are merged, deduplicated, and ranked
        by distance score.

        Parameters
        ----------
        genre : str  — genre name, e.g. "shoegaze", "trip-hop", "jazz"
        n : int

        Returns
        -------
        str — formatted context block.
        """
        # Pass 1: curated genre guides (most authoritative)
        guides = self._kb.retrieve(
            genre,
            n_results=n,
            where_filter={"type": "genre_guide"},
        )
        guides = _filter_by_distance(guides, max_dist=_MAX_DISTANCE + 0.3)

        # Pass 2: broader semantic search for artist bios + tag descriptions
        semantic = self._kb.retrieve(genre, n_results=n * 3)
        semantic = _filter_by_distance(semantic)

        # Merge: guides first, then semantic picks not already covered
        seen_texts = {r["text"] for r in guides}
        extras = [r for r in semantic if r["text"] not in seen_texts]
        extras = _deduplicate(extras)

        combined = guides + extras
        # Re-sort by distance so the most relevant float to the top
        combined.sort(key=lambda r: r["distance_score"])
        combined = combined[:n]

        logger.debug(
            "retrieve_for_genre(%r): %d guides + %d semantic extras → %d",
            genre, len(guides), len(extras), len(combined),
        )
        return _format_block(combined, query=genre)

    def retrieve_mixed(self, query: str, n: int = 5) -> str:
        """
        Retrieve a diverse mix of document types for a general query.

        Fetches a larger candidate pool (n*3), then applies three filters:
          - Distance cap (drop irrelevant results)
          - Per-artist cap (max 2 chunks per artist, to avoid one artist
            dominating the context when the KB has many bio chunks for them)
          - Near-duplicate removal

        The result is a context block that typically contains a mix of
        genre guides, artist bios, and tag descriptions, giving the LLM
        a rounded picture of the musical space the query describes.

        Parameters
        ----------
        query : str  — any natural-language query
        n : int

        Returns
        -------
        str — formatted context block.
        """
        raw = self._kb.retrieve(query, n_results=min(n * 3, 30))
        results = _filter_by_distance(raw)
        results = _cap_per_artist(results, max_per_artist=2)
        results = _deduplicate(results)
        results = results[:n]
        logger.debug(
            "retrieve_mixed(%r): %d raw → %d after filter/dedup/cap",
            query, len(raw), len(results),
        )
        return _format_block(results, query=query)
