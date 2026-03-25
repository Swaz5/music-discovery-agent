"""
Knowledge base ingestion.

Populates the MusicKnowledgeBase with real data from three sources:

  1. Last.fm artist bios — fetched via the artist.getInfo API, chunked by
     paragraph, and stored with artist/tag metadata.

  2. Last.fm tag descriptions — genre wiki pages from tag.getInfo, stored as
     single documents per genre with the tag as metadata.

  3. Curated Markdown files — human-written genre guides from data/knowledge/,
     split into sections at ## headers and stored with the filename as a tag.

All ingestion is idempotent: doc IDs are SHA-256 hashes of the text content,
so re-running never creates duplicates.

Usage
-----
    python -m src.rag.ingest          # full ingestion with default seed lists
    python -m src.rag.ingest --stats  # print current KB stats without ingesting
"""

import asyncio
import logging
import os
import re
from pathlib import Path

import httpx

from src.data import lastfm_client as lastfm
from src.rag.knowledge_base import MusicKnowledgeBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_ARTISTS = [
    # Classic rock / alternative
    "Radiohead", "The Smiths", "Joy Division", "Pixies", "Sonic Youth",
    "My Bloody Valentine", "Slowdive", "Ride", "Cocteau Twins", "The Cure",
    # Electronic
    "Aphex Twin", "Boards of Canada", "Massive Attack", "Portishead", "Burial",
    "Daft Punk", "Autechre", "Four Tet", "Caribou", "Bonobo",
    # Hip-hop / R&B
    "Kendrick Lamar", "Frank Ocean", "Tyler the Creator", "A Tribe Called Quest",
    "J Dilla", "MF DOOM", "Drake", "Kanye West", "SZA", "Steve Lacy",
    # Folk / indie
    "Nick Drake", "Elliott Smith", "Sufjan Stevens", "Bon Iver", "Fleet Foxes",
    "Iron & Wine", "Joanna Newsom", "Gillian Welch", "Wilco", "Big Thief",
    # Psych / garage
    "Tame Impala", "MGMT", "King Gizzard and the Lizard Wizard",
    "The Black Keys", "White Stripes",
]

SEED_TAGS = [
    "rock", "alternative rock", "indie rock", "post-punk", "shoegaze",
    "dream pop", "psychedelic rock", "grunge",
    "electronic", "ambient", "trip-hop", "house", "techno", "idm",
    "hip-hop", "rap", "rnb", "soul", "jazz",
    "folk", "singer-songwriter", "acoustic",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRIP_HTML = re.compile(r"<[^>]+>")
_STRIP_LASTFM_READ_MORE = re.compile(
    r"\s*<a\s+href=['\"]https?://www\.last\.fm[^\"']*['\"][^>]*>[^<]*</a>\s*",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Strip HTML tags and Last.fm boilerplate 'Read more' links."""
    text = _STRIP_LASTFM_READ_MORE.sub("", text)
    text = _STRIP_HTML.sub("", text)
    return text.strip()


def _chunk_by_paragraph(text: str, min_len: int = 50) -> list[str]:
    """Split on blank lines; discard chunks shorter than min_len characters."""
    chunks = [c.strip() for c in re.split(r"\n{2,}", text)]
    return [c for c in chunks if len(c) >= min_len]


def _sections_from_markdown(text: str, min_len: int = 50) -> list[tuple[str, str]]:
    """
    Split a Markdown file into (heading, body) pairs at ## headers.
    Returns a list of (section_title, section_text) tuples.
    The H1 title is treated as a preamble under the empty heading "".
    """
    # Split on lines that start with ##
    raw_sections = re.split(r"\n(?=##\s)", text)
    results = []
    for section in raw_sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        # ## Heading — strip leading #s
        if lines[0].startswith("##"):
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        elif lines[0].startswith("#"):
            # H1 title at top of file — use as preamble
            heading = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            heading = ""
            body = section.strip()

        if len(body) >= min_len:
            results.append((heading, body))

    return results


# ---------------------------------------------------------------------------
# Tag info fetcher (not in lastfm_client yet)
# ---------------------------------------------------------------------------

async def _fetch_tag_info(tag: str) -> str | None:
    """
    Fetch the wiki content for a Last.fm genre tag via tag.getInfo.
    Returns cleaned text or None if unavailable.
    """
    api_key = os.getenv("LASTFM_API_KEY")
    params = {
        "method": "tag.getInfo",
        "tag": tag,
        "api_key": api_key,
        "format": "json",
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://ws.audioscrobbler.com/2.0/", params=params)
            resp.raise_for_status()
            data = resp.json()

        content = (
            data.get("tag", {})
            .get("wiki", {})
            .get("content", "")
        )
        if not content:
            content = (
                data.get("tag", {})
                .get("wiki", {})
                .get("summary", "")
            )
        return _clean_text(content) if content else None
    except Exception as exc:
        logger.warning("tag.getInfo failed for %r: %s", tag, exc)
        return None


# ---------------------------------------------------------------------------
# Ingestion functions
# ---------------------------------------------------------------------------

async def ingest_artist_bios(
    artists: list[str],
    kb: MusicKnowledgeBase,
) -> int:
    """
    Fetch Last.fm bios for each artist, chunk by paragraph, and store.

    Each chunk gets metadata:
        source : "lastfm"
        type   : "artist_bio"
        artist : <artist name>
        tags   : top genre tags as comma-separated string

    Returns the number of document chunks stored.
    """
    total = 0

    async def _ingest_one(name: str) -> int:
        try:
            info = await lastfm.get_artist_info(name)
        except Exception as exc:
            logger.warning("Last.fm artist.getInfo failed for %r: %s", name, exc)
            return 0

        bio_raw = info.get("bio_summary", "")
        if not bio_raw:
            logger.debug("No bio for %r — skipping", name)
            return 0

        bio = _clean_text(bio_raw)
        chunks = _chunk_by_paragraph(bio)
        if not chunks:
            return 0

        tags_list = info.get("tags", [])[:10]
        tags_str = ", ".join(tags_list)

        texts, metadatas = [], []
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": "lastfm",
                "type": "artist_bio",
                "artist": name,
                "tags": tags_str,
            })

        kb.add_documents_batch(texts, metadatas)
        logger.info("Ingested %d bio chunks for %r", len(texts), name)
        return len(texts)

    # Fetch all bios concurrently
    counts = await asyncio.gather(*[_ingest_one(a) for a in artists])
    total = sum(counts)
    return total


async def ingest_tag_descriptions(
    tags: list[str],
    kb: MusicKnowledgeBase,
) -> int:
    """
    Fetch Last.fm wiki descriptions for genre tags and store each as one doc.

    Metadata:
        source : "lastfm"
        type   : "genre_description"
        tag    : <tag name>

    Returns the number of documents stored.
    """
    total = 0

    async def _ingest_one(tag: str) -> int:
        content = await _fetch_tag_info(tag)
        if not content:
            logger.debug("No wiki content for tag %r — skipping", tag)
            return 0

        kb.add_document(
            content,
            {
                "source": "lastfm",
                "type": "genre_description",
                "tag": tag,
                "artist": "",
            },
        )
        logger.info("Ingested tag description for %r (%d chars)", tag, len(content))
        return 1

    counts = await asyncio.gather(*[_ingest_one(t) for t in tags])
    total = sum(counts)
    return total


def ingest_curated_knowledge(
    kb: MusicKnowledgeBase,
    directory: str = "data/knowledge/",
) -> int:
    """
    Read all .md files from ``directory``, split into ## sections, and store.

    Metadata:
        source : "curated"
        type   : "genre_guide"
        tags   : filename stem with hyphens replaced by spaces
                 (e.g. "electronic.md" → "electronic")

    Returns the number of section documents stored.
    """
    knowledge_dir = Path(directory)
    if not knowledge_dir.exists():
        logger.warning("Knowledge directory %r does not exist — skipping curated ingestion", directory)
        return 0

    md_files = sorted(knowledge_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %r", directory)
        return 0

    total = 0
    for md_file in md_files:
        tag_hint = md_file.stem.replace("-", " ")  # "hip-hop-rnb" → "hip hop rnb"
        text = md_file.read_text(encoding="utf-8")
        sections = _sections_from_markdown(text)

        texts, metadatas = [], []
        for heading, body in sections:
            # Compose document: include heading in the text for better retrieval
            doc_text = f"{heading}\n\n{body}".strip() if heading else body
            texts.append(doc_text)
            metadatas.append({
                "source": "curated",
                "type": "genre_guide",
                "artist": "",
                "tags": tag_hint,
            })

        if texts:
            kb.add_documents_batch(texts, metadatas)
            logger.info("Ingested %d sections from %s", len(texts), md_file.name)
            total += len(texts)

    return total


async def run_full_ingestion(
    artists: list[str] | None = None,
    tags: list[str] | None = None,
    knowledge_dir: str = "data/knowledge/",
    kb: MusicKnowledgeBase | None = None,
) -> dict:
    """
    Run all three ingestion pipelines and return a summary dict.

    Parameters
    ----------
    artists       : seed artist list (defaults to SEED_ARTISTS)
    tags          : seed genre tag list (defaults to SEED_TAGS)
    knowledge_dir : path to .md files (defaults to "data/knowledge/")
    kb            : MusicKnowledgeBase instance (created if None)

    Returns
    -------
    {
        "artist_bios"      : int,
        "tag_descriptions" : int,
        "curated_docs"     : int,
        "total"            : int,
    }
    """
    if artists is None:
        artists = SEED_ARTISTS
    if tags is None:
        tags = SEED_TAGS
    if kb is None:
        kb = MusicKnowledgeBase()

    logger.info("Starting full ingestion — %d artists, %d tags", len(artists), len(tags))

    # Artist bios and tag descriptions can run in parallel;
    # curated ingestion is synchronous (local files) so it runs first.
    curated = ingest_curated_knowledge(kb, directory=knowledge_dir)

    artist_bios, tag_descriptions = await asyncio.gather(
        ingest_artist_bios(artists, kb),
        ingest_tag_descriptions(tags, kb),
    )

    stats = kb.get_stats()
    result = {
        "artist_bios": artist_bios,
        "tag_descriptions": tag_descriptions,
        "curated_docs": curated,
        "total": artist_bios + tag_descriptions + curated,
        "collection_total": stats["document_count"],
    }
    logger.info("Ingestion complete: %s", result)
    return result


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Populate the music knowledge base")
    parser.add_argument("--stats", action="store_true", help="Print KB stats without ingesting")
    parser.add_argument("--persist-dir", default=".chroma", help="ChromaDB persistence directory")
    args = parser.parse_args()

    kb = MusicKnowledgeBase(persist_dir=args.persist_dir)

    if args.stats:
        stats = kb.get_stats()
        print(f"\nKnowledge base stats ({args.persist_dir}):")
        print(f"  Total documents : {stats['document_count']}")
        print(f"  By source       : {stats['sources']}")
        print(f"  By type         : {stats['types']}")
        sys.exit(0)

    print(f"\nRunning full ingestion into {args.persist_dir!r} …")
    print(f"  {len(SEED_ARTISTS)} artists  ·  {len(SEED_TAGS)} genre tags  ·  data/knowledge/*.md")
    print()

    result = asyncio.run(run_full_ingestion(kb=kb))

    print()
    print("─" * 48)
    print("  Ingestion summary")
    print("─" * 48)
    print(f"  Artist bio chunks   : {result['artist_bios']:>4}")
    print(f"  Tag descriptions    : {result['tag_descriptions']:>4}")
    print(f"  Curated sections    : {result['curated_docs']:>4}")
    print(f"  ──────────────────────────")
    print(f"  Added this run      : {result['total']:>4}")
    print(f"  Collection total    : {result['collection_total']:>4}")
    print(f"  (re-run is safe — duplicates use upsert)")
    print()
