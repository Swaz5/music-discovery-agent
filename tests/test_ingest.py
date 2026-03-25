"""
Tests for src/rag/ingest.py.

All Last.fm HTTP calls are mocked so tests run offline and fast.
ChromaDB runs in-process against a tmp_path directory.
"""

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.rag.knowledge_base import MusicKnowledgeBase
from src.rag.ingest import (
    _clean_text,
    _chunk_by_paragraph,
    _sections_from_markdown,
    ingest_artist_bios,
    ingest_tag_descriptions,
    ingest_curated_knowledge,
    run_full_ingestion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    return MusicKnowledgeBase(persist_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Unit tests: text helpers
# ---------------------------------------------------------------------------

def test_clean_text_strips_html():
    assert _clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_clean_text_strips_lastfm_read_more():
    raw = 'Radiohead is a band. <a href="https://www.last.fm/music/Radiohead">Read more on Last.fm</a>'
    result = _clean_text(raw)
    assert "Read more" not in result
    assert "Radiohead is a band." in result


def test_clean_text_strips_nested_html():
    assert _clean_text("<div><span>text</span></div>") == "text"


def test_clean_text_plain_text_unchanged():
    plain = "Just plain text with no tags."
    assert _clean_text(plain) == plain


def test_chunk_by_paragraph_splits_on_blank_lines():
    text = "Para one.\n\nPara two.\n\nPara three."
    # Pass min_len=1 — we're testing splitting, not the length filter
    chunks = _chunk_by_paragraph(text, min_len=1)
    assert len(chunks) == 3


def test_chunk_by_paragraph_discards_short_chunks():
    text = "Short.\n\nThis paragraph is long enough to be kept as a valid chunk."
    chunks = _chunk_by_paragraph(text, min_len=50)
    assert len(chunks) == 1
    assert "long enough" in chunks[0]


def test_chunk_by_paragraph_strips_whitespace():
    text = "  Hello world paragraph.  \n\n  Another paragraph here.  "
    chunks = _chunk_by_paragraph(text, min_len=5)
    assert chunks[0] == "Hello world paragraph."
    assert chunks[1] == "Another paragraph here."


def test_chunk_by_paragraph_empty_string():
    assert _chunk_by_paragraph("") == []


def test_sections_from_markdown_splits_at_h2():
    md = textwrap.dedent("""\
        # Guide Title

        ## Section One
        Content of section one with enough text to pass the minimum length filter.

        ## Section Two
        Content of section two with enough text to pass the minimum length filter.
    """)
    sections = _sections_from_markdown(md, min_len=10)
    headings = [h for h, _ in sections]
    assert "Section One" in headings
    assert "Section Two" in headings


def test_sections_from_markdown_heading_in_body_text():
    """The heading should appear in the composed doc text."""
    md = "## Shoegaze\nShoegaze emerged in the late 1980s in the UK as a guitar-driven sound."
    sections = _sections_from_markdown(md, min_len=10)
    # Composed text = "Shoegaze\n\n<body>"
    combined = "\n".join(body for _, body in sections)
    assert "1980s" in combined


def test_sections_from_markdown_skips_short_bodies():
    md = "## Short\nToo short.\n\n## Long Enough\nThis body is certainly long enough to pass the minimum."
    sections = _sections_from_markdown(md, min_len=30)
    headings = [h for h, _ in sections]
    assert "Short" not in headings
    assert "Long Enough" in headings


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

MOCK_ARTIST_INFO = {
    "name": "Radiohead",
    "bio_summary": (
        "Radiohead is an English rock band formed in Abingdon, Oxfordshire in 1985.\n\n"
        "They are known for experimental sounds, complex arrangements, and innovative production."
    ),
    "tags": ["alternative rock", "art rock", "electronic"],
    "similar_artists": [],
    "listeners": "4800000",
    "playcount": "600000000",
}

MOCK_TAG_INFO_RESPONSE = {
    "tag": {
        "name": "shoegaze",
        "wiki": {
            "summary": "Shoegaze is a subgenre that emerged in the UK.",
            "content": (
                "Shoegaze is a subgenre of alternative rock that emerged from the "
                "United Kingdom in the late 1980s, characterised by ethereal vocals "
                "and walls of guitar noise created by heavy use of effects pedals."
            ),
        },
    }
}


def _mock_httpx_get(json_data: dict, status_code: int = 200):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    return patch("src.rag.ingest.httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# Tests: ingest_artist_bios
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_artist_bios_adds_documents(kb):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)):
        count = await ingest_artist_bios(["Radiohead"], kb)

    assert count > 0
    assert kb.get_stats()["document_count"] == count


@pytest.mark.asyncio
async def test_ingest_artist_bios_metadata(kb):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)):
        await ingest_artist_bios(["Radiohead"], kb)

    results = kb.retrieve("rock band", n_results=5, where_filter={"artist": "Radiohead"})
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["source"] == "lastfm"
        assert r["metadata"]["type"] == "artist_bio"
        assert r["metadata"]["artist"] == "Radiohead"


@pytest.mark.asyncio
async def test_ingest_artist_bios_chunks_long_bio(kb):
    """Bio with two paragraphs → 2 chunks."""
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)):
        count = await ingest_artist_bios(["Radiohead"], kb)

    assert count == 2  # mock bio has exactly two paragraphs above 50 chars


@pytest.mark.asyncio
async def test_ingest_artist_bios_skips_empty_bio(kb):
    empty_info = {**MOCK_ARTIST_INFO, "bio_summary": ""}
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=empty_info)):
        count = await ingest_artist_bios(["NoInfo"], kb)

    assert count == 0
    assert kb.get_stats()["document_count"] == 0


@pytest.mark.asyncio
async def test_ingest_artist_bios_multiple_artists(kb):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)):
        count = await ingest_artist_bios(["Radiohead", "Portishead"], kb)

    # Each artist has 2 paragraphs → 4 total, but since the mock returns the
    # same text for both, the hashes collide and upsert deduplicates to 2.
    assert kb.get_stats()["document_count"] == 2


@pytest.mark.asyncio
async def test_ingest_artist_bios_handles_lastfm_error(kb):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(side_effect=Exception("API down"))):
        count = await ingest_artist_bios(["Radiohead"], kb)

    assert count == 0
    assert kb.get_stats()["document_count"] == 0


@pytest.mark.asyncio
async def test_ingest_artist_bios_idempotent(kb):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)):
        count1 = await ingest_artist_bios(["Radiohead"], kb)
        count2 = await ingest_artist_bios(["Radiohead"], kb)

    assert kb.get_stats()["document_count"] == count1  # same after second run


# ---------------------------------------------------------------------------
# Tests: ingest_tag_descriptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ingest_tag_descriptions_adds_document(kb):
    with _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        count = await ingest_tag_descriptions(["shoegaze"], kb)

    assert count == 1
    assert kb.get_stats()["document_count"] == 1


@pytest.mark.asyncio
async def test_ingest_tag_descriptions_metadata(kb):
    with _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        await ingest_tag_descriptions(["shoegaze"], kb)

    results = kb.retrieve("guitar noise", n_results=1, where_filter={"type": "genre_description"})
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "lastfm"
    assert results[0]["metadata"]["type"] == "genre_description"
    assert results[0]["metadata"]["tag"] == "shoegaze"


@pytest.mark.asyncio
async def test_ingest_tag_descriptions_skips_empty_wiki(kb):
    empty_response = {"tag": {"name": "sometag", "wiki": {"content": "", "summary": ""}}}
    with _mock_httpx_get(empty_response):
        count = await ingest_tag_descriptions(["sometag"], kb)

    assert count == 0


@pytest.mark.asyncio
async def test_ingest_tag_descriptions_multiple_tags(kb):
    with _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        count = await ingest_tag_descriptions(["shoegaze", "dream pop"], kb)

    # Same mock content → same hash → 1 unique document
    assert count == 2
    assert kb.get_stats()["document_count"] == 1


@pytest.mark.asyncio
async def test_ingest_tag_descriptions_idempotent(kb):
    with _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        count1 = await ingest_tag_descriptions(["shoegaze"], kb)
        count2 = await ingest_tag_descriptions(["shoegaze"], kb)

    assert count1 == 1
    assert count2 == 1
    assert kb.get_stats()["document_count"] == 1  # upsert: no growth


# ---------------------------------------------------------------------------
# Tests: ingest_curated_knowledge
# ---------------------------------------------------------------------------

def test_ingest_curated_knowledge_reads_md_files(kb, tmp_path):
    (tmp_path / "electronic.md").write_text(
        "# Electronic Guide\n\n"
        "## House Music\n"
        "House music originated in Chicago in the early 1980s with four-on-the-floor beats.\n\n"
        "## Techno\n"
        "Detroit techno developed as a blend of Kraftwerk and funk in the mid-1980s.\n",
        encoding="utf-8",
    )
    count = ingest_curated_knowledge(kb, directory=str(tmp_path))
    assert count == 2  # H1 preamble has no body → skipped; 2 H2 sections kept


def test_ingest_curated_knowledge_metadata(kb, tmp_path):
    (tmp_path / "hip-hop-rnb.md").write_text(
        "## Boom Bap\nBoom bap is a style of hip-hop defined by hard-hitting kick-snare patterns and sampling.\n",
        encoding="utf-8",
    )
    ingest_curated_knowledge(kb, directory=str(tmp_path))
    results = kb.retrieve("hip hop beats", n_results=1)
    assert results[0]["metadata"]["source"] == "curated"
    assert results[0]["metadata"]["type"] == "genre_guide"
    assert "hip hop rnb" in results[0]["metadata"]["tags"]


def test_ingest_curated_knowledge_missing_directory(kb):
    count = ingest_curated_knowledge(kb, directory="/nonexistent/path/")
    assert count == 0
    assert kb.get_stats()["document_count"] == 0


def test_ingest_curated_knowledge_empty_directory(kb, tmp_path):
    count = ingest_curated_knowledge(kb, directory=str(tmp_path))
    assert count == 0


def test_ingest_curated_knowledge_idempotent(kb, tmp_path):
    (tmp_path / "test.md").write_text(
        "## Genre\nThis is a long enough section about a music genre with details.\n",
        encoding="utf-8",
    )
    count1 = ingest_curated_knowledge(kb, directory=str(tmp_path))
    count2 = ingest_curated_knowledge(kb, directory=str(tmp_path))
    assert count1 == count2
    assert kb.get_stats()["document_count"] == count1  # no growth


# ---------------------------------------------------------------------------
# Tests: run_full_ingestion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_full_ingestion_returns_summary(kb, tmp_path):
    (tmp_path / "electronic.md").write_text(
        "## Techno\nTechno music originated in Detroit and uses synthesisers and drum machines.\n",
        encoding="utf-8",
    )
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)), \
         _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        result = await run_full_ingestion(
            artists=["Radiohead"],
            tags=["shoegaze"],
            knowledge_dir=str(tmp_path),
            kb=kb,
        )

    assert "artist_bios" in result
    assert "tag_descriptions" in result
    assert "curated_docs" in result
    assert "total" in result
    assert "collection_total" in result
    assert result["total"] == result["artist_bios"] + result["tag_descriptions"] + result["curated_docs"]


@pytest.mark.asyncio
async def test_run_full_ingestion_collection_total_matches_kb(kb, tmp_path):
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)), \
         _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        result = await run_full_ingestion(
            artists=["Radiohead"],
            tags=["shoegaze"],
            knowledge_dir=str(tmp_path),
            kb=kb,
        )

    assert result["collection_total"] == kb.get_stats()["document_count"]


@pytest.mark.asyncio
async def test_run_full_ingestion_idempotent(kb, tmp_path):
    kwargs = dict(
        artists=["Radiohead"],
        tags=["shoegaze"],
        knowledge_dir=str(tmp_path),
        kb=kb,
    )
    with patch("src.rag.ingest.lastfm.get_artist_info", AsyncMock(return_value=MOCK_ARTIST_INFO)), \
         _mock_httpx_get(MOCK_TAG_INFO_RESPONSE):
        await run_full_ingestion(**kwargs)
        result2 = await run_full_ingestion(**kwargs)

    # Collection should not grow on the second run
    assert result2["collection_total"] == kb.get_stats()["document_count"]
