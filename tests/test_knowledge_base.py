"""
Tests for MusicKnowledgeBase.

Each test gets a fresh ChromaDB collection backed by a tmp_path directory
so tests are isolated from each other and from the real .chroma store.

Semantic-ranking tests use texts and queries that are far enough apart in
meaning that MiniLM-L6-v2 reliably distinguishes them. Exact distances will
vary between runs (ONNX model version differences, etc.) so we test relative
ordering, not specific values.
"""

import pytest

from src.rag.knowledge_base import MusicKnowledgeBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    """Fresh knowledge base in a temp directory."""
    return MusicKnowledgeBase(persist_dir=str(tmp_path))


DANCE_TEXT = (
    "High-energy electronic dance music with pounding four-on-the-floor kicks, "
    "relentless basslines, and euphoric synth stabs designed for peak-hour dancefloors."
)
DANCE_META = {
    "source": "curated",
    "type": "genre_guide",
    "artist": "",
    "tags": "electronic, dance, high energy, techno, house",
}

ACOUSTIC_TEXT = (
    "Quiet acoustic folk music featuring fingerpicked guitar, intimate vocals, "
    "and introspective lyrics recorded in a sparse, minimal arrangement."
)
ACOUSTIC_META = {
    "source": "curated",
    "type": "genre_guide",
    "artist": "",
    "tags": "folk, acoustic, quiet, minimal, singer-songwriter",
}

DREAMY_TEXT = (
    "Ethereal dream pop with heavily reverbed guitars, hushed vocals lost in "
    "layers of delay, and a floating, weightless atmosphere that washes over "
    "you like a warm wave."
)
DREAMY_META = {
    "source": "curated",
    "type": "genre_guide",
    "artist": "",
    "tags": "dream pop, shoegaze, atmospheric, ethereal, reverb",
}

BIO_TEXT = (
    "Radiohead is an English rock band from Abingdon, Oxfordshire. "
    "Known for blending rock with electronic and ambient textures, "
    "their albums include OK Computer and Kid A."
)
BIO_META = {
    "source": "lastfm",
    "type": "bio",
    "artist": "Radiohead",
    "tags": "alternative rock, art rock, electronic",
}

REVIEW_TEXT = (
    "Tame Impala's Currents is a psychedelic pop masterpiece. "
    "Kevin Parker's production is lush and enveloping, full of synthesizers "
    "and modern production techniques that feel both nostalgic and futuristic."
)
REVIEW_META = {
    "source": "curated",
    "type": "review",
    "artist": "Tame Impala",
    "tags": "psychedelic, pop, electronic, indie",
}


# ---------------------------------------------------------------------------
# Tests: add_document
# ---------------------------------------------------------------------------

def test_add_document_returns_string_id(kb):
    doc_id = kb.add_document(DANCE_TEXT, DANCE_META)
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0


def test_add_document_explicit_id(kb):
    doc_id = kb.add_document(DANCE_TEXT, DANCE_META, doc_id="my-custom-id")
    assert doc_id == "my-custom-id"


def test_add_document_autoid_is_deterministic(kb):
    id1 = kb.add_document(DANCE_TEXT, DANCE_META)
    id2 = kb.add_document(DANCE_TEXT, DANCE_META)
    assert id1 == id2  # same text → same hash → upsert, not duplicate


def test_add_document_different_texts_different_ids(kb):
    id1 = kb.add_document(DANCE_TEXT, DANCE_META)
    id2 = kb.add_document(ACOUSTIC_TEXT, ACOUSTIC_META)
    assert id1 != id2


def test_add_document_increments_count(kb):
    assert kb.get_stats()["document_count"] == 0
    kb.add_document(DANCE_TEXT, DANCE_META)
    assert kb.get_stats()["document_count"] == 1
    kb.add_document(ACOUSTIC_TEXT, ACOUSTIC_META)
    assert kb.get_stats()["document_count"] == 2


def test_add_document_upsert_does_not_duplicate(kb):
    kb.add_document(DANCE_TEXT, DANCE_META, doc_id="dup-test")
    kb.add_document(DANCE_TEXT, DANCE_META, doc_id="dup-test")  # same id
    assert kb.get_stats()["document_count"] == 1


def test_add_document_metadata_list_serialised(kb):
    """Lists in metadata should be stored as comma-separated strings."""
    kb.add_document(
        DANCE_TEXT,
        {"source": "curated", "type": "genre_guide", "artist": "", "tags": ["a", "b", "c"]},
        doc_id="list-meta",
    )
    results = kb.retrieve("dance energy", n_results=1)
    assert results[0]["metadata"]["tags"] == "a, b, c"


# ---------------------------------------------------------------------------
# Tests: add_documents_batch
# ---------------------------------------------------------------------------

def test_add_documents_batch_returns_ids(kb):
    ids = kb.add_documents_batch(
        [DANCE_TEXT, ACOUSTIC_TEXT],
        [DANCE_META, ACOUSTIC_META],
    )
    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)


def test_add_documents_batch_count(kb):
    kb.add_documents_batch(
        [DANCE_TEXT, ACOUSTIC_TEXT, DREAMY_TEXT],
        [DANCE_META, ACOUSTIC_META, DREAMY_META],
    )
    assert kb.get_stats()["document_count"] == 3


def test_add_documents_batch_explicit_ids(kb):
    ids = kb.add_documents_batch(
        [DANCE_TEXT, ACOUSTIC_TEXT],
        [DANCE_META, ACOUSTIC_META],
        doc_ids=["id-a", "id-b"],
    )
    assert ids == ["id-a", "id-b"]


def test_add_documents_batch_mismatched_lengths_raises(kb):
    with pytest.raises(ValueError, match="same length"):
        kb.add_documents_batch([DANCE_TEXT], [DANCE_META, ACOUSTIC_META])


# ---------------------------------------------------------------------------
# Tests: retrieve — basic
# ---------------------------------------------------------------------------

def _populate(kb) -> None:
    kb.add_documents_batch(
        [DANCE_TEXT, ACOUSTIC_TEXT, DREAMY_TEXT, BIO_TEXT, REVIEW_TEXT],
        [DANCE_META, ACOUSTIC_META, DREAMY_META, BIO_META, REVIEW_META],
    )


def test_retrieve_returns_list(kb):
    _populate(kb)
    results = kb.retrieve("electronic dance")
    assert isinstance(results, list)


def test_retrieve_result_has_required_keys(kb):
    _populate(kb)
    results = kb.retrieve("dance music", n_results=1)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "metadata" in r
    assert "distance_score" in r


def test_retrieve_respects_n_results(kb):
    _populate(kb)
    assert len(kb.retrieve("music", n_results=2)) == 2
    assert len(kb.retrieve("music", n_results=4)) == 4


def test_retrieve_n_results_capped_at_collection_size(kb):
    kb.add_document(DANCE_TEXT, DANCE_META)  # only 1 document
    results = kb.retrieve("dance", n_results=10)
    assert len(results) == 1


def test_retrieve_empty_collection_returns_empty(kb):
    assert kb.retrieve("anything") == []


def test_retrieve_distance_scores_are_floats(kb):
    _populate(kb)
    results = kb.retrieve("dance", n_results=3)
    for r in results:
        assert isinstance(r["distance_score"], float)


def test_retrieve_distances_non_negative(kb):
    _populate(kb)
    results = kb.retrieve("music", n_results=5)
    for r in results:
        assert r["distance_score"] >= 0.0


# ---------------------------------------------------------------------------
# Tests: retrieve — semantic ranking
# ---------------------------------------------------------------------------

def test_retrieve_dance_query_ranks_dance_first(kb):
    """
    'high energy dancefloor electronic music' should retrieve the dance
    document before the acoustic one.
    """
    _populate(kb)
    results = kb.retrieve("high energy dancefloor electronic music", n_results=5)
    texts = [r["text"] for r in results]
    assert texts.index(DANCE_TEXT) < texts.index(ACOUSTIC_TEXT)


def test_retrieve_acoustic_query_ranks_acoustic_first(kb):
    """
    'quiet fingerpicked acoustic folk guitar' should retrieve the acoustic
    document before the dance one.
    """
    _populate(kb)
    results = kb.retrieve("quiet fingerpicked acoustic folk guitar", n_results=5)
    texts = [r["text"] for r in results]
    assert texts.index(ACOUSTIC_TEXT) < texts.index(DANCE_TEXT)


def test_retrieve_dreamy_query_ranks_dreamy_first(kb):
    """
    'ethereal dreamy reverb atmosphere' should rank the dreamy text highest.
    """
    _populate(kb)
    results = kb.retrieve("ethereal dreamy reverb atmosphere", n_results=5)
    assert results[0]["text"] == DREAMY_TEXT


def test_retrieve_artist_bio_query(kb):
    """
    Query for 'Radiohead band biography' should rank the bio before the review.
    """
    _populate(kb)
    results = kb.retrieve("Radiohead band biography", n_results=5)
    texts = [r["text"] for r in results]
    assert texts.index(BIO_TEXT) < texts.index(REVIEW_TEXT)


def test_retrieve_results_sorted_by_distance(kb):
    """Results must be in ascending distance order (closest first)."""
    _populate(kb)
    results = kb.retrieve("music genre", n_results=5)
    distances = [r["distance_score"] for r in results]
    assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# Tests: retrieve — metadata filtering
# ---------------------------------------------------------------------------

def test_retrieve_filter_by_type(kb):
    _populate(kb)
    results = kb.retrieve("music", n_results=5, where_filter={"type": "bio"})
    assert all(r["metadata"]["type"] == "bio" for r in results)
    assert len(results) == 1  # only BIO_TEXT has type=bio


def test_retrieve_filter_by_source(kb):
    _populate(kb)
    results = kb.retrieve("music", n_results=5, where_filter={"source": "lastfm"})
    assert all(r["metadata"]["source"] == "lastfm" for r in results)


def test_retrieve_filter_by_artist(kb):
    _populate(kb)
    results = kb.retrieve("psychedelic", n_results=5, where_filter={"artist": "Tame Impala"})
    assert len(results) == 1
    assert results[0]["metadata"]["artist"] == "Tame Impala"


def test_retrieve_filter_no_matches_returns_empty(kb):
    _populate(kb)
    results = kb.retrieve("music", n_results=5, where_filter={"type": "nonexistent_type"})
    assert results == []


def test_retrieve_filter_genre_guide_only(kb):
    _populate(kb)
    results = kb.retrieve(
        "atmospheric sound",
        n_results=5,
        where_filter={"type": "genre_guide"},
    )
    assert all(r["metadata"]["type"] == "genre_guide" for r in results)
    assert len(results) == 3  # DANCE, ACOUSTIC, DREAMY


# ---------------------------------------------------------------------------
# Tests: delete_collection and get_stats
# ---------------------------------------------------------------------------

def test_delete_collection_resets_count(kb):
    _populate(kb)
    assert kb.get_stats()["document_count"] == 5
    kb.delete_collection()
    assert kb.get_stats()["document_count"] == 0


def test_delete_collection_object_still_usable(kb):
    _populate(kb)
    kb.delete_collection()
    # Should be able to add documents after reset
    kb.add_document(DANCE_TEXT, DANCE_META)
    assert kb.get_stats()["document_count"] == 1


def test_get_stats_empty(kb):
    stats = kb.get_stats()
    assert stats["document_count"] == 0
    assert stats["sources"] == {}
    assert stats["types"] == {}


def test_get_stats_counts_sources(kb):
    _populate(kb)
    stats = kb.get_stats()
    assert stats["sources"]["curated"] == 4
    assert stats["sources"]["lastfm"] == 1


def test_get_stats_counts_types(kb):
    _populate(kb)
    stats = kb.get_stats()
    assert stats["types"]["genre_guide"] == 3
    assert stats["types"]["bio"] == 1
    assert stats["types"]["review"] == 1
