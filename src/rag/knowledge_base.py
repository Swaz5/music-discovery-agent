"""
RAG knowledge base.

Loads curated genre guides and music articles from data/knowledge/, chunks and
embeds them into a ChromaDB vector store, and exposes a retrieval interface for
fetching context-relevant passages to ground agent recommendations.

Storage
-------
ChromaDB persists vectors to .chroma/ by default. The collection is called
"music_knowledge". Embeddings are computed locally via ChromaDB's built-in
DefaultEmbeddingFunction (ONNXMiniLM-L6-v2, no API key required).

Document metadata schema
------------------------
  source : str   — "lastfm" | "deezer" | "curated"
  type   : str   — "bio" | "genre_guide" | "review" | "track_note"
  artist : str   — artist name (empty string when not applicable)
  tags   : str   — comma-separated genre/mood tags

distance_score in retrieve() results is the raw L2 distance from ChromaDB:
lower = more similar.  Scores are NOT normalised to [0, 1] so callers that
want a "similarity" can compute  sim = 1 / (1 + distance).
"""

import hashlib
import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

COLLECTION_NAME = "music_knowledge"


class MusicKnowledgeBase:
    """
    Semantic vector store for music knowledge built on ChromaDB.

    All embedding is handled locally by ONNXMiniLM-L6-v2 (the ChromaDB
    default). No external API calls or keys are required.
    """

    def __init__(self, persist_dir: str = ".chroma") -> None:
        """
        Initialise a persistent ChromaDB client and get-or-create the
        "music_knowledge" collection.

        Parameters
        ----------
        persist_dir : str
            Directory where ChromaDB will persist its data.
            Created automatically if it does not exist.
        """
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "l2"},
        )
        logger.debug(
            "MusicKnowledgeBase ready — collection %r has %d documents",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Adding documents
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(text: str) -> str:
        """Stable, collision-resistant ID from the SHA-256 of the text."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    @staticmethod
    def _normalise_metadata(metadata: dict) -> dict:
        """
        Ensure all metadata values are ChromaDB-compatible (str/int/float/bool).
        Lists are serialised as comma-separated strings.
        """
        out = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                out[k] = ", ".join(str(i) for i in v)
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def add_document(
        self,
        text: str,
        metadata: dict,
        doc_id: str | None = None,
    ) -> str:
        """
        Embed and store a single document.

        Parameters
        ----------
        text     : str   — the document text to embed and retrieve later
        metadata : dict  — should include source, type, artist, tags
        doc_id   : str | None — if None, auto-generated from text hash

        Returns
        -------
        str — the doc_id used (auto-generated or as provided)
        """
        if doc_id is None:
            doc_id = self._make_id(text)

        self._collection.upsert(
            documents=[text],
            metadatas=[self._normalise_metadata(metadata)],
            ids=[doc_id],
        )
        logger.debug("Upserted document %r (type=%s)", doc_id, metadata.get("type"))
        return doc_id

    def add_documents_batch(
        self,
        texts: list[str],
        metadatas: list[dict],
        doc_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Batch-embed and store multiple documents.

        ChromaDB calls the embedding function once for the whole batch, which
        is faster than individual upserts for large document sets.

        Parameters
        ----------
        texts     : list[str]
        metadatas : list[dict]
        doc_ids   : list[str] | None — auto-generated from text hashes if None

        Returns
        -------
        list[str] — the doc_ids used
        """
        if len(texts) != len(metadatas):
            raise ValueError(
                f"texts and metadatas must have the same length "
                f"({len(texts)} vs {len(metadatas)})"
            )

        if doc_ids is None:
            doc_ids = [self._make_id(t) for t in texts]

        self._collection.upsert(
            documents=texts,
            metadatas=[self._normalise_metadata(m) for m in metadatas],
            ids=doc_ids,
        )
        logger.debug("Batch-upserted %d documents", len(texts))
        return doc_ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where_filter: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search: find the n_results most relevant documents.

        Parameters
        ----------
        query        : str  — natural-language query
        n_results    : int  — max results to return
        where_filter : dict | None — ChromaDB metadata filter, e.g.
                       {"type": "genre_guide"} or
                       {"$and": [{"source": "curated"}, {"artist": "Radiohead"}]}

        Returns
        -------
        list[dict] with keys:
            text           : str   — original document text
            metadata       : dict  — stored metadata
            distance_score : float — L2 distance (lower = more similar)
        """
        # ChromaDB raises if n_results exceeds collection size
        count = self._collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)

        # Unzip the per-query lists (we always send exactly one query)
        documents  = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance_score": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_collection(self) -> None:
        """
        Permanently delete the music_knowledge collection and all its data.
        Useful during development / test teardown.
        """
        self._client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted collection %r", COLLECTION_NAME)
        # Re-create so the object is still usable
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "l2"},
        )

    def get_stats(self) -> dict:
        """
        Return a summary of the collection: document count and a breakdown
        of metadata fields (source, type).
        """
        count = self._collection.count()
        if count == 0:
            return {"document_count": 0, "sources": {}, "types": {}}

        # Fetch all metadata (no embeddings needed)
        results = self._collection.get(include=["metadatas"])
        metadatas = results["metadatas"]

        sources: dict[str, int] = {}
        types: dict[str, int] = {}
        for m in metadatas:
            src = m.get("source", "unknown")
            typ = m.get("type", "unknown")
            sources[src] = sources.get(src, 0) + 1
            types[typ] = types.get(typ, 0) + 1

        return {
            "document_count": count,
            "sources": sources,
            "types": types,
        }


# ---------------------------------------------------------------------------
# __main__: demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Use a temp dir so the demo doesn't pollute the real .chroma store
    # ignore_cleanup_errors=True: ChromaDB holds SQLite/HNSW file handles
    # open on Windows, which would otherwise raise PermissionError on cleanup.
    with tempfile.TemporaryDirectory(prefix="chroma_demo_", ignore_cleanup_errors=True) as tmp:
        kb = MusicKnowledgeBase(persist_dir=tmp)

        # --- 1. Add three documents ---
        docs = [
            (
                "Cocteau Twins are a Scottish dream pop band formed in 1979. "
                "Their music is defined by heavily treated guitars, ethereal "
                "vocals from Elizabeth Fraser, and lush, atmospheric production. "
                "Albums like 'Heaven or Las Vegas' create an enveloping, "
                "otherworldly soundscape that is simultaneously dreamy and intense.",
                {
                    "source": "curated",
                    "type": "bio",
                    "artist": "Cocteau Twins",
                    "tags": "dream pop, shoegaze, atmospheric, ethereal, 4AD",
                },
            ),
            (
                "Shoegaze is a subgenre of alternative rock that emerged in the "
                "UK in the late 1980s. It is characterised by a 'wall of sound' "
                "created by heavily distorted guitars with lots of reverb and "
                "delay, introspective lyrics, and hushed vocals that are often "
                "treated as another instrument in the mix. Key artists include "
                "My Bloody Valentine, Slowdive, Ride, and Lush. The genre is "
                "notable for its dreamy, atmospheric quality.",
                {
                    "source": "curated",
                    "type": "genre_guide",
                    "artist": "",
                    "tags": "shoegaze, dream pop, atmospheric, indie, distortion",
                },
            ),
            (
                "Daft Punk's 'Homework' (1997) is a landmark in house and "
                "electronic dance music. Tracks like 'Da Funk' and 'Around the "
                "World' showcase the duo's ability to craft high-energy, "
                "groove-driven loops that are almost impossible to stand still "
                "to. The album's relentless four-on-the-floor kick and filtered "
                "basslines defined the French house sound and remain among the "
                "most dancefloor-effective records ever made.",
                {
                    "source": "curated",
                    "type": "review",
                    "artist": "Daft Punk",
                    "tags": "electronic, house, dance, high energy, French house",
                },
            ),
        ]

        print("Adding 3 documents…")
        for text, meta in docs:
            doc_id = kb.add_document(text, meta)
            print(f"  added {doc_id[:12]}… ({meta['type']})")

        stats = kb.get_stats()
        print(f"\nCollection stats: {stats}\n")

        # --- 2. Query: dreamy atmospheric ---
        print("=" * 60)
        print('Query: "dreamy atmospheric music"')
        print("=" * 60)
        results = kb.retrieve("dreamy atmospheric music", n_results=3)
        for i, r in enumerate(results, 1):
            dist = r["distance_score"]
            sim = 1 / (1 + dist)
            typ = r["metadata"].get("type", "?")
            src = r["metadata"].get("artist") or r["metadata"].get("type", "?")
            snippet = r["text"][:100].replace("\n", " ")
            print(f"  {i}. [{typ}] {src}  dist={dist:.3f}  sim={sim:.3f}")
            print(f"     {snippet}…")

        print()

        # --- 3. Query: high energy dance ---
        print("=" * 60)
        print('Query: "high energy dance music"')
        print("=" * 60)
        results = kb.retrieve("high energy dance music", n_results=3)
        for i, r in enumerate(results, 1):
            dist = r["distance_score"]
            sim = 1 / (1 + dist)
            typ = r["metadata"].get("type", "?")
            src = r["metadata"].get("artist") or r["metadata"].get("type", "?")
            snippet = r["text"][:100].replace("\n", " ")
            print(f"  {i}. [{typ}] {src}  dist={dist:.3f}  sim={sim:.3f}")
            print(f"     {snippet}…")

        print()
        print(
            "Results are ranked by semantic similarity — the Daft Punk review\n"
            "should rank #1 for 'high energy dance' even though that exact\n"
            "phrase doesn't appear verbatim in the text."
        )
