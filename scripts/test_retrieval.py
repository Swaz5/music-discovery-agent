"""
Manual retrieval quality evaluation.

Runs 15 diverse queries through MusicRetriever and prints ranked results
with distance scores so you can judge whether retrieval is working well.

Run from project root:
    python scripts/test_retrieval.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rag.knowledge_base import MusicKnowledgeBase
from src.rag.retriever import MusicRetriever

# ── ANSI colours ────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
MAGENTA = "\033[35m"
RED     = "\033[31m"
BG_DARK = "\033[48;5;235m"

def _c(*codes): return lambda t: "".join(codes) + str(t) + RESET

cyan    = _c(CYAN)
yellow  = _c(YELLOW)
green   = _c(GREEN)
magenta = _c(MAGENTA)
dim     = _c(DIM)
bold    = _c(BOLD)
red     = _c(RED)

W = 70

def section(title: str) -> None:
    pad = W - len(title) - 2
    print()
    print(_c(BOLD, CYAN)(f"{'─' * (pad // 2)} {title} {'─' * (pad - pad // 2)}"))


def dist_colour(d: float) -> str:
    if d < 1.2:   return green(f"{d:.3f}")
    if d < 1.5:   return yellow(f"{d:.3f}")
    return red(f"{d:.3f}")


# ── Queries ──────────────────────────────────────────────────────────────────

VIBE_QUERIES = [
    "energetic indie rock with female vocals",
    "music for studying late at night",
    "something that sounds like being underwater",
    "dark electronic with heavy bass",
    "happy upbeat summer music",
    "experimental noise but still melodic",
    "jazz but modern and electronic",
    "sad songs that are still beautiful",
    "aggressive but not metal",
    "music my parents would hate",
    # five more
    "cinematic and orchestral but not classical",
    "music that sounds like it was recorded in a church or cathedral",
    "slow burning and hypnotic with a groove",
    "bedroom producer vibes — intimate and lo-fi",
    "art school pretentious but actually good",
]

ARTIST_QUERIES = [
    "Radiohead",
    "Frank Ocean",
    "Boards of Canada",
]

GENRE_QUERIES = [
    "shoegaze",
    "trip-hop",
    "neo-soul",
]

MIXED_QUERIES = [
    "I want something that feels like early morning before anyone else is awake",
    "music that is simultaneously beautiful and deeply unsettling",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def print_result(rank: int, r: dict) -> None:
    meta = r["metadata"]
    doc_type = meta.get("type", "?")
    artist   = meta.get("artist", "")
    tags     = meta.get("tags", "")
    dist     = r["distance_score"]

    label_parts = [magenta(doc_type)]
    if artist:
        label_parts.append(cyan(artist))
    if tags:
        label_parts.append(dim(tags[:50]))

    label = "  ".join(label_parts)
    text_preview = r["text"][:160].replace("\n", " ").strip()
    if len(r["text"]) > 160:
        text_preview += "…"

    print(f"  {bold(rank)}.  {label}")
    print(f"      dist={dist_colour(dist)}")
    print(f"      {dim(text_preview)}")


def run_section(title: str, queries: list[str], method_name: str, retriever: MusicRetriever) -> dict:
    section(title)
    timings = []
    for q in queries:
        print(f"\n  {yellow('▶')} {bold(q)}")
        t0 = time.monotonic()
        method = getattr(retriever, method_name)
        # retrieve_* methods return a formatted string; we need raw results
        # so call the underlying KB directly for display, then show the string
        raw_results = retriever._kb.retrieve(q, n_results=3)
        elapsed = time.monotonic() - t0
        timings.append(elapsed)
        for i, r in enumerate(raw_results[:3], 1):
            print_result(i, r)
        print(f"      {dim(f'({elapsed*1000:.0f} ms)')}")
    return {"count": len(queries), "total_ms": sum(timings) * 1000}


def run_method_section(
    title: str,
    queries: list[str],
    method_name: str,
    retriever: MusicRetriever,
    n: int = 3,
) -> None:
    """Show the full formatted output from a retriever method, truncated for readability."""
    section(title)
    for q in queries:
        print(f"\n  {yellow('▶')} {bold(q)}")
        method = getattr(retriever, method_name)
        output = method(q, n=n)
        # Print first 600 chars of each formatted block so it doesn't scroll forever
        preview = output[:600]
        if len(output) > 600:
            preview += f"\n{dim('  … [truncated] …')}"
        for line in preview.splitlines():
            print(f"    {line}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(_c(BOLD, CYAN)(f"{'═' * W}"))
    print(_c(BOLD, CYAN)(f"  MUSIC RETRIEVER — QUALITY EVALUATION  ({len(VIBE_QUERIES) + len(ARTIST_QUERIES) + len(GENRE_QUERIES) + len(MIXED_QUERIES)} queries)"))
    print(_c(BOLD, CYAN)(f"{'═' * W}"))

    kb = MusicKnowledgeBase()
    stats = kb.get_stats()
    print(f"\n  KB: {bold(stats['document_count'])} documents  "
          f"| sources: {dim(str(stats['sources']))}  "
          f"| types: {dim(str(stats['types']))}")

    retriever = MusicRetriever(kb=kb)

    # ── 1. Vibe queries: show raw ranked results ──────────────────────────
    section("VIBE QUERIES  (retrieve_for_vibe — raw ranked results)")
    all_timings = []

    for q in VIBE_QUERIES:
        print(f"\n  {yellow('▶')} {bold(q)}")
        t0 = time.monotonic()
        raw = kb.retrieve(q, n_results=3)
        elapsed = time.monotonic() - t0
        all_timings.append(elapsed)
        for i, r in enumerate(raw, 1):
            print_result(i, r)
        # Also note if the formatted output looks reasonable
        formatted = retriever.retrieve_for_vibe(q, n=3)
        block_lines = formatted.count("---")
        print(f"      {dim(f'({elapsed*1000:.0f} ms · {block_lines} formatted blocks)')}")

    avg_ms = (sum(all_timings) / len(all_timings)) * 1000
    print(f"\n  {dim(f'Avg query time: {avg_ms:.0f} ms')}")

    # ── 2. Artist queries: show formatted output ──────────────────────────
    run_method_section(
        "ARTIST QUERIES  (retrieve_for_artist — formatted prompt context)",
        ARTIST_QUERIES,
        "retrieve_for_artist",
        retriever,
        n=3,
    )

    # ── 3. Genre queries: show formatted output ───────────────────────────
    run_method_section(
        "GENRE QUERIES  (retrieve_for_genre — formatted prompt context)",
        GENRE_QUERIES,
        "retrieve_for_genre",
        retriever,
        n=3,
    )

    # ── 4. Mixed queries: show formatted output ───────────────────────────
    run_method_section(
        "MIXED QUERIES  (retrieve_mixed — diversity + dedup)",
        MIXED_QUERIES,
        "retrieve_mixed",
        retriever,
        n=5,
    )

    # ── 5. Distance distribution sanity check ────────────────────────────
    section("DISTANCE DISTRIBUTION SANITY CHECK")
    print()
    # (query, check_fn, description_of_check)
    test_cases = [
        (
            "shoegaze ethereal guitar walls",
            lambda results: results and "shoegaze" in results[0]["metadata"].get("tags", ""),
            "top result tagged 'shoegaze'",
        ),
        (
            "hip hop boom bap beats",
            lambda results: results and results[0]["distance_score"] < 1.0,
            "strong match (dist < 1.0) for hip-hop query",
        ),
        (
            "completely unrelated: astrophysics",
            lambda results: not results or results[0]["distance_score"] > 1.4,
            "weak match (dist > 1.4) for off-topic query",
        ),
    ]
    for query, check_fn, description in test_cases:
        results = kb.retrieve(query, n_results=5, where_filter={"type": "genre_guide"})
        passed = check_fn(results)
        mark = green("✓") if passed else red("✗")
        top_tags = results[0]["metadata"].get("tags", "—") if results else "—"
        top_dist = results[0]["distance_score"] if results else 0
        print(f"  {mark} {bold(query[:40]):<42} {dim(description)}")
        print(f"      → {cyan(top_tags[:35])}  dist={dist_colour(top_dist)}")

    print()
    print(_c(BOLD, CYAN)(f"{'═' * W}"))
    print(_c(BOLD, CYAN)("  Done. Review results above — look for:"))
    print(dim("    • Top result clearly relevant to query"))
    print(dim("    • Distance scores below 1.5 for good matches"))
    print(dim("    • Diversity across doc types in mixed results"))
    print(dim("    • No duplicate text blocks in formatted output"))
    print(_c(BOLD, CYAN)(f"{'═' * W}"))
    print()


if __name__ == "__main__":
    main()
