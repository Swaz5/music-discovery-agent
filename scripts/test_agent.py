"""
Comprehensive agent test script.

Runs the MusicDiscoveryAgent against 8 diverse queries that exercise
different reasoning capabilities, prints a detailed trace for each run,
and saves the full structured output to logs/agent_test_results.json.

Usage (from the project root)::

    python scripts/test_agent.py

Output::

    • Per-query: tool calls + args, iteration count, recommendations, tokens
    • Final: summary table across all queries
    • File: logs/agent_test_results.json
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Makes ``src.*`` importable when invoked as a standalone script from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on Windows terminals so box-drawing chars and em-dashes render.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.agent.discovery_agent import MusicDiscoveryAgent  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

QUERIES = [
    {
        "id": 1,
        "query": "Something that sounds like Radiohead but more upbeat",
        "tests": "artist similarity + audio feature filtering",
    },
    {
        "id": 2,
        "query": "Music for a rainy Sunday morning",
        "tests": "mood interpretation + vibe matching",
    },
    {
        "id": 3,
        "query": "I want to discover Japanese jazz fusion",
        "tests": "specific genre + cultural context from knowledge base",
    },
    {
        "id": 4,
        "query": "Give me something that blends electronic and folk",
        "tests": "cross-genre bridging",
    },
    {
        "id": 5,
        "query": "Music that feels like being alone in a big city at 3am",
        "tests": "abstract/poetic vibe interpretation",
    },
    {
        "id": 6,
        "query": "I only listen to metal but want to try something completely different",
        "tests": "genre-breaking recommendations",
    },
    {
        "id": 7,
        "query": "Happy music that isn't cheesy pop",
        "tests": "positive vibe with negative constraint",
    },
    {
        "id": 8,
        "query": "Something with complex rhythms but still groovy",
        "tests": "contradicting audio features — high complexity + high danceability",
    },
]

# ── Formatting helpers ────────────────────────────────────────────────────────

_W = 72  # console width


def _rule(char: str = "─") -> str:
    return char * _W


def _header(text: str, char: str = "═") -> None:
    print(f"\n{char * _W}")
    print(f"  {text}")
    print(char * _W)


def _section(text: str) -> None:
    print(f"\n  ── {text} {'─' * max(0, _W - len(text) - 6)}")


def _print_query_result(entry: dict) -> None:
    """Pretty-print a single query result to stdout."""
    q = entry["query"]
    r = entry["result"]
    elapsed = entry["elapsed_seconds"]
    tests = entry["tests"]

    _header(f"Query {entry['id']}: {q}")
    print(f"  Tests: {tests}")
    print(f"  Elapsed: {elapsed:.1f}s  |  "
          f"Iterations: {r['iterations']}  |  "
          f"Tokens: {r['total_tokens']:,}")

    # ── Tool calls ─────────────────────────────────────────────────────────
    _section("Tool calls")
    if r["reasoning_trace"]:
        for step in r["reasoning_trace"]:
            iter_tag = f"[{step['iteration']}]"
            args_str = json.dumps(step.get("arguments", {}), ensure_ascii=False)
            # Truncate long arg strings for readability
            if len(args_str) > 60:
                args_str = args_str[:57] + "…"

            if "error" in step:
                status_icon = "✗"
                preview = f"ERROR: {step['error'][:80]}"
            else:
                status_icon = "✓"
                preview = step.get("result_preview", "")[:100]

            print(f"    {iter_tag} {status_icon} {step['tool']}({args_str})")
            print(f"         → {preview}")
    else:
        print("    (no tool calls recorded)")

    # ── Recommendations ────────────────────────────────────────────────────
    _section(f"Recommendations ({len(r['recommendations'])} found)")
    if r["recommendations"]:
        for i, rec in enumerate(r["recommendations"], 1):
            artist = rec.get("artist") or "?"
            track = rec.get("track") or "?"
            score = rec.get("vibe_match_score", 0.0)
            tags = ", ".join(rec.get("genre_tags", [])) or "—"
            why = rec.get("why", "")
            vibe_desc = rec.get("vibe_description", "")
            url = rec.get("deezer_url", "")

            print(f"\n    {i}. {artist} — {track}")
            print(f"       Tags:       {tags}")
            print(f"       Vibe match: {score}")
            if why:
                # Wrap at 60 chars with indent
                _print_wrapped(why, prefix="       Why: ", width=_W - 8)
            if vibe_desc:
                _print_wrapped(vibe_desc, prefix="       Vibe: ", width=_W - 8)
            if url:
                print(f"       Deezer: {url}")
    else:
        print("    (no structured recommendations parsed)")


def _print_wrapped(text: str, prefix: str, width: int) -> None:
    """Print text with word-wrapping, indenting continuation lines."""
    indent = " " * len(prefix)
    words = text.split()
    line = prefix
    for word in words:
        if len(line) + len(word) + 1 > width and line != prefix:
            print(line)
            line = indent + word
        else:
            line = (line + " " + word) if line != prefix else line + word
    if line.strip():
        print(line)


def _print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table of all queries."""
    _header("Summary", char="═")
    print(f"\n  {'#':>2}  {'Iters':>5}  {'Tokens':>8}  {'Recs':>4}  "
          f"{'Time':>6}  {'Status':<8}  Query")
    print(f"  {_rule('─')}")

    total_tokens = 0
    total_elapsed = 0.0

    for entry in results:
        r = entry["result"]
        status = "OK" if not entry.get("error") else "ERROR"
        tokens = r.get("total_tokens", 0)
        iters = r.get("iterations", 0)
        recs = len(r.get("recommendations", []))
        elapsed = entry.get("elapsed_seconds", 0.0)
        query_short = entry["query"][:42] + ("…" if len(entry["query"]) > 42 else "")

        total_tokens += tokens
        total_elapsed += elapsed

        print(f"  {entry['id']:>2}  {iters:>5}  {tokens:>8,}  {recs:>4}  "
              f"{elapsed:>5.1f}s  {status:<8}  {query_short}")

    print(f"  {_rule('─')}")
    print(f"  {'':>2}  {'':>5}  {total_tokens:>8,}  {'':>4}  "
          f"{total_elapsed:>5.1f}s  {'TOTAL':<8}")
    print()


# ── Core runner ───────────────────────────────────────────────────────────────

async def run_all_queries() -> list[dict]:
    """Run all 8 queries sequentially and return their full result records."""
    agent = MusicDiscoveryAgent()
    all_results: list[dict] = []

    for spec in QUERIES:
        print(f"\n  → Starting query {spec['id']}/8: {spec['query'][:60]}")
        t0 = time.monotonic()
        error_msg = None

        try:
            result = await agent.discover(spec["query"])
        except Exception as exc:
            logging.exception("Query %d failed: %s", spec["id"], exc)
            error_msg = str(exc)
            result = {
                "recommendations": [],
                "reasoning_trace": [],
                "iterations": 0,
                "total_tokens": 0,
            }

        elapsed = time.monotonic() - t0

        entry = {
            "id": spec["id"],
            "query": spec["query"],
            "tests": spec["tests"],
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }
        if error_msg:
            entry["error"] = error_msg

        all_results.append(entry)
        _print_query_result(entry)

    return all_results


# ── JSON serialiser ───────────────────────────────────────────────────────────

def _serialise(obj):
    """JSON default handler: stringify anything the encoder can't handle."""
    return str(obj)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,          # suppress chatty INFO from the agent
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    _header("Music Discovery Agent — Comprehensive Test Suite", char="█")
    print(f"\n  Running {len(QUERIES)} queries against MusicDiscoveryAgent …")
    print(f"  Results will be saved to logs/agent_test_results.json\n")
    print(_rule())

    all_results = await run_all_queries()

    # ── Summary table ──────────────────────────────────────────────────────
    _print_summary_table(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = LOGS_DIR / "agent_test_results.json"

    payload = {
        "test_run": {
            "query_count": len(QUERIES),
            "total_tokens": sum(e["result"].get("total_tokens", 0) for e in all_results),
            "total_elapsed_seconds": round(sum(e["elapsed_seconds"] for e in all_results), 2),
        },
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_serialise)

    print(f"  Full results saved → {output_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
