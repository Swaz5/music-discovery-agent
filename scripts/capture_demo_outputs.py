"""
Capture demo outputs for the README.

Runs 3 queries through the full music discovery system and saves:
  - Raw JSON in logs/demo_outputs/
  - Nicely formatted markdown in logs/demo_outputs/ (ready to paste into README)
  - Combined readme_demo.md with all three queries

Also captures:
  - Project stats (Python lines, modules, tools, endpoints)
  - pytest summary (test count + pass/fail)

Usage:
    python scripts/capture_demo_outputs.py
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `src` is importable regardless of
# how the script is invoked (e.g. `python scripts/capture_demo_outputs.py`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force UTF-8 stdout/stderr so Unicode characters render correctly on Windows.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import httpx
from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

OUTPUT_DIR = Path("logs/demo_outputs")

# ── Project stats ──────────────────────────────────────────────────────────────


def collect_project_stats() -> dict:
    """Count Python files, lines of code, modules, agent tools, API endpoints."""
    src_files = sorted(Path("src").rglob("*.py"))
    test_files = sorted(Path("tests").rglob("*.py"))

    src_lines = sum(
        len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        for f in src_files
    )

    # Modules = non-__init__ source files
    modules = [f for f in src_files if f.name != "__init__.py"]

    # Agent tools from the registry
    from src.agent.tools import TOOLS_REGISTRY  # noqa: PLC0415

    n_tools = len(TOOLS_REGISTRY)
    tool_names = list(TOOLS_REGISTRY.keys())

    # API endpoints: count decorators in routes.py + root in main.py
    routes_text = Path("src/api/routes.py").read_text(encoding="utf-8")
    n_endpoints = len(re.findall(r"@router\.(get|post|put|delete|patch)", routes_text))
    n_endpoints += 1  # root GET / in main.py

    return {
        "python_files": len(src_files),
        "test_files": len(test_files),
        "src_lines": src_lines,
        "modules": len(modules),
        "agent_tools": n_tools,
        "tool_names": tool_names,
        "api_endpoints": n_endpoints,
    }


def capture_test_summary() -> str:
    """Run pytest --tb=no -q and return the summary line."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--tb=no", "-q", "--timeout=60"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = (result.stdout + result.stderr).strip()
    lines = output.splitlines()
    # The summary line contains "passed", "failed", or "error"
    for line in reversed(lines):
        if any(kw in line for kw in ("passed", "failed", "error")):
            return line.strip()
    return lines[-1].strip() if lines else "unknown"


# ── Live queries ───────────────────────────────────────────────────────────────


async def run_queries() -> dict:
    """Drive all three queries against the full FastAPI app via ASGI transport."""
    from src.api.main import app  # noqa: PLC0415

    transport = httpx.ASGITransport(app=app)
    results: dict = {}

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=300.0
    ) as client:

        # ── Query 1: Discovery ─────────────────────────────────────────────────
        print("  [1/3] Discovery - 'driving through a city at night'...")
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/discover",
                json={
                    "query": "Something that sounds like driving through a city at night",
                    "include_reasoning": True,
                },
            )
            resp.raise_for_status()
            results["discover"] = resp.json()
            results["discover"]["_elapsed"] = round(time.perf_counter() - t0, 1)
            n = len(results["discover"].get("recommendations", []))
            iters = results["discover"].get("iterations", "?")
            tokens = results["discover"].get("total_tokens", 0)
            print(
                f"        OK {n} recommendations  |  {iters} iterations  "
                f"|  {tokens:,} tokens  |  {results['discover']['_elapsed']}s"
            )
        except Exception as exc:  # noqa: BLE001
            results["discover"] = {"error": str(exc)}
            print(f"        FAIL: {exc}")

        # ── Query 2: Taste analysis ────────────────────────────────────────────
        print("  [2/3] Taste analysis - Radiohead + Tame Impala...")
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/analyze-taste",
                json={
                    "artists": ["Radiohead", "Tame Impala"],
                    "discover_blind_spots": True,
                },
            )
            resp.raise_for_status()
            results["taste"] = resp.json()
            results["taste"]["_elapsed"] = round(time.perf_counter() - t0, 1)
            identity = results["taste"].get("taste_identity", "?")
            n_bs = len(results["taste"].get("blind_spots", []))
            print(
                f"        OK identity: \"{identity}\"  |  {n_bs} blind spots"
                f"  |  {results['taste']['_elapsed']}s"
            )
        except Exception as exc:  # noqa: BLE001
            results["taste"] = {"error": str(exc)}
            print(f"        FAIL: {exc}")

        # ── Query 3: Genre bridge ──────────────────────────────────────────────
        print("  [3/3] Genre bridge - jazz to electronic...")
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/bridge",
                json={"genre_a": "jazz", "genre_b": "electronic", "max_hops": 3},
            )
            resp.raise_for_status()
            results["bridge"] = resp.json()
            results["bridge"]["_elapsed"] = round(time.perf_counter() - t0, 1)
            n_ba = len(results["bridge"].get("bridge_artists", []))
            n_pl = len(results["bridge"].get("transition_playlist", []))
            print(
                f"        OK {n_ba} bridge artists  |  {n_pl}-track playlist"
                f"  |  {results['bridge']['_elapsed']}s"
            )
        except Exception as exc:  # noqa: BLE001
            results["bridge"] = {"error": str(exc)}
            print(f"        FAIL: {exc}")

    return results


# ── Markdown formatters ────────────────────────────────────────────────────────


def _safe(s: object, limit: int = 0) -> str:
    """Stringify and optionally truncate, escaping pipe characters for tables."""
    text = str(s).replace("|", "\\|")
    return text[:limit] + ("…" if limit and len(text) > limit else "") if limit else text


def fmt_discover_md(data: dict) -> str:
    if "error" in data:
        return f"## Query 1: Discovery\n\n**Error:** {data['error']}\n"

    query = "Something that sounds like driving through a city at night"
    recs: list[dict] = data.get("recommendations", [])
    trace: list[dict] = data.get("reasoning_trace") or []
    iters = data.get("iterations", "?")
    tokens = data.get("total_tokens", 0)
    elapsed = data.get("_elapsed", "?")

    lines = [
        "## Query 1: Discovery",
        "",
        f'> **"{query}"**',
        "",
        f"*{iters} agent iterations · {tokens:,} tokens · {elapsed}s*",
        "",
    ]

    if trace:
        lines += ["### Reasoning Trace", ""]
        for i, step in enumerate(trace, 1):
            tool = step.get("tool", "?")
            args = step.get("arguments", {})
            preview = step.get("result_preview") or step.get("error", "")
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            lines.append(f"{i}. **`{tool}({args_str})`**")
            if preview:
                short = preview[:220].replace("\n", " ")
                if len(preview) > 220:
                    short += "…"
                lines.append(f"   > {short}")
            lines.append("")

    lines += [
        "### Recommendations",
        "",
        "| # | Artist | Track | Score | Why |",
        "|---|--------|-------|-------|-----|",
    ]
    for i, r in enumerate(recs, 1):
        score = f"{r.get('vibe_match_score', 0):.2f}"
        why = _safe(r.get("why", ""), 90)
        artist = r.get("artist", "")
        track = r.get("track", "")
        lines.append(f"| {i} | **{artist}** | {track} | {score} | {why} |")

    lines += ["", "#### Vibe Descriptions", ""]
    for r in recs:
        lines.append(f"- **{r.get('artist')}** — {r.get('vibe_description', '')}")

    deezer_links = [(r.get("artist"), r.get("track"), r.get("deezer_url")) for r in recs if r.get("deezer_url")]
    if deezer_links:
        lines += ["", "#### Listen on Deezer", ""]
        for artist, track, url in deezer_links:
            lines.append(f"- [{artist} — {track}]({url})")

    return "\n".join(lines)


def fmt_taste_md(data: dict) -> str:
    if "error" in data:
        return f"## Query 2: Taste Analysis\n\n**Error:** {data['error']}\n"

    elapsed = data.get("_elapsed", "?")
    identity = data.get("taste_identity", "")
    analysis = data.get("analysis", "")
    threads: list[str] = data.get("common_threads", [])
    audio: dict = data.get("audio_profile", {})
    blind_spots: list[dict] = data.get("blind_spots", [])
    outlier: dict | None = data.get("outlier")

    lines = [
        "## Query 2: Taste Analysis",
        "",
        '> **"I love Radiohead and Tame Impala — analyze my taste and find my blind spots"**',
        "",
        f"*{elapsed}s*",
        "",
        f"### Taste Identity: *{identity}*",
        "",
        "### Analysis",
        "",
        analysis,
        "",
        "### Common Threads",
        "",
    ]
    for t in threads:
        lines.append(f"- {t}")

    if audio:
        lines += ["", "### Audio Profile", "", "| Feature | Value |", "|---------|-------|"]
        for k, v in audio.items():
            label = k.replace("_", " ").title()
            fmt_v = f"{v:.0f} BPM" if k == "tempo" else f"{float(v):.3f}"
            lines.append(f"| {label} | {fmt_v} |")

    if blind_spots:
        lines += [
            "",
            "### Blind Spots",
            "",
            "| Genre | Why You'd Love It | Try This |",
            "|-------|-------------------|----------|",
        ]
        for bs in blind_spots:
            genre = bs.get("genre", "")
            why = _safe(bs.get("why", ""), 100)
            try_this = bs.get("try_this", "")
            lines.append(f"| **{genre}** | {why} | *{try_this}* |")

    if outlier:
        lines += [
            "",
            "### Outlier Artist",
            "",
            f"**{outlier.get('artist', '')}** — {outlier.get('why_different', '')}",
        ]

    return "\n".join(lines)


def fmt_bridge_md(data: dict) -> str:
    if "error" in data:
        return f"## Query 3: Genre Bridge\n\n**Error:** {data['error']}\n"

    genre_a = data.get("genre_a", "jazz")
    genre_b = data.get("genre_b", "electronic")
    elapsed = data.get("_elapsed", "?")
    bridge_artists: list[dict] = data.get("bridge_artists", [])
    playlist: list[dict] = data.get("transition_playlist", [])
    explanation = data.get("explanation", "")

    lines = [
        "## Query 3: Genre Bridge",
        "",
        '> **"Bridge me from jazz to electronic music"**',
        "",
        f"*{genre_a.title()} → {genre_b.title()} · {elapsed}s*",
        "",
        "### Bridge Artists",
        "",
        "| Artist | Genres | Why They Bridge the Gap |",
        "|--------|--------|------------------------|",
    ]
    for ba in bridge_artists:
        name = ba.get("name", "")
        genres = ", ".join(ba.get("genres", []))
        reason = _safe(ba.get("connects_because", ""), 130)
        lines.append(f"| **{name}** | {genres} | {reason} |")

    lines += [
        "",
        "### Transition Playlist",
        "",
        "| # | Track | Artist | Position in Journey |",
        "|---|-------|--------|---------------------|",
    ]
    for i, t in enumerate(playlist, 1):
        track = t.get("track", "")
        artist = t.get("artist", "")
        pos = t.get("position", "")
        lines.append(f"| {i} | {track} | {artist} | *{pos}* |")

    lines += ["", "### The Connection", "", explanation]

    return "\n".join(lines)


def fmt_stats_md(stats: dict, test_summary: str) -> str:
    tool_rows = "\n".join(
        f"| `{name}` | |" for name in stats.get("tool_names", [])
    )

    # Tool descriptions (hardcoded — matches tools.py)
    tool_desc = {
        "search_artist": "Look up an artist's profile, tags, and social stats from Last.fm",
        "get_similar_artists": "Find sonically related artists via the Last.fm similarity graph",
        "search_by_tag": "Get top artists for a genre or mood tag",
        "get_audio_features": "Quantified audio analysis — energy, valence, tempo, danceability…",
        "search_knowledge_base": "Semantic search over curated genre guides and artist bios (RAG)",
        "search_tracks": "Search Deezer for specific tracks with preview URLs",
    }

    lines = [
        "## Project Stats",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Python source files | {stats['python_files']} |",
        f"| Lines of Python (src/) | {stats['src_lines']:,} |",
        f"| Source modules | {stats['modules']} |",
        f"| Agent tools | {stats['agent_tools']} |",
        f"| API endpoints | {stats['api_endpoints']} |",
        f"| Test files | {stats['test_files']} |",
        f"| Tests | `{test_summary}` |",
        "",
        "### API Endpoints",
        "",
        "| Method | Path | Description |",
        "|--------|------|-------------|",
        "| `GET`  | `/` | Health check — service liveness |",
        "| `POST` | `/discover` | Run the discovery agent (natural-language vibe → curated recs) |",
        "| `GET`  | `/artist/{name}` | Full artist profile with bio, tags, audio features, similar artists |",
        "| `POST` | `/explore` | Genre exploration with RAG knowledge-base context |",
        "| `POST` | `/rate` | Save a like/dislike preference to shape future recommendations |",
        "| `GET`  | `/taste-profile` | View the current personal taste profile |",
        "| `GET`  | `/history` | Session recommendation history |",
        "| `POST` | `/analyze-taste` | Taste analysis + blind-spot discovery via Claude |",
        "| `POST` | `/bridge` | Genre bridge: BFS similarity graph + Claude explanation |",
        "",
        "### Agent Tools",
        "",
        "| Tool | Description |",
        "|------|-------------|",
    ]
    for name in stats.get("tool_names", []):
        desc = tool_desc.get(name, "")
        lines.append(f"| `{name}` | {desc} |")

    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("=" * 58)
    print("  Music Discovery Agent — Demo Output Capture")
    print("=" * 58)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stats ──────────────────────────────────────────────────────────────────
    print("Collecting project stats…")
    stats = collect_project_stats()
    print(
        f"  {stats['python_files']} Python files  |  {stats['src_lines']:,} lines  "
        f"|  {stats['modules']} modules  |  {stats['agent_tools']} tools  "
        f"|  {stats['api_endpoints']} endpoints"
    )

    print("Running pytest…")
    test_summary = capture_test_summary()
    print(f"  {test_summary}")

    # ── Queries ────────────────────────────────────────────────────────────────
    print("\nRunning queries against the live system:")
    results = asyncio.run(run_queries())

    # ── Save raw JSON ──────────────────────────────────────────────────────────
    print("\nSaving outputs…")
    (OUTPUT_DIR / "discover_raw.json").write_text(
        json.dumps(results.get("discover", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "taste_raw.json").write_text(
        json.dumps(results.get("taste", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "bridge_raw.json").write_text(
        json.dumps(results.get("bridge", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "stats.json").write_text(
        json.dumps({"stats": stats, "test_summary": test_summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Save formatted markdown ────────────────────────────────────────────────
    md_stats = fmt_stats_md(stats, test_summary)
    md_discover = fmt_discover_md(results.get("discover", {"error": "not run"}))
    md_taste = fmt_taste_md(results.get("taste", {"error": "not run"}))
    md_bridge = fmt_bridge_md(results.get("bridge", {"error": "not run"}))

    (OUTPUT_DIR / "stats.md").write_text(md_stats, encoding="utf-8")
    (OUTPUT_DIR / "discover.md").write_text(md_discover, encoding="utf-8")
    (OUTPUT_DIR / "taste.md").write_text(md_taste, encoding="utf-8")
    (OUTPUT_DIR / "bridge.md").write_text(md_bridge, encoding="utf-8")

    # Combined file ready to paste into README
    readme_demo = "\n\n---\n\n".join([md_stats, md_discover, md_taste, md_bridge])
    (OUTPUT_DIR / "readme_demo.md").write_text(readme_demo, encoding="utf-8")

    # ── Final summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("  Summary")
    print("=" * 58)
    print()

    discover = results.get("discover", {})
    taste = results.get("taste", {})
    bridge = results.get("bridge", {})

    print("Project:")
    print(f"  {stats['python_files']} Python files · {stats['src_lines']:,} lines of source code")
    print(f"  {stats['modules']} modules · {stats['agent_tools']} agent tools · {stats['api_endpoints']} API endpoints")
    print(f"  Tests: {test_summary}")
    print()

    print("Query results:")
    if "error" not in discover:
        n_recs = len(discover.get("recommendations", []))
        print(
            f"  [1] Discovery        — {n_recs} recommendations  "
            f"|  {discover.get('iterations', '?')} iterations  "
            f"|  {discover.get('total_tokens', 0):,} tokens  "
            f"|  {discover.get('_elapsed', '?')}s"
        )
        for r in discover.get("recommendations", [])[:3]:
            print(f"        • {r.get('artist')} — {r.get('track')}  [{r.get('vibe_match_score', 0):.2f}]")
    else:
        print(f"  [1] Discovery        — ERROR: {discover['error']}")

    if "error" not in taste:
        print(
            f"  [2] Taste analysis   — \"{taste.get('taste_identity', '?')}\"  "
            f"|  {len(taste.get('blind_spots', []))} blind spots  "
            f"|  {taste.get('_elapsed', '?')}s"
        )
    else:
        print(f"  [2] Taste analysis   — ERROR: {taste['error']}")

    if "error" not in bridge:
        print(
            f"  [3] Genre bridge     — {len(bridge.get('bridge_artists', []))} bridge artists  "
            f"|  {len(bridge.get('transition_playlist', []))}-track playlist  "
            f"|  {bridge.get('_elapsed', '?')}s"
        )
    else:
        print(f"  [3] Genre bridge     — ERROR: {bridge['error']}")

    print()
    print(f"Files saved to {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<35} {size:>8,} bytes")

    print()
    print(f"README-ready: {OUTPUT_DIR}/readme_demo.md")
    print()


if __name__ == "__main__":
    main()
