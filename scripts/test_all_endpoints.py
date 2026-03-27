"""
Comprehensive API integration test for the Music Discovery Agent.

Spins up the FastAPI server, exercises every endpoint against real API calls
(Last.fm + Anthropic), runs the pytest unit suite, and prints a structured
PASS / FAIL report.

Usage (from the project root)::

    python scripts/test_all_endpoints.py

Options:
    --port PORT     Server port (default 8000)
    --skip-slow     Skip /discover and /analyze-taste (Anthropic calls)
    --no-pytest     Skip the pytest unit suite at the end

Estimated runtime without --skip-slow: 8-15 minutes (Anthropic API calls).
With --skip-slow: 2-3 minutes.

Output files (written to logs/):
    endpoint_test_results.json   structured results for every test case
    openapi_spec.json            the full OpenAPI schema from the running server
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from textwrap import shorten
from typing import Any

import httpx

# ── Path setup ────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Console width ─────────────────────────────────────────────────────────────

_W = 70


def _rule(char: str = "─") -> str:
    return char * _W


def _header(text: str, char: str = "═") -> None:
    print(f"\n{char * _W}")
    print(f"  {text}")
    print(char * _W)


def _sub(text: str) -> None:
    print(f"\n  ▸ {text}")


# ── Test runner ───────────────────────────────────────────────────────────────


class TestRunner:
    """
    Wraps an httpx.Client with PASS/FAIL recording and pretty-printing.

    Each call to self.check(name, condition) records a result and prints
    a coloured PASS / FAIL line.  Use self.section() to group related checks.
    """

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)
        self._results: list[dict] = []
        self._current_section = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.client.close()

    # ── Recording ─────────────────────────────────────────────────────────────

    def section(self, name: str) -> None:
        self._current_section = name
        print(f"\n{_rule()}")
        print(f"  {name}")
        print(_rule())

    def check(
        self,
        name: str,
        passed: bool,
        detail: str = "",
        response: httpx.Response | None = None,
    ) -> bool:
        icon = "✓" if passed else "✗"
        label = "PASS" if passed else "FAIL"
        # Truncate long names so the line stays within _W
        display = shorten(name, width=55, placeholder="…")
        print(f"  {icon} {label}  {display}")
        if not passed:
            if detail:
                for line in detail.splitlines()[:4]:
                    print(f"           {line}")
            if response is not None:
                snippet = shorten(response.text, width=200, placeholder="…")
                print(f"           HTTP {response.status_code}: {snippet}")
        self._results.append({
            "section": self._current_section,
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        return passed

    def exception(self, name: str, exc: Exception) -> bool:
        return self.check(name, False, f"{type(exc).__name__}: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self._results if r["passed"])

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self._results if not r["passed"])

    def print_summary(self) -> None:
        _header("SUMMARY", "═")

        # Per-section tallies
        sections: dict[str, dict[str, int]] = {}
        for r in self._results:
            s = r["section"]
            if s not in sections:
                sections[s] = {"pass": 0, "fail": 0}
            if r["passed"]:
                sections[s]["pass"] += 1
            else:
                sections[s]["fail"] += 1

        col_w = max(len(s) for s in sections) + 2 if sections else 20
        print(f"  {'Section':<{col_w}}  {'PASS':>4}  {'FAIL':>4}  {'Total':>5}")
        print(f"  {'-' * col_w}  {'-' * 4}  {'-' * 4}  {'-' * 5}")
        for section, counts in sections.items():
            total = counts["pass"] + counts["fail"]
            print(
                f"  {section:<{col_w}}  {counts['pass']:>4}  "
                f"{counts['fail']:>4}  {total:>5}"
            )

        print(f"  {_rule()}")
        total = self.total_passed + self.total_failed
        print(
            f"  {'TOTAL':<{col_w}}  {self.total_passed:>4}  "
            f"{self.total_failed:>4}  {total:>5}"
        )

        if self.total_failed == 0:
            print(f"\n  All {total} tests passed.")
        else:
            failed_names = [r["name"] for r in self._results if not r["passed"]]
            print(f"\n  {self.total_failed} test(s) failed:")
            for name in failed_names:
                print(f"    • {name}")

    def save_results(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Results saved → {path}")


# ── Server lifecycle ──────────────────────────────────────────────────────────


def start_server(port: int) -> subprocess.Popen:
    """Start uvicorn in a subprocess and return the process handle."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--port", str(port),
        "--log-level", "warning",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for_server(base_url: str, timeout: int = 40) -> bool:
    """Poll the health endpoint until the server responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/", timeout=2.0)
            if r.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Test sections ─────────────────────────────────────────────────────────────


def test_discover(t: TestRunner, skip_slow: bool) -> None:
    t.section("1 · POST /discover")

    if skip_slow:
        print("  (skipped — use without --skip-slow to run Anthropic calls)")
        return

    cases = [
        (
            "energetic indie rock for a road trip",
            200,
            "normal case",
        ),
        (
            "asdfghjkl",
            200,
            "gibberish — agent should handle gracefully",
        ),
        (
            "",
            422,
            "empty string — Pydantic validation error",
        ),
        (
            "music that sounds like the color purple",
            200,
            "abstract/synesthetic — tests creativity",
        ),
    ]

    for query, expected_status, label in cases:
        display = f'"{shorten(query, 35, placeholder="…")}"  ({label})'
        _sub(f"Testing {display}")
        t0 = time.time()
        try:
            r = t.client.post("/discover", json={"query": query})
            elapsed = time.time() - t0

            if expected_status == 422:
                t.check(
                    display,
                    r.status_code == 422,
                    f"expected 422, got {r.status_code}",
                    r,
                )
                continue

            if r.status_code != 200:
                t.check(display, False, f"HTTP {r.status_code}", r)
                continue

            body = r.json()
            recs = body.get("recommendations", [])
            iters = body.get("iterations", 0)
            tokens = body.get("total_tokens", 0)

            passed = (
                isinstance(recs, list)
                and isinstance(iters, int)
                and iters > 0
                and isinstance(tokens, int)
            )
            detail = (
                f"{len(recs)} recommendation(s), "
                f"{iters} iteration(s), "
                f"{tokens:,} tokens, "
                f"{elapsed:.1f}s"
            )
            t.check(display, passed, detail if not passed else "", r if not passed else None)

            if recs:
                top = recs[0]
                print(f"           Top pick: {top.get('artist')} — {top.get('track')}")

        except httpx.TimeoutException:
            t.check(display, False, "Request timed out (>180s)")
        except Exception as exc:
            t.exception(display, exc)


def test_artist(t: TestRunner) -> None:
    t.section("2 · GET /artist/{name}")

    cases = [
        ("Radiohead", 200, "well-known artist"),
        ("Totally Fake Band Name 12345", 404, "not found — all data fields empty"),
        # björk: Last.fm may return the artist (200) or report not found (404).
        # Either is correct.  A 5xx means the server crashed on the special char,
        # which is the only failure we care about preventing.
        ("björk", None, "special characters — 200 or 404 accepted, not 5xx"),
    ]

    for name, expected_status, label in cases:
        display = f'"{name}"  ({label})'
        _sub(f"Testing {display}")
        try:
            r = t.client.get(f"/artist/{name}")

            if expected_status is None:
                # Special char case: any non-5xx is a pass
                passed = r.status_code < 500
                t.check(
                    display,
                    passed,
                    f"HTTP {r.status_code} (any non-5xx accepted)",
                    r if not passed else None,
                )
            else:
                passed = r.status_code == expected_status
                detail = f"expected {expected_status}, got {r.status_code}"
                t.check(display, passed, detail if not passed else "", r if not passed else None)

            if r.status_code == 200:
                body = r.json()
                print(f"           Name: {body.get('name')}")
                tags = body.get("tags", [])
                if tags:
                    print(f"           Tags: {', '.join(tags[:5])}")

        except Exception as exc:
            t.exception(display, exc)


def test_analyze_taste(t: TestRunner, skip_slow: bool) -> None:
    t.section("3 · POST /analyze-taste")

    if skip_slow:
        print("  (skipped — use without --skip-slow to run Anthropic calls)")
        return

    cases = [
        (
            ["Radiohead", "Tame Impala", "Beach House", "Aphex Twin", "Bon Iver"],
            "5 diverse artists — standard case",
        ),
        (
            ["Miles Davis"],
            "1 artist — edge case (minimal input)",
        ),
        (
            [
                "The Beatles", "Led Zeppelin", "Pink Floyd", "The Rolling Stones",
                "David Bowie", "Queen", "Jimi Hendrix", "Bob Dylan",
                "Neil Young", "The Doors",
            ],
            "10 artists — larger input set",
        ),
    ]

    for artists, label in cases:
        display = f"{label} ({len(artists)} artist(s))"
        _sub(f"Testing {display}")
        t0 = time.time()
        try:
            r = t.client.post(
                "/analyze-taste",
                json={"artists": artists, "discover_blind_spots": True},
            )
            elapsed = time.time() - t0

            if r.status_code != 200:
                t.check(display, False, f"HTTP {r.status_code}", r)
                continue

            body = r.json()
            identity = body.get("taste_identity", "")
            analysis = body.get("analysis", "")
            threads = body.get("common_threads", [])
            blind_spots = body.get("blind_spots", [])
            audio = body.get("audio_profile", {})

            passed = bool(identity) and bool(analysis) and isinstance(audio, dict)
            detail = (
                f"identity={repr(identity)}, "
                f"{len(threads)} thread(s), "
                f"{len(blind_spots)} blind spot(s), "
                f"{elapsed:.1f}s"
            )
            t.check(display, passed, detail if not passed else "", r if not passed else None)

            if identity:
                print(f"           Identity: {identity}")
            if blind_spots:
                bs = blind_spots[0]
                print(f"           Blind spot #1: {bs.get('genre')} — try {bs.get('try_this')}")

        except httpx.TimeoutException:
            t.check(display, False, "Request timed out (>180s)")
        except Exception as exc:
            t.exception(display, exc)


def test_bridge(t: TestRunner) -> None:
    t.section("4 · POST /bridge")

    cases = [
        ("jazz", "electronic", "clear bridge exists"),
        ("classical", "punk", "very different — harder bridge"),
        ("rock", "rock", "same genre — should return 422"),
    ]

    for genre_a, genre_b, label in cases:
        display = f'"{genre_a}" → "{genre_b}"  ({label})'
        _sub(f"Testing {display}")
        t0 = time.time()
        try:
            r = t.client.post(
                "/bridge",
                json={"genre_a": genre_a, "genre_b": genre_b, "max_hops": 2},
            )
            elapsed = time.time() - t0

            if genre_a == genre_b:
                # Same-genre case: expect 422
                t.check(
                    display,
                    r.status_code == 422,
                    f"expected 422, got {r.status_code}",
                    r if r.status_code != 422 else None,
                )
                continue

            if r.status_code != 200:
                t.check(display, False, f"HTTP {r.status_code}", r)
                continue

            body = r.json()
            bridge_artists = body.get("bridge_artists", [])
            playlist = body.get("transition_playlist", [])
            explanation = body.get("explanation", "")

            passed = (
                len(bridge_artists) >= 1
                and len(playlist) >= 2
                and bool(explanation)
            )
            detail = (
                f"{len(bridge_artists)} bridge artist(s), "
                f"{len(playlist)} track(s), "
                f"{elapsed:.1f}s"
            )
            t.check(display, passed, detail if not passed else "", r if not passed else None)

            if bridge_artists:
                ba = bridge_artists[0]
                print(f"           Bridge: {ba.get('name')} — {ba.get('connects_because','')[:80]}")
            if playlist:
                pl = playlist[0]
                print(f"           Starts: {pl.get('artist')} — {pl.get('track')}  [{pl.get('position')}]")
            if len(playlist) > 1:
                pl = playlist[-1]
                print(f"           Ends:   {pl.get('artist')} — {pl.get('track')}  [{pl.get('position')}]")

        except httpx.TimeoutException:
            t.check(display, False, "Request timed out (>180s)")
        except Exception as exc:
            t.exception(display, exc)


def test_rate_and_profile(t: TestRunner) -> None:
    t.section("5 · POST /rate  +  GET /taste-profile")

    ratings = [
        ("Portishead", True, "love the dark atmosphere"),
        ("Massive Attack", True, "perfect trip-hop"),
        ("Taylor Swift", False, "not my style"),
        ("Brian Eno", True, "ambient perfection"),
        ("Nickelback", False, "too generic"),
    ]

    liked_names = [a for a, liked, _ in ratings if liked]
    disliked_names = [a for a, liked, _ in ratings if not liked]

    _sub("Rating artists…")
    for artist, liked, notes in ratings:
        sentiment = "liked" if liked else "disliked"
        try:
            r = t.client.post(
                "/rate",
                json={"artist": artist, "liked": liked, "notes": notes},
            )
            passed = r.status_code == 200
            name = f'rate "{artist}" ({sentiment})'
            if passed:
                body = r.json()
                t.check(name, passed, body.get("message", ""))
            else:
                t.check(name, False, f"HTTP {r.status_code}", r)
        except Exception as exc:
            t.exception(f'rate "{artist}"', exc)

    _sub("Checking taste profile…")
    try:
        r = t.client.get("/taste-profile")
        t.check(
            "GET /taste-profile returns 200",
            r.status_code == 200,
            response=r if r.status_code != 200 else None,
        )

        if r.status_code == 200:
            profile = r.json()
            liked_count = profile.get("liked_count", 0)
            disliked_count = profile.get("disliked_count", 0)
            pref_tags = profile.get("preferred_tags", [])
            summary = profile.get("summary", "")

            t.check(
                f"liked_count >= {len(liked_names)}",
                liked_count >= len(liked_names),
                f"got liked_count={liked_count}",
            )
            t.check(
                f"disliked_count >= {len(disliked_names)}",
                disliked_count >= len(disliked_names),
                f"got disliked_count={disliked_count}",
            )
            t.check(
                "preferred_tags is populated",
                len(pref_tags) > 0,
                f"got: {pref_tags}",
            )
            t.check(
                "summary is non-empty",
                bool(summary),
                f"got: {repr(summary)}",
            )

            if pref_tags:
                print(f"           Top tags: {', '.join(pref_tags[:5])}")
            if summary:
                print(f"           Summary: {shorten(summary, 80, placeholder='…')}")

    except Exception as exc:
        t.exception("GET /taste-profile", exc)

    _sub("Checking history…")
    try:
        r = t.client.get("/history")
        passed = r.status_code == 200
        t.check("GET /history returns 200", passed, response=r if not passed else None)
        if passed:
            body = r.json()
            print(f"           {body.get('total', 0)} discover queries in history this session")
    except Exception as exc:
        t.exception("GET /history", exc)


# ── Docs / OpenAPI ────────────────────────────────────────────────────────────


def save_openapi_spec(base_url: str, out_dir: Path) -> None:
    """Fetch and save the OpenAPI JSON schema from the running server."""
    _sub("Fetching OpenAPI spec…")
    try:
        r = httpx.get(f"{base_url}/openapi.json", timeout=10.0)
        if r.status_code == 200:
            out_dir.mkdir(parents=True, exist_ok=True)
            spec_path = out_dir / "openapi_spec.json"
            spec_path.write_text(r.text, encoding="utf-8")
            spec = r.json()
            routes = list(spec.get("paths", {}).keys())
            print(f"           Saved → {spec_path}")
            print(f"           {len(routes)} route(s): {', '.join(routes)}")
        else:
            print(f"           Could not fetch spec (HTTP {r.status_code})")
    except Exception as exc:
        print(f"           Could not fetch spec: {exc}")


# ── Pytest runner ─────────────────────────────────────────────────────────────


def run_pytest() -> int:
    """Run the pytest unit suite and return the exit code."""
    _header("pytest tests/ -v --tb=short", "─")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(ROOT_DIR),
    )
    return result.returncode


# ── Environment check ─────────────────────────────────────────────────────────


def check_env() -> None:
    missing = []
    if not os.getenv("LASTFM_API_KEY"):
        missing.append("LASTFM_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"\n  WARNING: Missing env var(s): {', '.join(missing)}")
        print("  Some tests may fail. Set them in .env or your shell.\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip endpoints that call Anthropic API")
    parser.add_argument("--no-pytest", action="store_true",
                        help="Skip the pytest unit suite")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    _header("Music Discovery Agent — Integration Test Suite", "═")
    print(f"  Server:  {base_url}")
    print(f"  Docs:    {base_url}/docs")
    print(f"  Mode:    {'fast (--skip-slow)' if args.skip_slow else 'full (Anthropic API calls enabled)'}")

    check_env()

    # ── Start server ──────────────────────────────────────────────────────────
    _header("Starting server…", "─")
    server_proc = start_server(args.port)

    print(f"  PID {server_proc.pid}  waiting for server to accept connections…")
    ready = wait_for_server(base_url, timeout=40)

    if not ready:
        stderr_output = server_proc.stderr.read().decode(errors="replace")
        print(f"\n  ERROR: Server did not start within 40 seconds.")
        if stderr_output:
            print(f"  stderr:\n{stderr_output[:600]}")
        stop_server(server_proc)
        return 1

    print(f"  Server ready.")
    print(f"\n  Interactive docs: {base_url}/docs")
    print(f"  OpenAPI schema:   {base_url}/openapi.json")

    save_openapi_spec(base_url, LOGS_DIR)

    # ── Run integration tests ─────────────────────────────────────────────────
    # Bridge calls make several rounds of Last.fm BFS + Claude; give them
    # up to 5 minutes.  Artist/rate calls should complete in under 30 s now
    # that lastfm_client has a 10 s timeout per HTTP request.
    t = TestRunner(base_url, timeout=300.0)
    overall_start = time.time()

    try:
        test_discover(t, args.skip_slow)
        test_artist(t)
        test_analyze_taste(t, args.skip_slow)
        test_bridge(t)
        test_rate_and_profile(t)
    finally:
        t.close()

    elapsed_total = time.time() - overall_start

    # ── Summary ───────────────────────────────────────────────────────────────
    t.print_summary()
    print(f"\n  Integration tests completed in {elapsed_total:.1f}s")

    results_path = LOGS_DIR / "endpoint_test_results.json"
    t.save_results(results_path)

    # ── Stop server ───────────────────────────────────────────────────────────
    _header("Stopping server…", "─")
    stop_server(server_proc)
    print(f"  Server stopped (PID {server_proc.pid})")

    # ── Pytest ────────────────────────────────────────────────────────────────
    pytest_rc = 0
    if not args.no_pytest:
        pytest_rc = run_pytest()
    else:
        print("\n  (pytest skipped — remove --no-pytest to run)")

    # ── Final exit code ───────────────────────────────────────────────────────
    _header("Done", "═")
    integration_ok = t.total_failed == 0
    pytest_ok = pytest_rc == 0

    if integration_ok and pytest_ok:
        print("  All integration tests passed. Pytest clean.")
    else:
        if not integration_ok:
            print(f"  {t.total_failed} integration test(s) failed.")
        if not pytest_ok:
            print(f"  pytest exited with code {pytest_rc}.")

    return 0 if (integration_ok and pytest_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
