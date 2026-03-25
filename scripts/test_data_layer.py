"""
Comprehensive data layer smoke test.

Builds full artist profiles for 6 diverse artists, runs vibe matching
searches, and prints timing breakdowns. Run from the project root:

    python scripts/test_data_layer.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (block chars need it)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Make sure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.music_service import (
    get_full_artist_profile,
    get_similar_with_features,
    search_artists_by_vibe,
    cache_clear,
)

# ---------------------------------------------------------------------------
# ANSI colour helpers (no third-party deps)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"

BG_BLUE    = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN    = "\033[46m"


def _c(text, *codes):
    return "".join(codes) + str(text) + RESET


def bar(value: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Render a simple 0-1 value as an ASCII progress bar."""
    filled = round(value * width)
    return fill * filled + empty * (width - filled)


def colour_bar(value: float, width: int = 20) -> str:
    """Colour-coded bar: green (high) → yellow → red (low)."""
    if value >= 0.65:
        colour = GREEN
    elif value >= 0.35:
        colour = YELLOW
    else:
        colour = RED
    return _c(bar(value, width), colour)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.splits: list[tuple[str, float]] = []
        self._t0 = time.monotonic()
        self._lap = self._t0

    def split(self, label: str) -> float:
        now = time.monotonic()
        elapsed = now - self._lap
        self._lap = now
        self.splits.append((label, elapsed))
        return elapsed

    def total(self) -> float:
        return time.monotonic() - self._t0


# ---------------------------------------------------------------------------
# Vibe summary builder
# ---------------------------------------------------------------------------

def vibe_summary(ap: dict) -> str:
    """Turn an audio profile dict into a short human-readable sentence."""
    if not ap:
        return _c("(no audio profile available)", DIM)

    energy     = ap.get("avg_energy", 0)
    dance      = ap.get("avg_danceability", 0)
    valence    = ap.get("avg_valence", 0)
    tempo      = ap.get("avg_tempo", 0)
    acoustic   = ap.get("avg_acousticness", 0)

    # Energy
    if energy >= 0.75:   e_word = "high-energy"
    elif energy >= 0.45: e_word = "mid-energy"
    else:                e_word = "low-energy"

    # Mood
    if valence >= 0.60:   m_word = "uplifting"
    elif valence >= 0.42: m_word = "bittersweet"
    else:                 m_word = "moody"

    # Texture
    if acoustic >= 0.70: t_word = "acoustic"
    elif dance >= 0.75:  t_word = "groove-driven"
    else:                t_word = "textured"

    # Tempo
    if tempo >= 135:    bpm_word = "at a racing pace"
    elif tempo >= 115:  bpm_word = f"around {tempo:.0f} BPM"
    elif tempo >= 90:   bpm_word = "at a mid-tempo groove"
    else:               bpm_word = "at a slow burn"

    return f"{e_word}, {m_word}, {t_word} — {bpm_word}"


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

W = 66  # total display width


def section_header(title: str, bg=BG_BLUE):
    pad = W - len(title) - 2
    left = pad // 2
    right = pad - left
    print()
    print(_c(f" {' ' * left}{title}{' ' * right} ", bg, BOLD, WHITE))


def artist_header(name: str, genre: str):
    label = f"  {name}  "
    genre_label = f" {genre} "
    print()
    print(_c(label, BG_MAGENTA, BOLD, WHITE) + "  " + _c(genre_label, DIM))


def metric_row(label: str, value: float, show_bar: bool = True, suffix: str = ""):
    bar_str = colour_bar(value) if show_bar else ""
    val_str = _c(f"{value:.3f}", BOLD)
    suffix_str = _c(f" {suffix}", DIM) if suffix else ""
    if show_bar:
        print(f"    {label:<18} {val_str}  {bar_str}{suffix_str}")
    else:
        print(f"    {label:<18} {val_str}{suffix_str}")


def timing_row(label: str, elapsed: float, note: str = ""):
    ms = elapsed * 1000
    if ms < 5:
        t_str = _c(f"{ms:6.1f} ms", GREEN, BOLD)
    elif ms < 500:
        t_str = _c(f"{ms:6.1f} ms", GREEN)
    elif ms < 3000:
        t_str = _c(f"{ms:6.1f} ms", YELLOW)
    else:
        t_str = _c(f"{ms:6.1f} ms", RED)
    note_str = _c(f"  {note}", DIM) if note else ""
    print(f"    {label:<36} {t_str}{note_str}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

ARTISTS = [
    ("The Black Keys",  "alt rock"),
    ("Flume",           "electronic"),
    ("Frank Ocean",     "R&B / pop"),
    ("Steve Lacy",      "soul / indie"),
    ("Tame Impala",     "psych rock"),
    ("Drake",           "hip hop"),
]

VIBE_SEARCHES = [
    {
        "label": "Party / High Energy + High Danceability",
        "tags": ["dance", "electronic", "pop"],
        "energy_range": (0.70, 1.0),
        "valence_range": (0.0, 1.0),
        "bg": BG_CYAN,
    },
    {
        "label": "Chill Acoustic / Low Energy + High Acousticness",
        "tags": ["folk", "acoustic", "singer-songwriter"],
        "energy_range": (0.0, 0.55),
        "valence_range": (0.0, 1.0),
        "bg": BG_CYAN,
    },
    {
        "label": "Intense / Dark — High Energy + Low Valence",
        "tags": ["metal", "post-rock", "darkwave"],
        "energy_range": (0.65, 1.0),
        "valence_range": (0.0, 0.45),
        "bg": BG_CYAN,
    },
]


async def build_profiles(timer: Timer) -> dict[str, dict]:
    """Fetch all 6 artist profiles IN PARALLEL and record timing."""
    section_header("BUILDING ARTIST PROFILES  (parallel fetch)")
    print()
    print(_c("  Firing all 6 profile requests concurrently…", DIM))

    t_start = time.monotonic()
    results = await asyncio.gather(
        *[get_full_artist_profile(name) for name, _ in ARTISTS],
        return_exceptions=True,
    )
    wall = time.monotonic() - t_start
    timer.split("parallel profile fetch (all 6)")

    profiles: dict[str, dict] = {}
    for (name, genre), result in zip(ARTISTS, results):
        if isinstance(result, Exception):
            print(_c(f"  ✗ {name}: {result}", RED))
        else:
            profiles[name] = result

    sequential_estimate = wall * len(profiles)  # very rough lower bound
    print()
    timing_row("Wall time (parallel)", wall, f"~{sequential_estimate:.0f}s if sequential")
    return profiles


async def print_artist_summaries(profiles: dict[str, dict], timer: Timer):
    section_header("ARTIST SUMMARIES")

    for name, genre in ARTISTS:
        profile = profiles.get(name)
        if not profile:
            print(_c(f"\n  ✗ {name} — no data", RED))
            continue

        artist_header(name, genre)

        # Tags & stats
        tags = profile.get("tags", [])[:3]
        fans = profile.get("fan_count", 0)
        listeners = profile.get("listeners", 0)
        print(f"    Tags:        {_c(', '.join(tags) or '—', CYAN)}")
        print(f"    Deezer fans: {_c(f'{fans:,}', YELLOW)}   "
              f"Last.fm listeners: {_c(f'{listeners:,}', YELLOW)}")

        # Similar artists
        similar = [a["name"] for a in profile.get("similar_artists", [])[:3]]
        print(f"    Similar to:  {_c(', '.join(similar) or '—', MAGENTA)}")

        # Audio profile
        ap = profile.get("audio_profile", {})
        if ap:
            print()
            print(_c("    Audio Profile:", BOLD))
            metric_row("energy",       ap.get("avg_energy", 0))
            metric_row("danceability", ap.get("avg_danceability", 0))
            metric_row("valence",      ap.get("avg_valence", 0))
            bpm = ap.get("avg_tempo", 0)
            print(f"    {'tempo':<18} {_c(f'{bpm:.1f} BPM', BOLD)}")
            acoustic = ap.get("avg_acousticness", 0)
            metric_row("acousticness", acoustic)
        else:
            print(_c("    [no audio profile]", DIM))

        # Top tracks with per-track features
        tracks = profile.get("top_tracks", [])[:3]
        if tracks:
            print()
            print(_c("    Top tracks:", BOLD))
            for t in tracks:
                title = t.get("title", t.get("name", "?"))
                af = t.get("audio_features", {})
                e = af.get("energy", None)
                v = af.get("valence", None)
                bpm_t = af.get("tempo", None)
                feat_str = ""
                if e is not None:
                    feat_str = (f"  {_c(f'e={e:.2f}', DIM)}"
                                f"  {_c(f'v={v:.2f}', DIM)}"
                                f"  {_c(f'{bpm_t:.0f}bpm', DIM)}")
                print(f"      {_c('•', CYAN)} {title[:42]:<42}{feat_str}")

        # Vibe summary
        print()
        vibe = vibe_summary(ap)
        print(f"    {_c('Vibe:', BOLD)} {_c(vibe, YELLOW)}")

    timer.split("print summaries")


async def run_vibe_searches(timer: Timer):
    section_header("VIBE MATCHING SEARCHES")

    for search in VIBE_SEARCHES:
        label = search["label"]
        tags = search["tags"]
        er = search["energy_range"]
        vr = search["valence_range"]

        print()
        print(_c(f"  ▶ {label}", BOLD, CYAN))
        print(_c(f"    tags={tags}  energy={er}  valence={vr}", DIM))

        t0 = time.monotonic()
        try:
            matches = await search_artists_by_vibe(tags, energy_range=er, valence_range=vr)
        except Exception as exc:
            print(_c(f"    ✗ failed: {exc}", RED))
            continue
        elapsed = time.monotonic() - t0
        timer.split(f"vibe search: {label[:30]}")

        if not matches:
            print(_c("    No artists matched — try wider ranges.", DIM))
        else:
            for m in matches[:6]:
                name = m["name"]
                rel = m.get("relevance", 0)
                af = m.get("audio_features", {})
                e = af.get("energy", 0)
                v = af.get("valence", 0)
                d = af.get("danceability", 0)
                bpm_m = af.get("tempo", 0)
                rel_stars = _c("★" * rel + "☆" * (3 - min(rel, 3)), YELLOW)
                print(
                    f"    {rel_stars}  {name:<28}"
                    f"  {_c(f'e={e:.2f}', DIM)}"
                    f"  {_c(f'v={v:.2f}', DIM)}"
                    f"  {_c(f'd={d:.2f}', DIM)}"
                    f"  {_c(f'{bpm_m:.0f}bpm', DIM)}"
                )

        timing_row("  fetch + analyze", elapsed)


async def print_timing_summary(timer: Timer):
    section_header("TIMING SUMMARY")
    print()
    for label, elapsed in timer.splits:
        timing_row(label, elapsed)
    print()
    timing_row("TOTAL", timer.total())
    print()
    print(_c(
        "  Note: Last.fm + Deezer calls fire in parallel per artist,\n"
        "  and all 6 artist requests above fired concurrently too.\n"
        "  Cache is warm for any re-run — second run will be <10 ms.",
        DIM
    ))


async def main():
    # Suppress noisy mpg123 warnings from stderr
    import os
    os.environ.setdefault("AUDIOREAD_FFDEC_STDERR", "0")

    cache_clear()  # start cold so timing is realistic

    print()
    print(_c("  MUSIC DISCOVERY — DATA LAYER SMOKE TEST  ", BG_BLUE, BOLD, WHITE))
    print(_c("  6 artists · vibe matching · timing breakdown  ", BG_BLUE, DIM, WHITE))

    timer = Timer()

    profiles = await build_profiles(timer)
    await print_artist_summaries(profiles, timer)
    await run_vibe_searches(timer)
    await print_timing_summary(timer)

    # --- Sanity checks (fail loudly if something is obviously wrong) ---
    section_header("SANITY CHECKS")
    print()
    failures = []

    for name, _ in ARTISTS:
        p = profiles.get(name, {})
        ap = p.get("audio_profile", {})

        if not p:
            failures.append(f"{name}: no profile returned")
            continue
        if not p.get("tags"):
            failures.append(f"{name}: no tags")
        if not ap:
            failures.append(f"{name}: no audio profile (previews failed?)")
        else:
            for key in ("avg_energy", "avg_danceability", "avg_valence", "avg_acousticness"):
                v = ap.get(key, -1)
                if not (0.0 <= v <= 1.0):
                    failures.append(f"{name}: {key}={v} out of [0,1]")
            bpm = ap.get("avg_tempo", -1)
            if not (40 <= bpm <= 220):
                failures.append(f"{name}: avg_tempo={bpm:.1f} outside [40,220]")

    if failures:
        print(_c(f"  ✗ {len(failures)} check(s) failed:", RED, BOLD))
        for f in failures:
            print(_c(f"    • {f}", RED))
    else:
        print(_c("  ✓ All sanity checks passed.", GREEN, BOLD))

    print()


if __name__ == "__main__":
    asyncio.run(main())
