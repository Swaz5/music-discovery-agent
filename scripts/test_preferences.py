"""
Preference engine integration test.

1. Saves 6 ratings (mix of liked and disliked) and prints the derived
   taste profile after each one so you can watch it evolve.
2. Runs a discovery query through MusicDiscoveryAgent and prints the
   system prompt fragment it received, so you can verify personalisation.

Run from the project root::

    python scripts/test_preferences.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent.preference_engine import PreferenceEngine
from src.agent.discovery_agent import MusicDiscoveryAgent

_W = 70

def _rule(c="─"):
    return c * _W

def _h(text, c="═"):
    print(f"\n{c * _W}\n  {text}\n{c * _W}")

# ── Seed ratings ──────────────────────────────────────────────────────────────

RATINGS = [
    # (artist, liked, notes)
    ("Massive Attack",   True,  "perfect dark trip-hop atmosphere"),
    ("Portishead",       True,  "haunting and cinematic"),
    ("Burial",           True,  "love the urban nighttime feel"),
    ("Taylor Swift",     False, "too polished and mainstream"),
    ("The Prodigy",      False, "too aggressive, not my energy"),
    ("Thom Yorke",       True,  "solo work is beautifully bleak"),
]

async def main() -> None:
    engine = PreferenceEngine()

    _h("Step 1 — Saving 6 ratings")

    for artist, liked, notes in RATINGS:
        sentiment = "LIKED  ✓" if liked else "DISLIKED ✗"
        print(f"\n  [{sentiment}] {artist}")
        if notes:
            print(f"  Notes: {notes}")

        print("  Fetching audio features …", end=" ", flush=True)
        await engine.save_preference(artist, liked=liked, notes=notes)
        print("saved.")

    # ── Print the derived taste profile ──────────────────────────────
    _h("Step 2 — Derived taste profile")
    profile = engine.get_taste_profile()

    print(f"\n  Ratings: {profile['liked_count']} liked / "
          f"{profile['disliked_count']} disliked\n")

    print("  Summary:")
    for line in profile.get("summary", "").split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

    pref_audio = profile.get("preferred_audio", {})
    if pref_audio:
        print("\n  Preferred audio (liked artists avg):")
        for k, v in pref_audio.items():
            bar_len = int((v / (200 if k == "tempo" else 1)) * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            fmt = f"{v:.2f}" if k != "tempo" else f"{v:.0f} BPM"
            print(f"    {k:<14} {bar} {fmt}")

    dis_audio = profile.get("disliked_audio", {})
    if dis_audio:
        print("\n  Disliked audio (disliked artists avg):")
        for k, v in dis_audio.items():
            fmt = f"{v:.2f}" if k != "tempo" else f"{v:.0f} BPM"
            print(f"    {k:<14} {fmt}")

    sweet = profile.get("sweet_spots", {})
    if sweet:
        print("\n  Sweet spots (liked vs. disliked delta ≥ 0.15):")
        for feat, delta in sweet.items():
            direction = "higher" if delta > 0 else "lower"
            print(f"    {feat}: {direction} by {abs(delta):.3f}")

    pref_tags = profile.get("preferred_tags", [])
    dis_tags  = profile.get("disliked_tags", [])
    if pref_tags:
        print(f"\n  Top preferred tags : {', '.join(pref_tags[:8])}")
    if dis_tags:
        print(f"  Top disliked tags  : {', '.join(dis_tags[:6])}")

    # ── Show what the agent will see in its system prompt ─────────────
    _h("Step 3 — Recommendation context (injected into system prompt)")
    ctx = engine.get_recommendation_context()
    if ctx:
        print()
        for line in ctx.splitlines():
            print(f"  {line}")
    else:
        print("  (empty — no ratings yet)")

    # ── Run a discovery query and check personalisation ───────────────
    _h("Step 4 — Running discovery with personalised prompt")
    query = "Something atmospheric and cinematic for a late night"
    print(f"\n  Query: {query!r}")
    print(f"  (The agent's system prompt now includes the taste profile above)\n")
    print(_rule())

    agent = MusicDiscoveryAgent(preference_engine=engine)
    result = await agent.discover(query)

    print(f"\n  Iterations : {result['iterations']}")
    print(f"  Tokens     : {result['total_tokens']:,}")

    print(f"\n  Tool calls:")
    for step in result["reasoning_trace"]:
        args = json.dumps(step.get("arguments", {}), ensure_ascii=False)[:65]
        status = "✗" if "error" in step else "✓"
        print(f"    [{step['iteration']}] {status} {step['tool']}({args})")

    recs = result["recommendations"]
    print(f"\n  Recommendations ({len(recs)} found):")
    for i, rec in enumerate(recs, 1):
        artist = rec.get("artist", "?")
        track  = rec.get("track", "?")
        score  = rec.get("vibe_match_score", 0.0)
        tags   = ", ".join(rec.get("genre_tags", [])) or "—"
        why    = rec.get("why", "")[:130]
        print(f"\n    {i}. {artist} — {track}")
        print(f"       Tags: {tags}  |  Vibe match: {score}")
        if why:
            print(f"       Why: {why}")

    # ── Full preference history ────────────────────────────────────────
    _h("Step 5 — Full preference history")
    history = engine.get_preference_history()
    for entry in history:
        sentiment = "✓" if entry["liked"] else "✗"
        ts = entry.get("timestamp", "")[:10]
        af = entry.get("audio_features", {})
        af_str = ""
        if af:
            parts = []
            for k in ("energy", "valence", "tempo"):
                if k in af:
                    v = af[k]
                    parts.append(f"{k}={v:.2f}" if k != "tempo" else f"tempo={v:.0f}")
            af_str = "  [" + "  ".join(parts) + "]"
        print(f"  {sentiment} {entry['artist']:<22} {ts}{af_str}")


if __name__ == "__main__":
    asyncio.run(main())
