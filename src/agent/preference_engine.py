"""
Preference engine — tracks and learns from user ratings.

Persists every rating to data/preferences.json and derives a taste profile
from the accumulated history. The profile can be injected into the agent's
system prompt so Claude personalises recommendations without you having to
re-state your tastes on every query.

Data layout of preferences.json
--------------------------------
{
  "ratings": [
    {
      "artist":         "Portishead",
      "liked":          true,
      "notes":          "love the dark atmosphere",
      "timestamp":      "2026-03-25T14:30:00",
      "tags":           ["trip-hop", "electronic", "dark"],
      "audio_features": {"energy": 0.42, "valence": 0.18, ...}
    },
    ...
  ],
  "taste_profile": { ... }   # recalculated after every save
}

Audio features schema (Librosa-derived, same as music_service output)
----------------------------------------------------------------------
  energy        0–1  (loudness / intensity)
  danceability  0–1  (rhythm regularity)
  valence       0–1  (0 = dark/sad, 1 = bright/happy)
  tempo         BPM
  acousticness  0–1
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PREFERENCES_PATH = Path("data/preferences.json")

# Audio features we care about for profile averaging
_AUDIO_KEYS = ("energy", "danceability", "valence", "tempo", "acousticness")

# Features where liked vs. disliked diverge most dramatically → "sweet spots"
_SWEET_SPOT_THRESHOLD = 0.15   # delta above this is reported


class PreferenceEngine:
    """
    Tracks user ratings and derives a taste profile from them.

    Usage::

        engine = PreferenceEngine()

        # Saving is async because it may fetch audio features from Deezer
        await engine.save_preference("Massive Attack", liked=True,
                                     notes="perfect dark energy")
        await engine.save_preference("Taylor Swift", liked=False)

        # Instant — reads from cache
        profile = engine.get_taste_profile()
        context  = engine.get_recommendation_context()
    """

    def __init__(self, path: Path = PREFERENCES_PATH) -> None:
        self._path = path
        self._data = self._load()

    # ──────────────────────────────────────────────────────────────────
    # Persistence helpers
    # ──────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read preferences file: %s", exc)
        return {"ratings": [], "taste_profile": {}}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def save_preference(
        self,
        artist: str,
        liked: bool,
        notes: str = "",
        audio_features: dict | None = None,
    ) -> None:
        """
        Persist a rating for an artist and refresh the taste profile.

        If *audio_features* is not supplied the engine fetches the artist
        profile from the music service and extracts the averaged audio
        profile from there (non-blocking, graceful if unavailable).

        Parameters
        ----------
        artist : str
            Artist name exactly as used elsewhere in the system.
        liked : bool
            True = liked, False = disliked.
        notes : str
            Optional free-text note explaining the rating.
        audio_features : dict, optional
            Pre-computed feature dict with keys matching *_AUDIO_KEYS*.
            If omitted the engine will fetch them automatically.
        """
        tags: list[str] = []
        features: dict[str, Any] = {}

        # Fetch tags + audio features from the music service when not supplied
        if audio_features is None:
            try:
                from src.data import music_service  # local import — avoids circular at module level
                profile = await music_service.get_full_artist_profile(artist)
                tags = profile.get("tags", [])[:8]
                features = profile.get("audio_profile", {})
            except Exception as exc:
                logger.warning(
                    "Could not fetch profile for %r while saving preference: %s", artist, exc
                )
        else:
            features = {k: v for k, v in audio_features.items() if k in _AUDIO_KEYS}

        entry: dict[str, Any] = {
            "artist": artist,
            "liked": liked,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "tags": tags,
            "audio_features": features,
        }

        # Overwrite any existing rating for the same artist
        ratings: list[dict] = self._data.setdefault("ratings", [])
        ratings[:] = [r for r in ratings if r["artist"].lower() != artist.lower()]
        ratings.append(entry)

        self._data["taste_profile"] = self._compute_profile(ratings)
        self._save()

        sentiment = "liked" if liked else "disliked"
        logger.info("Saved: %s → %s", artist, sentiment)

    def get_taste_profile(self) -> dict:
        """
        Return the derived taste profile.

        The profile is computed from all stored ratings and cached in the
        JSON file. Keys:

            preferred_tags        list[str] — most common tags of liked artists
            preferred_audio       dict      — avg audio features of liked artists
            disliked_tags         list[str] — most common tags of disliked artists
            disliked_audio        dict      — avg audio features of disliked artists
            sweet_spots           dict      — features where liked ↔ disliked diverge most
            total_ratings         int
            liked_count           int
            disliked_count        int
            summary               str       — human-readable taste description
        """
        return self._data.get("taste_profile", {})

    def get_recommendation_context(self) -> str:
        """
        Return a formatted block describing the user's taste for injection
        into the agent's system prompt.

        Returns an empty string when no ratings exist yet.
        """
        profile = self.get_taste_profile()
        if not profile or profile.get("total_ratings", 0) == 0:
            return ""

        lines: list[str] = ["## User taste profile (personalise to this)"]

        # ── Summary sentence ──────────────────────────────────────────
        summary = profile.get("summary", "")
        if summary:
            lines.append(summary)
            lines.append("")

        # ── Preferred audio ───────────────────────────────────────────
        pref_audio = profile.get("preferred_audio", {})
        if pref_audio:
            lines.append("**Preferred audio characteristics** (averaged across liked artists):")
            for key in _AUDIO_KEYS:
                if key in pref_audio:
                    v = pref_audio[key]
                    fmt = f"{v:.2f}" if key != "tempo" else f"{v:.0f} BPM"
                    lines.append(f"  • {key}: {fmt}")
            lines.append("")

        # ── Preferred tags ────────────────────────────────────────────
        pref_tags = profile.get("preferred_tags", [])
        if pref_tags:
            lines.append(f"**Gravitates toward:** {', '.join(pref_tags[:8])}")

        # ── Disliked tags ─────────────────────────────────────────────
        dis_tags = profile.get("disliked_tags", [])
        if dis_tags:
            lines.append(f"**Tends to dislike:** {', '.join(dis_tags[:6])}")

        # ── Sweet spots ───────────────────────────────────────────────
        sweet = profile.get("sweet_spots", {})
        if sweet:
            notes: list[str] = []
            for feat, delta in sweet.items():
                direction = "higher" if delta > 0 else "lower"
                notes.append(f"{feat} ({direction} than their dislikes by {abs(delta):.2f})")
            lines.append(f"**Strongest preferences:** {'; '.join(notes)}")

        # ── Counts ────────────────────────────────────────────────────
        lines.append(
            f"\n_(Based on {profile['liked_count']} liked / "
            f"{profile['disliked_count']} disliked ratings. "
            f"Still suggest surprises — the user wants to discover, not just confirm.)_"
        )

        return "\n".join(lines)

    def get_preference_history(self) -> list[dict]:
        """Return the full list of rating entries, oldest first."""
        return list(self._data.get("ratings", []))

    # ──────────────────────────────────────────────────────────────────
    # Profile computation
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _avg_audio(entries: list[dict]) -> dict:
        """Average the audio features across a set of rating entries."""
        buckets: dict[str, list[float]] = {k: [] for k in _AUDIO_KEYS}
        for e in entries:
            for key in _AUDIO_KEYS:
                val = e.get("audio_features", {}).get(key)
                if val is not None:
                    buckets[key].append(float(val))
        return {
            k: round(sum(v) / len(v), 3)
            for k, v in buckets.items()
            if v
        }

    @staticmethod
    def _top_tags(entries: list[dict], n: int = 10) -> list[str]:
        """Return the *n* most common tags across a set of rating entries."""
        counter: Counter = Counter()
        for e in entries:
            for tag in e.get("tags", []):
                if tag:
                    counter[tag.lower()] += 1
        return [tag for tag, _ in counter.most_common(n)]

    @staticmethod
    def _build_summary(
        liked_count: int,
        disliked_count: int,
        pref_tags: list[str],
        dis_tags: list[str],
        pref_audio: dict,
        sweet_spots: dict,
    ) -> str:
        """Produce a one-paragraph human-readable taste summary."""
        if liked_count == 0:
            return "No liked artists recorded yet."

        parts: list[str] = []

        # Energy characterisation
        energy = pref_audio.get("energy")
        if energy is not None:
            if energy >= 0.70:
                e_desc = "high-energy"
            elif energy >= 0.50:
                e_desc = "mid-energy"
            else:
                e_desc = "low-energy, atmospheric"
            parts.append(f"prefers {e_desc} music (avg energy {energy:.2f})")

        # Valence characterisation
        valence = pref_audio.get("valence")
        if valence is not None:
            if valence <= 0.35:
                v_desc = "emotionally dark or melancholic"
            elif valence <= 0.55:
                v_desc = "emotionally ambiguous or bittersweet"
            else:
                v_desc = "uplifting or positive"
            parts.append(f"tends toward {v_desc} sounds (avg valence {valence:.2f})")

        # Tempo characterisation
        tempo = pref_audio.get("tempo")
        if tempo is not None:
            if tempo >= 140:
                t_desc = "fast-paced"
            elif tempo >= 110:
                t_desc = "moderate-tempo"
            else:
                t_desc = "slow or downtempo"
            parts.append(f"{t_desc} rhythms (avg {tempo:.0f} BPM)")

        sentence = "This user " + ", ".join(parts) + "." if parts else ""

        if pref_tags:
            sentence += (
                f" They gravitate toward: {', '.join(pref_tags[:5])}."
            )
        if dis_tags:
            sentence += (
                f" They tend to dislike: {', '.join(dis_tags[:4])}."
            )

        return sentence

    def _compute_profile(self, ratings: list[dict]) -> dict:
        """Derive the full taste profile from a list of rating entries."""
        liked = [r for r in ratings if r.get("liked")]
        disliked = [r for r in ratings if not r.get("liked")]

        pref_audio = self._avg_audio(liked)
        dis_audio = self._avg_audio(disliked)
        pref_tags = self._top_tags(liked)
        dis_tags = self._top_tags(disliked)

        # Sweet spots: features where liked avg differs from disliked avg
        # by more than the threshold. Positive delta means user prefers higher.
        sweet_spots: dict[str, float] = {}
        for key in _AUDIO_KEYS:
            if key in pref_audio and key in dis_audio:
                delta = round(pref_audio[key] - dis_audio[key], 3)
                if abs(delta) >= _SWEET_SPOT_THRESHOLD:
                    sweet_spots[key] = delta

        summary = self._build_summary(
            liked_count=len(liked),
            disliked_count=len(disliked),
            pref_tags=pref_tags,
            dis_tags=dis_tags,
            pref_audio=pref_audio,
            sweet_spots=sweet_spots,
        )

        return {
            "preferred_tags": pref_tags,
            "preferred_audio": pref_audio,
            "disliked_tags": dis_tags,
            "disliked_audio": dis_audio,
            "sweet_spots": sweet_spots,
            "total_ratings": len(ratings),
            "liked_count": len(liked),
            "disliked_count": len(disliked),
            "summary": summary,
        }
