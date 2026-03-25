"""
Audio analyzer.

Downloads and analyzes audio preview files using librosa to extract features
such as tempo, key, energy, danceability, and spectral characteristics.
Results are used to enrich track recommendations with acoustic attributes.

NOTE ON VALENCE AND DANCEABILITY
---------------------------------
Spotify's valence and danceability metrics were trained on millions of
human-labeled tracks using supervised ML models. The estimates here are
principled signal-processing approximations — they correlate with the
underlying constructs but are not equivalent:

  - Valence: derived from chroma (major/minor tendency) and spectral
    brightness. Happy/bright tracks tend to sit in major keys with higher
    spectral centroids; sad/dark tracks lean minor and dull. This heuristic
    works well on clear cases but struggles with ironic or complex pieces.

  - Danceability: derived from onset regularity and tempo proximity to a
    "sweet spot" (100-130 BPM). Tracks with steady, predictable beats in
    that range score higher. The metric will underrate syncopated or
    polyrhythmic music whose groove is complex rather than metronomic.

These limitations are features, not bugs — document them and let
downstream consumers weigh them accordingly.
"""

import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _energy_label(energy: float) -> str:
    if energy >= 0.75:
        return "high energy"
    if energy >= 0.45:
        return "moderate energy"
    return "low energy"


def _tempo_label(bpm: float) -> str:
    if bpm >= 160:
        return "very fast"
    if bpm >= 130:
        return "fast"
    if bpm >= 100:
        return "upbeat"
    if bpm >= 76:
        return "moderate"
    if bpm >= 60:
        return "slow"
    return "very slow"


def _valence_label(valence: float) -> str:
    if valence >= 0.70:
        return "joyful"
    if valence >= 0.55:
        return "positive"
    if valence >= 0.40:
        return "neutral"
    if valence >= 0.25:
        return "melancholic"
    return "dark"


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


# ---------------------------------------------------------------------------
# Feature extractors (module-level so they are individually testable)
# ---------------------------------------------------------------------------

def extract_energy(y: np.ndarray) -> float:
    """
    Energy from RMS amplitude, normalized to 0-1.

    Calibration: quiet acoustic passages sit around RMS 0.01-0.03;
    heavily compressed rock/electronic tracks hit 0.15-0.30. We map
    [0, 0.25] linearly to [0, 1] and clip above.
    """
    rms = float(np.mean(librosa.feature.rms(y=y)))
    return _clip(rms / 0.25)


def extract_loudness(y: np.ndarray) -> float:
    """Overall loudness in dBFS (negative; 0 dBFS = clipping)."""
    rms = float(np.mean(librosa.feature.rms(y=y)))
    rms = max(rms, 1e-9)  # avoid log(0)
    return float(20 * np.log10(rms))


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Beat-tracked tempo in BPM."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # beat_track returns an array in newer librosa; unpack scalar
    return float(np.atleast_1d(tempo)[0])


def extract_danceability(y: np.ndarray, sr: int) -> float:
    """
    Estimate danceability from beat regularity + tempo proximity (0-1).

    Approach (approximate — see module docstring):
      1. Inter-beat-interval (IBI) regularity: the coefficient of variation of
         the time gaps between detected beat frames. A perfectly regular 120 BPM
         grid has CV ≈ 0 (all gaps identical); irregular or sparse beats have
         higher CV. Inverted and clamped to [0, 1].
      2. Tempo proximity: proximity to the 100-130 BPM "sweet spot" common in
         dance music adds up to 0.3.
    The two components are blended 70/30. If fewer than 3 beats are detected
    (e.g., a pure sine wave), regularity defaults to 0.
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    if len(beats) >= 3:
        ibi = np.diff(beats.astype(float))  # inter-beat intervals in frames
        cv = float(np.std(ibi) / (np.mean(ibi) + 1e-6))
        regularity = _clip(1.0 - cv)
    else:
        regularity = 0.0

    if bpm > 0:
        center = 115.0
        spread = 45.0
        tempo_score = _clip(1.0 - abs(bpm - center) / spread)
    else:
        tempo_score = 0.0

    return _clip(0.70 * regularity + 0.30 * tempo_score)


def extract_valence(y: np.ndarray, sr: int) -> float:
    """
    Estimate valence (musical positivity) from chroma + brightness (0-1).

    Approach (approximate — see module docstring):
      1. Major-key tendency: compare energy on major-triad chroma bins
         (C, E, G) vs minor-triad bins (C, Eb, G) for the dominant chroma.
      2. Spectral brightness: higher spectral centroid → brighter → more
         positive affect.
    Blended 50/50.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)  # shape (12,)

    # Find the dominant root (highest mean energy)
    root = int(np.argmax(mean_chroma))
    # Major triad intervals: root, +4, +7 semitones
    # Minor triad intervals: root, +3, +7 semitones
    major_bins = [(root + i) % 12 for i in (0, 4, 7)]
    minor_bins = [(root + i) % 12 for i in (0, 3, 7)]
    major_energy = float(np.mean(mean_chroma[major_bins]))
    minor_energy = float(np.mean(mean_chroma[minor_bins]))
    total = major_energy + minor_energy + 1e-6
    major_score = major_energy / total  # 0.5 = ambiguous, >0.5 = major-leaning

    # Brightness: spectral centroid normalized over 0-11025 Hz (Nyquist at sr=22050)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = float(np.mean(centroid))
    brightness = _clip(mean_centroid / 4000.0)  # typical speech/music centroid ≈ 1000-3000

    return _clip(0.50 * major_score + 0.50 * brightness)


def extract_acousticness(y: np.ndarray) -> float:
    """
    Estimate acousticness from spectral flatness and high-frequency energy (0-1).

    Low spectral flatness → tonal (acoustic instruments).
    Little high-frequency energy → acoustic (not electronic/distorted).
    """
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    # Typical range: near-0 (pure tone) to ~0.3 (white noise)
    flatness_score = _clip(1.0 - flatness / 0.15)

    # High-frequency energy ratio above 4 kHz (bin ~4000/22050*n_fft)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=22050)
    hi_mask = freqs >= 4000
    hi_energy = float(np.mean(S[hi_mask, :]))
    lo_energy = float(np.mean(S[~hi_mask, :]))
    hf_ratio = hi_energy / (lo_energy + 1e-6)
    hf_score = _clip(1.0 - hf_ratio / 0.5)

    return _clip(0.60 * flatness_score + 0.40 * hf_score)


def extract_instrumentalness(y: np.ndarray, sr: int) -> float:
    """
    Estimate instrumentalness from spectral contrast in vocal range (0-1).

    High spectral contrast in 300-3000 Hz vocal range → vocals likely present
    → lower instrumentalness score. Flat contrast → no prominent harmonic
    source in that range → more instrumental.

    This is an inversion heuristic: it detects "vocal-like contrast", not
    vocals directly. A dense orchestral texture can read as low
    instrumentalness; a minimal synth pad can read as high.
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    # Bands 1-3 roughly cover 200-3200 Hz — the vocal range
    vocal_contrast = float(np.mean(contrast[1:4, :]))
    # Typical contrast 10-40 dB; high → vocals likely present → less instrumental
    instrumentalness = _clip(1.0 - vocal_contrast / 35.0)
    return instrumentalness


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------

class AudioAnalyzer:
    """
    Extracts acoustic features from MP3/WAV audio files using librosa.

    All features are normalized to [0, 1] except tempo (BPM) and
    loudness (dBFS). See module-level docstring for caveats on valence
    and danceability estimates.
    """

    def analyze_track(self, file_path: str) -> dict:
        """
        Load and analyze a single audio file. Returns a feature dict.

        Parameters
        ----------
        file_path : str
            Path to an MP3 or WAV file. Deezer 30-second previews work well.

        Returns
        -------
        dict with keys: energy, energy_label, tempo, tempo_label,
            danceability, valence, valence_label, acousticness,
            instrumentalness, loudness.
        """
        y, sr = librosa.load(file_path, sr=22050)

        energy = extract_energy(y)
        tempo = extract_tempo(y, sr)
        danceability = extract_danceability(y, sr)
        valence = extract_valence(y, sr)
        acousticness = extract_acousticness(y)
        instrumentalness = extract_instrumentalness(y, sr)
        loudness = extract_loudness(y)

        return {
            "file_path": file_path,
            "energy": round(energy, 3),
            "energy_label": _energy_label(energy),
            "tempo": round(tempo, 1),
            "tempo_label": _tempo_label(tempo),
            "danceability": round(danceability, 3),
            "valence": round(valence, 3),
            "valence_label": _valence_label(valence),
            "acousticness": round(acousticness, 3),
            "instrumentalness": round(instrumentalness, 3),
            "loudness": round(loudness, 1),
        }

    def analyze_batch(self, file_paths: list[str]) -> list[dict]:
        """Analyze multiple audio files. Returns a list of feature dicts."""
        return [self.analyze_track(fp) for fp in file_paths]

    def compute_artist_audio_profile(self, track_analyses: list[dict]) -> dict:
        """
        Average numeric features across a set of track analyses.

        Returns an aggregated "audio profile" representing the artist's
        typical sound. Non-numeric fields (labels, file_path) are excluded.
        """
        if not track_analyses:
            return {}

        numeric_keys = [
            "energy", "tempo", "danceability", "valence",
            "acousticness", "instrumentalness", "loudness",
        ]

        profile = {}
        for key in numeric_keys:
            values = [t[key] for t in track_analyses if key in t]
            if values:
                profile[key] = round(float(np.mean(values)), 3)

        # Add labels for the averaged values
        if "energy" in profile:
            profile["energy_label"] = _energy_label(profile["energy"])
        if "tempo" in profile:
            profile["tempo_label"] = _tempo_label(profile["tempo"])
        if "valence" in profile:
            profile["valence_label"] = _valence_label(profile["valence"])
        profile["track_count"] = len(track_analyses)

        return profile


# ---------------------------------------------------------------------------
# __main__: demo with Deezer previews
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import asyncio
    from pathlib import Path
    from src.data.deezer_client import search_tracks, download_preview

    PREVIEWS_DIR = Path("data/previews")

    async def fetch_preview(query: str) -> str | None:
        """Search for a track and download its preview. Returns local path or None."""
        tracks = await search_tracks(query, limit=1)
        if not tracks or not tracks[0]["preview_url"]:
            print(f"  [!] No preview found for: {query}")
            return None
        t = tracks[0]
        safe_name = t["title"].replace(" ", "_").replace("/", "-")[:40] + ".mp3"
        save_path = str(PREVIEWS_DIR / safe_name)
        path = await download_preview(t["preview_url"], save_path)
        print(f"  Downloaded: {t['artist']} — {t['title']} -> {path}")
        return path

    async def main():
        analyzer = AudioAnalyzer()
        PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

        # --- 1. Analyze the existing Creep preview if present ---
        creep_path = str(PREVIEWS_DIR / "Creep.mp3")
        if Path(creep_path).exists():
            print("=" * 60)
            print(f"Analyzing existing preview: {creep_path}")
            print("=" * 60)
            features = analyzer.analyze_track(creep_path)
            for k, v in features.items():
                if k != "file_path":
                    print(f"  {k:<22} {v}")
        else:
            print(f"[!] {creep_path} not found — run deezer_client.py first")

        # --- 2. Compare three contrasting tracks ---
        print("\n" + "=" * 60)
        print("Fetching 3 contrasting tracks for comparison...")
        print("=" * 60)

        contrasting = [
            ("Slayer Raining Blood metal", "metal / fast / heavy"),
            ("Nick Drake Pink Moon acoustic", "acoustic / slow / delicate"),
            ("Daft Punk Around the World dance", "electronic / dance / 120 BPM"),
        ]

        analyses = []
        for query, description in contrasting:
            print(f"\n[{description}] searching: '{query}'")
            path = await fetch_preview(query)
            if path:
                features = analyzer.analyze_track(path)
                features["description"] = description
                analyses.append(features)

        if analyses:
            print("\n" + "=" * 60)
            print("Feature comparison:")
            print("=" * 60)
            col_w = 26
            header = f"{'Feature':<20}" + "".join(
                f"{a['description'][:col_w]:<{col_w}}" for a in analyses
            )
            print(header)
            print("-" * (20 + col_w * len(analyses)))
            for key in ("energy", "tempo", "danceability", "valence",
                        "acousticness", "instrumentalness", "loudness"):
                row = f"{key:<20}"
                for a in analyses:
                    row += f"{str(a.get(key, 'N/A')):<{col_w}}"
                print(row)

            # Artist audio profile from all three
            print("\n" + "=" * 60)
            print("Aggregate profile (all 3 tracks):")
            print("=" * 60)
            profile = analyzer.compute_artist_audio_profile(analyses)
            for k, v in profile.items():
                print(f"  {k:<22} {v}")

    asyncio.run(main())
