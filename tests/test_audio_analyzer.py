"""
Tests for AudioAnalyzer using synthetically generated audio signals.

Rather than downloading real MP3s, we generate known audio signals
(sine waves at specific frequencies, white noise, silent clips) and
verify that the extractor produces sensible relative values — e.g.,
loud noise > quiet sine wave for energy; a 120 BPM click track returns
a tempo near 120 BPM.

All tests write to a temporary directory (tmp_path) via soundfile.
"""

import math
import numpy as np
import pytest
import soundfile as sf

from src.data.audio_analyzer import (
    AudioAnalyzer,
    extract_energy,
    extract_loudness,
    extract_tempo,
    extract_danceability,
    extract_valence,
    extract_acousticness,
    extract_instrumentalness,
    _energy_label,
    _tempo_label,
    _valence_label,
)

SR = 22050
DURATION = 10  # seconds — short enough to keep tests fast


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

def _sine(freq: float = 440.0, duration: float = DURATION, amplitude: float = 0.5) -> np.ndarray:
    """Pure sine wave at a given frequency."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(duration: float = DURATION, amplitude: float = 0.5) -> np.ndarray:
    """White noise."""
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(SR * duration))).astype(np.float32)


def _silence(duration: float = DURATION) -> np.ndarray:
    return np.zeros(int(SR * duration), dtype=np.float32)


def _click_track(bpm: float, duration: float = DURATION) -> np.ndarray:
    """
    Impulse click track at a given BPM, embedded in low-level noise so
    librosa's beat tracker can find it.
    """
    y = _noise(duration, amplitude=0.01)
    samples_per_beat = int(SR * 60.0 / bpm)
    for i in range(0, len(y), samples_per_beat):
        y[i : i + 441] += 0.8  # sharp transient
    return np.clip(y, -1.0, 1.0)


def _save(y: np.ndarray, path) -> str:
    sf.write(str(path), y, SR)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: label helpers
# ---------------------------------------------------------------------------

def test_energy_label_high():
    assert _energy_label(0.80) == "high energy"

def test_energy_label_moderate():
    assert _energy_label(0.55) == "moderate energy"

def test_energy_label_low():
    assert _energy_label(0.20) == "low energy"

def test_tempo_label_fast():
    assert _tempo_label(140) == "fast"

def test_tempo_label_upbeat():
    assert _tempo_label(115) == "upbeat"

def test_tempo_label_slow():
    assert _tempo_label(65) == "slow"

def test_valence_label_joyful():
    assert _valence_label(0.75) == "joyful"

def test_valence_label_melancholic():
    assert _valence_label(0.30) == "melancholic"


# ---------------------------------------------------------------------------
# Tests: extract_energy
# ---------------------------------------------------------------------------

def test_energy_loud_gt_quiet():
    loud = _sine(amplitude=0.8)
    quiet = _sine(amplitude=0.05)
    assert extract_energy(loud) > extract_energy(quiet)

def test_energy_silence_near_zero():
    assert extract_energy(_silence()) < 0.02

def test_energy_bounded():
    y = _noise(amplitude=1.0)
    e = extract_energy(y)
    assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# Tests: extract_loudness
# ---------------------------------------------------------------------------

def test_loudness_loud_gt_quiet():
    loud = _sine(amplitude=0.8)
    quiet = _sine(amplitude=0.1)
    assert extract_loudness(loud) > extract_loudness(quiet)

def test_loudness_is_negative_dbfs():
    # A proper audio signal should be below 0 dBFS
    y = _sine(amplitude=0.5)
    assert extract_loudness(y) < 0.0

def test_loudness_silence_very_negative():
    assert extract_loudness(_silence()) < -60.0


# ---------------------------------------------------------------------------
# Tests: extract_tempo
# ---------------------------------------------------------------------------

def test_tempo_click_120bpm(tmp_path):
    """Click track at 120 BPM should be detected within ±20 BPM."""
    y = _click_track(120.0)
    tempo = extract_tempo(y, SR)
    assert abs(tempo - 120.0) <= 20.0

def test_tempo_positive():
    y = _noise()
    assert extract_tempo(y, SR) > 0


# ---------------------------------------------------------------------------
# Tests: extract_danceability
# ---------------------------------------------------------------------------

def test_danceability_bounded():
    y = _sine()
    d = extract_danceability(y, SR)
    assert 0.0 <= d <= 1.0

def test_danceability_click_120_above_noise():
    """
    A 120 BPM click track (near the sweet spot) should score higher
    danceability than pure white noise with no beat structure.
    """
    click = _click_track(120.0)
    noise = _noise()
    assert extract_danceability(click, SR) >= extract_danceability(noise, SR)


# ---------------------------------------------------------------------------
# Tests: extract_valence
# ---------------------------------------------------------------------------

def test_valence_bounded():
    y = _sine()
    v = extract_valence(y, SR)
    assert 0.0 <= v <= 1.0

def test_valence_high_freq_brighter_than_low_freq():
    """
    A high-frequency sine (bright) should score higher valence than a
    low-frequency sine (dark/dull) — the brightness component should dominate.
    """
    high = _sine(freq=3000.0)
    low = _sine(freq=80.0)
    assert extract_valence(high, SR) >= extract_valence(low, SR)


# ---------------------------------------------------------------------------
# Tests: extract_acousticness
# ---------------------------------------------------------------------------

def test_acousticness_bounded():
    y = _sine()
    a = extract_acousticness(y)
    assert 0.0 <= a <= 1.0

def test_acousticness_sine_gt_noise():
    """
    A pure sine wave (maximally tonal) should score higher acousticness
    than white noise (maximally flat spectrum).
    """
    sine = _sine(amplitude=0.5)
    noise = _noise(amplitude=0.5)
    assert extract_acousticness(sine) > extract_acousticness(noise)


# ---------------------------------------------------------------------------
# Tests: extract_instrumentalness
# ---------------------------------------------------------------------------

def test_instrumentalness_bounded():
    y = _sine()
    i = extract_instrumentalness(y, SR)
    assert 0.0 <= i <= 1.0


# ---------------------------------------------------------------------------
# Tests: AudioAnalyzer.analyze_track
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return AudioAnalyzer()

@pytest.fixture
def sine_wav(tmp_path):
    y = _sine(freq=440.0, amplitude=0.4)
    return _save(y, tmp_path / "sine.wav")

@pytest.fixture
def noise_wav(tmp_path):
    y = _noise(amplitude=0.6)
    return _save(y, tmp_path / "noise.wav")


def test_analyze_track_returns_all_keys(analyzer, sine_wav):
    result = analyzer.analyze_track(sine_wav)
    expected_keys = {
        "file_path", "energy", "energy_label", "tempo", "tempo_label",
        "danceability", "valence", "valence_label",
        "acousticness", "instrumentalness", "loudness",
    }
    assert expected_keys.issubset(result.keys())


def test_analyze_track_numeric_types(analyzer, sine_wav):
    result = analyzer.analyze_track(sine_wav)
    for key in ("energy", "tempo", "danceability", "valence",
                "acousticness", "instrumentalness", "loudness"):
        assert isinstance(result[key], float), f"{key} should be float"


def test_analyze_track_normalized_features_bounded(analyzer, sine_wav):
    result = analyzer.analyze_track(sine_wav)
    for key in ("energy", "danceability", "valence", "acousticness", "instrumentalness"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"


def test_analyze_track_tempo_non_negative(analyzer, sine_wav):
    # Pure sine waves have no transients; beat_track legitimately returns 0.
    assert analyzer.analyze_track(sine_wav)["tempo"] >= 0


def test_analyze_track_loudness_negative_dbfs(analyzer, sine_wav):
    assert analyzer.analyze_track(sine_wav)["loudness"] < 0


def test_analyze_track_energy_loud_gt_quiet(analyzer, tmp_path):
    loud_path = _save(_sine(amplitude=0.8), tmp_path / "loud.wav")
    quiet_path = _save(_sine(amplitude=0.05), tmp_path / "quiet.wav")
    loud_result = analyzer.analyze_track(loud_path)
    quiet_result = analyzer.analyze_track(quiet_path)
    assert loud_result["energy"] > quiet_result["energy"]


def test_analyze_track_labels_are_strings(analyzer, sine_wav):
    result = analyzer.analyze_track(sine_wav)
    for key in ("energy_label", "tempo_label", "valence_label"):
        assert isinstance(result[key], str) and len(result[key]) > 0


# ---------------------------------------------------------------------------
# Tests: AudioAnalyzer.analyze_batch
# ---------------------------------------------------------------------------

def test_analyze_batch_returns_correct_count(analyzer, tmp_path):
    paths = [
        _save(_sine(freq=440), tmp_path / "a.wav"),
        _save(_sine(freq=880), tmp_path / "b.wav"),
        _save(_noise(), tmp_path / "c.wav"),
    ]
    results = analyzer.analyze_batch(paths)
    assert len(results) == 3


def test_analyze_batch_all_have_keys(analyzer, tmp_path):
    paths = [
        _save(_sine(), tmp_path / "x.wav"),
        _save(_noise(), tmp_path / "y.wav"),
    ]
    for result in analyzer.analyze_batch(paths):
        assert "energy" in result
        assert "tempo" in result


# ---------------------------------------------------------------------------
# Tests: AudioAnalyzer.compute_artist_audio_profile
# ---------------------------------------------------------------------------

def test_compute_artist_audio_profile_averages(analyzer, tmp_path):
    paths = [
        _save(_sine(amplitude=0.1), tmp_path / "quiet.wav"),
        _save(_sine(amplitude=0.9), tmp_path / "loud.wav"),
    ]
    analyses = analyzer.analyze_batch(paths)
    profile = analyzer.compute_artist_audio_profile(analyses)

    quiet_energy = analyses[0]["energy"]
    loud_energy = analyses[1]["energy"]
    expected_avg = (quiet_energy + loud_energy) / 2
    assert abs(profile["energy"] - expected_avg) < 0.01


def test_compute_artist_audio_profile_track_count(analyzer, tmp_path):
    paths = [_save(_sine(), tmp_path / f"t{i}.wav") for i in range(4)]
    analyses = analyzer.analyze_batch(paths)
    profile = analyzer.compute_artist_audio_profile(analyses)
    assert profile["track_count"] == 4


def test_compute_artist_audio_profile_has_labels(analyzer, tmp_path):
    paths = [_save(_sine(), tmp_path / "t.wav")]
    analyses = analyzer.analyze_batch(paths)
    profile = analyzer.compute_artist_audio_profile(analyses)
    assert "energy_label" in profile
    assert "tempo_label" in profile
    assert "valence_label" in profile


def test_compute_artist_audio_profile_empty_returns_empty(analyzer):
    assert analyzer.compute_artist_audio_profile([]) == {}
