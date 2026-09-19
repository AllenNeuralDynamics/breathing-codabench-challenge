import numpy as np

from scoring.processing import detect_inhalation_events


def test_detect_inhalation_events_maps_peaks_and_troughs() -> None:
    fs = 100.0
    time = np.arange(200) / fs
    signal = np.sin(2 * np.pi * time)

    inhale_onsets, exhale_onsets = detect_inhalation_events(signal, fs)

    np.testing.assert_array_equal(inhale_onsets, np.array([25, 125]))
    np.testing.assert_array_equal(exhale_onsets, np.array([75, 175]))


def test_detect_inhalation_events_keeps_inhale_without_exhale() -> None:
    inhale_onsets, exhale_onsets = detect_inhalation_events(
        np.array([0.0, 1.0, 0.0]), fs=100.0
    )

    np.testing.assert_array_equal(inhale_onsets, np.array([1]))
    assert exhale_onsets.size == 0


def test_detect_inhalation_events_keeps_exhale_without_inhale() -> None:
    inhale_onsets, exhale_onsets = detect_inhalation_events(
        np.array([0.0, -1.0, 0.0]), fs=100.0
    )

    assert inhale_onsets.size == 0
    np.testing.assert_array_equal(exhale_onsets, np.array([1]))
