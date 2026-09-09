"""Signal processing primitives for breathing data.

This module is the single authoritative source for all signal-processing
logic used by:

* the QC pipeline (``dataset_gen.qc.sniff``) — raw thermistor HARP data;
* the competition scoring programme (``breathing_metrics.metrics``) — clean
  predicted / ground-truth breathing signals.

All functions operate on plain NumPy arrays so they are usable without the
HARP I/O layer or any QC context.

Public API
----------
filter_sniff_signal(values, fs)
    0.2 Hz high-pass + 20 Hz low-pass pipeline.
detect_breathing_events(signal, fs, ...)
    Peak- and trough-detection on a filtered breathing signal.
detect_inhalation_events(signal, fs)
    Onset / offset detection for scoring (used by metrics.py).
process_sniff_signal(data, fs) -> SniffProcessingResult
    Full end-to-end pipeline for raw HARP Series data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_sniff_signal(values: np.ndarray, fs: float) -> np.ndarray:
    """Apply the standard SniffDetector denoising pipeline.

    Steps applied in order:

    1. 2nd-order Butterworth **high-pass** at 0.2 Hz — removes slow DC drift.
    2. 2nd-order Butterworth **low-pass** at 20 Hz — removes high-frequency
       noise above the physiological breathing band (≤ 20 Hz).

    Parameters
    ----------
    values:
        Raw ADC samples as ``float64``.  Must be **uniformly** sampled at
        *fs* Hz.
    fs:
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Filtered signal, same length as *values*.
    """
    b_hp, a_hp = butter(2, 0.2, "highpass", fs=fs)
    y = filtfilt(b_hp, a_hp, values)

    b_lp, a_lp = butter(2, 40.0, "lowpass", fs=fs)
    return filtfilt(b_lp, a_lp, y)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

CANONICAL_BREATHING_SAMPLING_RATE = 60.0
"""Breathing signal sampling rate (Hz) — the canonical scoring grid.

The thermistor rate varies across recordings, so both ground-truth and
predicted signals are resampled onto this fixed grid before any metric is
computed.  Every clip is therefore scored at an identical temporal
resolution regardless of its native rate.
"""

TIME_COLUMN = "Time"
"""Time column name in the clip parquet files (seconds).
"""

BREATHING_SIGNAL_COLUMN = "Signal"
"""Breathing-signal column name in the clip parquet files.
"""


def resample_uniform(
    thermistor: pd.DataFrame,
    *,
    target_fs: float = CANONICAL_BREATHING_SAMPLING_RATE,
) -> pd.DataFrame:
    """Resample a thermistor signal onto a uniform grid via linear interpolation.

    Used to bring ground-truth (thermistor, rate inferred from timestamps) and
    predicted signals onto a common time base before signal-level comparison.

    Parameters
    ----------
    thermistor:
        Input dataframe as extracted from a clip parquet.  Must contain the
        parquet schema columns ``Time`` (seconds, monotonically increasing)
        and ``Signal``.
    target_fs:
        Target sampling rate in Hz.  Default ``CANONICAL_BREATHING_SAMPLING_RATE``
        (the canonical scoring grid).

    Returns
    -------
    pd.DataFrame
        New dataframe with the same two columns, where ``Time`` is a
        uniform grid ``t[0], t[0]+1/target_fs, ...`` spanning the input range
        and ``Signal`` is linearly interpolated onto that grid.
    """
    t = thermistor[TIME_COLUMN].to_numpy(dtype=float)
    v = thermistor[BREATHING_SIGNAL_COLUMN].to_numpy(dtype=float)

    t_uniform = np.arange(t[0], t[-1], 1.0 / target_fs)
    interp_fn = interp1d(
        t, v, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return pd.DataFrame(
        {TIME_COLUMN: t_uniform, BREATHING_SIGNAL_COLUMN: interp_fn(t_uniform)}
    )


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

MAX_BREATHING_RATE_HZ = 20.0
"""Conservative upper bound on mouse breathing/sniffing rate (Hz).

Sets the minimum allowed separation between detected breathing events
(1 / 20 Hz = 50 ms).  Mouse respiration is ~2-4 Hz at rest and sniffing
reaches 4-12+ Hz; 20 Hz is a deliberately conservative cap.
"""

MIN_BREATH_CYCLE_S = 1.0 / MAX_BREATHING_RATE_HZ
"""Minimum separation between detected breathing events (seconds)."""

DEFAULT_PROMINENCE_FRAC = 0.10
"""Minimum prominence as a fraction of peak-to-peak signal range.

Events must stand out by at least this fraction of the total signal
amplitude.  Suppresses small-amplitude artifacts (e.g. transient dips
or plateaus during a breath transition) that would otherwise be detected
as spurious inhale/exhale pairs in close succession.
"""


def detect_breathing_events(
    signal: np.ndarray,
    fs: float,
    *,
    min_cycle_s: float = MIN_BREATH_CYCLE_S,
    savgol_window_s: float = 0.025,
    savgol_poly: int = 3,
    prominence_frac: float = DEFAULT_PROMINENCE_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect inhalation peaks and exhalation troughs via Savitzky-Golay smoothing.

    Strategy
    --------

    1. Smooth the filtered signal with a 25-ms Savitzky-Golay filter to remove
       sample-level noise without distorting peak positions.
    2. Find peaks and troughs on the smoothed signal with a minimum inter-event
       separation of *min_cycle_s* (guards against sub-physiological artefacts)
       and a minimum prominence of *prominence_frac* × peak-to-peak range
       (suppresses small transient artifacts that produce spurious close pairs).

    Parameters
    ----------
    signal:
        Filtered, zero-mean 1-D breathing signal (output of
        :func:`filter_sniff_signal`).
    fs:
        Sampling rate in Hz.
    min_cycle_s:
        Minimum separation between events (seconds).  Events closer than this
        are discarded as sub-physiological artefacts.  Default 0.05 s.
    savgol_window_s:
        Savitzky-Golay smoothing window in seconds.  Default 0.025 s (25 ms).
    savgol_poly:
        Polynomial order for the Savitzky-Golay filter.  Default 3.
    prominence_frac:
        Minimum prominence expressed as a fraction of the smoothed signal's
        total peak-to-peak range (0–1).  Default 0.10.  Setting to 0 disables
        the prominence gate.

    Returns
    -------
    inhale_peaks : np.ndarray of int
        Sample indices of detected inhalation peaks.
    exhale_troughs : np.ndarray of int
        Sample indices of detected exhalation troughs.
    """
    min_samp = max(1, int(min_cycle_s * fs))

    # ── Step 1: Savitzky-Golay smoothing ─────────────────────────────────────
    win_samp = max(3, int(savgol_window_s * fs))
    if win_samp % 2 == 0:  # window length must be odd
        win_samp += 1
    poly = min(savgol_poly, win_samp - 1)
    smoothed = np.asarray(
        savgol_filter(signal, window_length=win_samp, polyorder=poly), dtype=float
    )

    # ── Step 2: peak / trough detection (with prominence gate) ───────────────
    prominence = prominence_frac * float(smoothed.max() - smoothed.min())
    inhale_peaks, _ = find_peaks(smoothed, distance=min_samp, prominence=prominence)
    exhale_troughs, _ = find_peaks(-smoothed, distance=min_samp, prominence=prominence)

    return inhale_peaks.astype(int), exhale_troughs.astype(int)


def detect_inhalation_events(
    sig: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect inhalation onset and offset sample indices for scoring.

    This function is used by :mod:`breathing_metrics.metrics` to align
    predicted and ground-truth breathing events for onset / offset MAE and
    IBI distribution metrics.

    The signal is expected to be a **clean** breathing trace (ground-truth or
    participant prediction), not raw thermistor ADC data.

    Parameters
    ----------
    sig:
        1-D breathing signal array (uniformly sampled at *fs* Hz).
    fs:
        Sampling rate in Hz.

    Returns
    -------
    onsets : np.ndarray of int
        Sample indices of inhalation onsets (troughs preceding a peak).
    offsets : np.ndarray of int
        Sample indices of inhalation offsets (troughs following a peak).
    """
    min_distance = max(1, int(fs * MIN_BREATH_CYCLE_S))
    prominence = 0.1 * float(np.ptp(sig))

    peaks, _ = find_peaks(sig, distance=min_distance, prominence=prominence)
    troughs, _ = find_peaks(-sig, distance=min_distance, prominence=prominence)

    if len(peaks) == 0 or len(troughs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    onsets = troughs[troughs < peaks[-1]]
    offsets = troughs[troughs > peaks[0]]
    return onsets, offsets


# ---------------------------------------------------------------------------
# Full-pipeline result
# ---------------------------------------------------------------------------


@dataclass
class SniffProcessingResult:
    """Complete output of the SniffDetector signal-processing pipeline.

    Attributes
    ----------
    raw_timestamps:
        Original (possibly irregularly-spaced) HARP timestamps (seconds).
    raw_values:
        Original ADC values aligned to *raw_timestamps*.
    timestamps:
        Uniform time grid used for filtering (seconds).
    filtered:
        Filtered signal values aligned to *timestamps*.
    inhale_peaks:
        Indices into *timestamps* / *filtered* of detected inhalation peaks.
    exhale_troughs:
        Indices into *timestamps* / *filtered* of detected exhalation troughs.
    ibi_inhale_s:
        Inter-inhale intervals (seconds between successive inhalation peaks).
        Empty when fewer than 2 peaks are detected.
    ibi_exhale_s:
        Inter-exhale intervals (seconds between successive exhalation troughs).
        Empty when fewer than 2 troughs are detected.
    breathing_rate_hz:
        Mean breathing rate derived from inhale IBIs, or ``None`` when fewer
        than 2 peaks are present.
    n_peaks:
        Number of inhalation peaks detected.
    n_troughs:
        Number of exhalation troughs detected.
    """

    raw_timestamps: np.ndarray
    raw_values: np.ndarray
    timestamps: np.ndarray
    filtered: np.ndarray
    inhale_peaks: np.ndarray
    exhale_troughs: np.ndarray
    ibi_inhale_s: np.ndarray
    ibi_exhale_s: np.ndarray
    breathing_rate_hz: float | None
    n_peaks: int
    n_troughs: int


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def process_sniff_signal(data: pd.Series, fs: float) -> SniffProcessingResult:
    """Run the end-to-end SniffDetector processing pipeline.

    Steps:

    1. Re-sample *data* to a uniform time grid at *fs* Hz using linear
       interpolation (HARP clocks may produce slightly irregular intervals).
    2. Apply :func:`filter_sniff_signal` (notch + high-pass + low-pass).
    3. Detect inhalation peaks and exhalation troughs via
       :func:`detect_breathing_events`.
    4. Compute inter-event intervals and mean breathing rate.

    Parameters
    ----------
    data:
        Raw ``RawVoltage`` samples as a :class:`pandas.Series` indexed by HARP
        timestamps in seconds.
    fs:
        Declared sampling rate in Hz (from ``RawVoltageDispatchRate``).

    Returns
    -------
    SniffProcessingResult
        Contains the uniform time grid, filtered signal, detected event
        indices, inter-event intervals, and derived breathing rate.
    """
    raw_t = data.index.to_numpy(dtype=float)
    raw_v = data.values.astype(float)

    dt = 1.0 / fs
    t_uniform = np.arange(raw_t[0], raw_t[-1], dt)

    interp_fn = interp1d(
        raw_t,
        raw_v,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    y_uniform = interp_fn(t_uniform)

    y_filtered = filter_sniff_signal(y_uniform, fs)
    inhale_peaks, exhale_troughs = detect_breathing_events(y_filtered, fs)

    ibi_inhale = np.diff(inhale_peaks) * dt if len(inhale_peaks) >= 2 else np.array([])
    ibi_exhale = (
        np.diff(exhale_troughs) * dt if len(exhale_troughs) >= 2 else np.array([])
    )

    breathing_rate: float | None = (
        float(1.0 / np.mean(ibi_inhale)) if len(ibi_inhale) >= 1 else None
    )

    return SniffProcessingResult(
        raw_timestamps=raw_t,
        raw_values=raw_v,
        timestamps=t_uniform,
        filtered=y_filtered,
        inhale_peaks=inhale_peaks,
        exhale_troughs=exhale_troughs,
        ibi_inhale_s=ibi_inhale,
        ibi_exhale_s=ibi_exhale,
        breathing_rate_hz=breathing_rate,
        n_peaks=len(inhale_peaks),
        n_troughs=len(exhale_troughs),
    )
