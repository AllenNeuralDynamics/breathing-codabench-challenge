"""Utility helpers for the WASM / local Marimo notebook.

Imported by 01_explore_data_wasm.py in local-edit mode.
In WASM (Pyodide) mode the notebook falls back to defining these inline,
so this file is for human readers and local development convenience only.
"""

import asyncio
import io
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

# ── Constants ──────────────────────────────────────────────────────────────────

CANONICAL_BREATHING_SAMPLING_RATE = 60.0
"""Canonical scoring grid (Hz) — both GT and predictions land here."""

# ── Async HTTP helpers ─────────────────────────────────────────────────────────
# Prefer pyodide.http (WASM); fall back to asyncio+urllib (local Marimo).


async def http_get_bytes(url: str) -> bytes:
    """Fetch *url* and return the raw bytes (works in Pyodide and CPython)."""
    try:
        import pyodide.http as _ph
        resp = await _ph.pyfetch(url)
        return await resp.bytes()
    except ImportError:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(url).read()
        )


async def http_get_str(url: str) -> str:
    """Fetch *url* and return the decoded text."""
    return (await http_get_bytes(url)).decode()


# ── Signal processing ──────────────────────────────────────────────────────────
# Inlined from scoring.processing to avoid the dtaidistance C-extension dep
# (which is not available in Pyodide).


def filter_sniff_signal(values: np.ndarray, fs: float) -> np.ndarray:
    """0.2 Hz high-pass + 40 Hz low-pass Butterworth pipeline."""
    b_hp, a_hp = butter(2, 0.2, "highpass", fs=fs)
    y = filtfilt(b_hp, a_hp, values)
    b_lp, a_lp = butter(2, 40.0, "lowpass", fs=fs)
    return filtfilt(b_lp, a_lp, y)


def resample_uniform(
    thermistor: pd.DataFrame,
    *,
    target_fs: float = CANONICAL_BREATHING_SAMPLING_RATE,
) -> pd.DataFrame:
    """Resample a thermistor DataFrame onto a uniform grid via linear interpolation."""
    t = thermistor["time"].to_numpy(dtype=float)
    v = thermistor["adc_voltage"].to_numpy(dtype=float)
    t_uniform = np.arange(t[0], t[-1], 1.0 / target_fs)
    fn = interp1d(t, v, kind="linear", bounds_error=False, fill_value="extrapolate")
    return pd.DataFrame({"time": t_uniform, "adc_voltage": fn(t_uniform)})


def detect_breathing_events(
    signal: np.ndarray,
    fs: float,
    *,
    min_cycle_s: float = 0.05,
    savgol_window_s: float = 0.025,
    savgol_poly: int = 3,
    prominence_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect inhalation peaks and exhalation troughs via Savitzky-Golay + find_peaks."""
    min_samp = max(1, int(min_cycle_s * fs))
    win_samp = max(3, int(savgol_window_s * fs))
    if win_samp % 2 == 0:
        win_samp += 1
    poly = min(savgol_poly, win_samp - 1)
    smoothed = np.asarray(
        savgol_filter(signal, window_length=win_samp, polyorder=poly), dtype=float
    )
    prominence = prominence_frac * float(smoothed.max() - smoothed.min())
    inhale_peaks, _ = find_peaks(smoothed, distance=min_samp, prominence=prominence)
    exhale_troughs, _ = find_peaks(-smoothed, distance=min_samp, prominence=prominence)
    return inhale_peaks.astype(int), exhale_troughs.astype(int)
