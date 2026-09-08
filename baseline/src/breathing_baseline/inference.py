"""
Baseline breathing inference from video.

Algorithm: dense optical flow (Farneback) averaged over the full frame.
The mean flow magnitude across frames is taken as a proxy respiratory signal
and then resampled to the target sampling rate.

This is intentionally simple — replace with your own approach.
"""

import cv2
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scoring.processing import ADC_VOLTAGE_COLUMN, TIME_COLUMN


def infer_breathing(video_path: str, target_fs: float = 1000.0) -> pd.DataFrame:
    """
    Infer a breathing signal from a video file.

    Parameters
    ----------
    video_path:
        Path to an MP4 video clip.
    target_fs:
        Desired output sampling rate in Hz. The raw optical-flow signal is
        computed at video frame rate and then resampled to `target_fs`.

    Returns
    -------
    DataFrame with columns ``time`` (float64) and ``adc_voltage`` (float64).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    flow_magnitudes: list[float] = []
    prev_gray = None

    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
            flow_magnitudes.append(mag)
        prev_gray = gray

    cap.release()

    if not flow_magnitudes:
        raise ValueError(f"No frames extracted from {video_path}")

    raw = np.array(flow_magnitudes, dtype=np.float64)
    raw_times = np.arange(len(raw)) / fps

    # Resample to target_fs
    n_out = round(raw_times[-1] * target_fs)
    out_times = np.linspace(0.0, raw_times[-1], n_out)
    resampled = np.interp(out_times, raw_times, raw)

    # Band-pass filter to typical breathing range (0.5–5 Hz)
    lo, hi = 0.5, 5.0
    b, a = sp_signal.butter(
        4, [lo / (target_fs / 2), hi / (target_fs / 2)], btype="band"
    )
    filtered = sp_signal.filtfilt(b, a, resampled)

    return pd.DataFrame({TIME_COLUMN: out_times, ADC_VOLTAGE_COLUMN: filtered})
