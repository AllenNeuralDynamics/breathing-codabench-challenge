# Run from the repo root (data lives in ./data/packaged):
#  uv run marimo edit baseline-cnn-tcn/notebooks/01_explore_data.py

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import subprocess
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from scoring.processing import (
        CANONICAL_BREATHING_SAMPLING_RATE,
        detect_breathing_events,
        filter_sniff_signal,
        resample_uniform,
    )

    return (
        CANONICAL_BREATHING_SAMPLING_RATE,
        Path,
        detect_breathing_events,
        filter_sniff_signal,
        mo,
        np,
        pd,
        plt,
        re,
        resample_uniform,
        subprocess,
    )


@app.cell
def _(mo):
    mo.md("""
    # Breathing Challenge — Clip Explorer

    Walk through the full data pipeline for a single packaged clip:

    1. **Load** — thermistor time-series and camera frame timestamps
    2. **Filter** — notch + high-pass + low-pass denoising
    3. **Resample** — onto the canonical 60 Hz scoring grid
    4. **Detect** — inhalation peaks and exhalation troughs
    5. **Align** — scrub the video frame-by-frame against the breathing trace
    """)
    return


@app.cell
def _(mo):
    data_dir_input = mo.ui.text(
        value="./data/packaged",
        label="Packaged data root  (`{root}/{split}/thermistor_*.parquet`)",
        full_width=True,
    )
    data_dir_input
    return (data_dir_input,)


@app.cell
def _(Path, data_dir_input, re):
    _root = Path(data_dir_input.value)
    _parquets = sorted(_root.rglob("thermistor_*.parquet"))

    _clip_map: dict[str, Path] = {}
    for _p in _parquets:
        _m = re.match(r"thermistor_(\d+)_part_(\d+)\.parquet", _p.name)
        if _m:
            _label = f"{_p.parent.name} / session {_m.group(1)}, part {_m.group(2)}"
            _clip_map[_label] = _p

    clip_map = _clip_map
    return (clip_map,)


@app.cell
def _(clip_map, mo):
    if not clip_map:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**No clips found.** Check the data directory — "
                    "it should contain `thermistor_N_part_M.parquet` files."
                ),
                kind="warn",
            ),
        )

    clip_selector = mo.ui.dropdown(
        options=list(clip_map.keys()),
        value=next(iter(clip_map)),
        label="Clip",
        full_width=True,
    )
    clip_selector
    return (clip_selector,)


@app.cell
def _(clip_map, clip_selector, pd, re):
    therm_path = clip_map[clip_selector.value]
    therm = pd.read_parquet(therm_path)

    # Discover all camera streams for this clip: video_{cam}_{N}_part_{M}.*
    _numeric = re.search(r"thermistor_(\d+_part_\d+)", therm_path.name)
    _suffix = _numeric.group(1) if _numeric else None
    _cams = {}
    if _suffix:
        for _mp4 in sorted(therm_path.parent.glob(f"video_*_{_suffix}.mp4")):
            _m = re.match(r"video_(.+)_\d+_part_\d+\.mp4", _mp4.name)
            if _m:
                _cams[_m.group(1)] = _mp4

    cam_map = _cams  # {slug: mp4_path}
    return cam_map, therm, therm_path


@app.cell
def _(cam_map, mo):
    if not cam_map:
        mo.stop(
            True,
            mo.callout(mo.md("No video files found next to this thermistor clip."), kind="warn"),
        )
    cam_selector = mo.ui.dropdown(
        options=list(cam_map.keys()),
        value=next(iter(cam_map)),
        label="Camera",
    )
    cam_selector
    return (cam_selector,)


@app.cell
def _(cam_map, cam_selector, pd):
    vid_mp4_path = cam_map[cam_selector.value]
    _parquet = vid_mp4_path.with_suffix(".parquet")
    vid_times = pd.read_parquet(_parquet) if _parquet.exists() else None
    vid_parquet_path = _parquet
    return vid_mp4_path, vid_parquet_path, vid_times


@app.cell
def _(mo, np, therm, therm_path, vid_mp4_path, vid_parquet_path, vid_times):
    _t = therm["Time"].to_numpy()
    _fs_approx = 1.0 / float(np.median(np.diff(_t))) if len(_t) > 1 else float("nan")

    mo.md(
        f"""
        ### Clip summary

        | File | Status |
        |---|---|
        | `{therm_path.name}` | ✅  {len(therm):,} samples · {_t[-1]:.1f} s · ~{_fs_approx:.0f} Hz |
        | `{vid_parquet_path.name}` | {"✅  " + str(len(vid_times)) + " frames" if vid_times is not None else "❌ not found"} |
        | `{vid_mp4_path.name}` | {"✅ found" if vid_mp4_path.exists() else "❌ not found"} |
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Processing pipeline

    The four steps below transform raw thermistor ADC samples into detected
    breathing events on the canonical 60 Hz scoring grid.
    """)
    return


@app.cell
def _(np, therm):
    t_raw = therm["Time"].to_numpy(dtype=float)
    v_raw = therm["Signal"].to_numpy(dtype=float)
    fs_raw = 1.0 / float(np.median(np.diff(t_raw)))
    return fs_raw, t_raw, v_raw


@app.cell
def _(mo):
    mo.md("""
    ### Step 1 — Raw ADC signal
    """)
    return


@app.cell
def _(fs_raw, mo, plt, t_raw, v_raw):
    _WIN = 30.0
    _mask = t_raw <= _WIN

    fig_raw, ax_raw = plt.subplots(figsize=(12, 2.8))
    ax_raw.plot(t_raw[_mask], v_raw[_mask], lw=0.6, color="#888888")
    ax_raw.set_title(f"Raw thermistor signal  (first {_WIN:.0f} s · fs ≈ {fs_raw:.0f} Hz)")
    ax_raw.set_xlabel("Time (s)")
    ax_raw.set_ylabel("ADC voltage")
    plt.tight_layout()
    mo.as_html(fig_raw)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 2 — `filter_sniff_signal`
    """)
    return


@app.cell
def _(filter_sniff_signal, fs_raw, v_raw):
    v_filtered = filter_sniff_signal(v_raw, fs_raw)
    return (v_filtered,)


@app.cell
def _(mo, plt, t_raw, v_filtered, v_raw):
    _WIN = 30.0
    _mask = t_raw <= _WIN

    fig_filt, axes_filt = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes_filt[0].plot(t_raw[_mask], v_raw[_mask], lw=0.6, color="#888888", label="raw")
    axes_filt[0].set_ylabel("ADC voltage")
    axes_filt[0].set_title("Raw vs filtered  (first 30 s)")
    axes_filt[0].legend(loc="upper right", fontsize=8)

    axes_filt[1].plot(t_raw[_mask], v_filtered[_mask], lw=0.8, color="#1f6aa5", label="filtered")
    axes_filt[1].set_xlabel("Time (s)")
    axes_filt[1].set_ylabel("ADC voltage")
    axes_filt[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    mo.as_html(fig_filt)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 3 — `resample_uniform`

    Linear interpolation onto the canonical **60 Hz** scoring grid.
    Both ground-truth and participant predictions land on this common grid
    before any metric is computed.
    """)
    return


@app.cell
def _(
    CANONICAL_BREATHING_SAMPLING_RATE,
    pd,
    resample_uniform,
    t_raw,
    v_filtered,
):
    _df_in = pd.DataFrame({"time": t_raw, "adc_voltage": v_filtered})
    _df_rs = resample_uniform(_df_in, target_fs=CANONICAL_BREATHING_SAMPLING_RATE)

    t_rs = _df_rs["time"].to_numpy(dtype=float)
    v_rs = _df_rs["adc_voltage"].to_numpy(dtype=float)
    fs_rs = CANONICAL_BREATHING_SAMPLING_RATE
    return fs_rs, t_rs, v_rs


@app.cell
def _(fs_rs, mo, plt, t_raw, t_rs, v_filtered, v_rs):
    _WIN = 5.0
    _mask_raw = t_raw <= _WIN
    _mask_rs = t_rs <= _WIN

    fig_rs, ax_rs = plt.subplots(figsize=(12, 2.8))
    ax_rs.plot(
        t_raw[_mask_raw], v_filtered[_mask_raw],
        lw=1.0, color="#aaaaaa", alpha=0.7,
        label=f"filtered (~{1.0 / float(t_raw[1] - t_raw[0]):.0f} Hz)",
    )
    ax_rs.plot(
        t_rs[_mask_rs], v_rs[_mask_rs],
        "o", ms=2.5, color="#e05c2a",
        label=f"resampled ({fs_rs:.0f} Hz)",
    )
    ax_rs.set_title(f"Resampling check  (first {_WIN:.0f} s)")
    ax_rs.set_xlabel("Time (s)")
    ax_rs.set_ylabel("ADC voltage")
    ax_rs.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    mo.as_html(fig_rs)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step 4 — `detect_breathing_events`

    Savitzky-Golay smoothing → `scipy.signal.find_peaks`.
    """)
    return


@app.cell
def _(mo):
    # ── Detection parameter sliders ───────────────────────────────────────────
    _DEFAULTS = dict(min_cycle_s=0.05, savgol_window_s=0.025, savgol_poly=3, prominence_frac=0.10)

    detect_sliders = mo.ui.dictionary({
        "min_cycle_s": mo.ui.slider(
            start=0.01, stop=0.50, step=0.01, value=_DEFAULTS["min_cycle_s"],
            label="min_cycle_s — min separation between events (s)",
            full_width=True, show_value=True,
        ),
        "savgol_window_s": mo.ui.slider(
            start=0.005, stop=0.100, step=0.005, value=_DEFAULTS["savgol_window_s"],
            label="savgol_window_s — Savitzky-Golay window (s)",
            full_width=True, show_value=True,
        ),
        "savgol_poly": mo.ui.slider(
            start=1, stop=6, step=1, value=_DEFAULTS["savgol_poly"],
            label="savgol_poly — polynomial order",
            full_width=True, show_value=True,
        ),
        "prominence_frac": mo.ui.slider(
            start=0.0, stop=0.50, step=0.01, value=_DEFAULTS["prominence_frac"],
            label="prominence_frac — min prominence (fraction of p2p)",
            full_width=True, show_value=True,
        ),
    })

    # Switch overrides slider values with hard-coded defaults.
    # Flip it on to restore defaults, flip it off to return to sliders.
    use_defaults = mo.ui.switch(label="Use defaults")
    return detect_sliders, use_defaults


@app.cell
def _(detect_breathing_events, detect_sliders, fs_rs, use_defaults, v_rs):
    _DEFAULTS = dict(min_cycle_s=0.05, savgol_window_s=0.025, savgol_poly=3, prominence_frac=0.10)
    _params = _DEFAULTS if use_defaults.value else detect_sliders.value
    inhale_peaks, exhale_troughs = detect_breathing_events(v_rs, fs_rs, **_params)
    return exhale_troughs, inhale_peaks


@app.cell
def _(
    detect_sliders,
    exhale_troughs,
    inhale_peaks,
    mo,
    plt,
    t_rs,
    use_defaults,
    v_rs,
):
    _WIN = 30.0
    _mask = t_rs <= _WIN
    _inh_in = inhale_peaks[t_rs[inhale_peaks] <= _WIN]
    _exh_in = exhale_troughs[t_rs[exhale_troughs] <= _WIN]

    fig_peaks, ax_peaks = plt.subplots(figsize=(12, 3))
    ax_peaks.plot(t_rs[_mask], v_rs[_mask], lw=0.8, color="#1f6aa5", label="60 Hz signal")
    ax_peaks.plot(
        t_rs[_inh_in], v_rs[_inh_in], "^", ms=5, color="#e05c2a", zorder=4,
        label=f"inhale peaks ({len(inhale_peaks)} total)",
    )
    ax_peaks.plot(
        t_rs[_exh_in], v_rs[_exh_in], "v", ms=5, color="#2eaa5e", zorder=4,
        label=f"exhale troughs ({len(exhale_troughs)} total)",
    )
    ax_peaks.set_title(f"Detected breathing events  (first {_WIN:.0f} s)")
    ax_peaks.set_xlabel("Time (s)")
    ax_peaks.set_ylabel("ADC voltage")
    ax_peaks.legend(loc="upper right", fontsize=8, ncol=3)
    plt.tight_layout()

    mo.vstack([
        mo.hstack(
            [mo.md("**Detection parameters**"), use_defaults],
            justify="start", gap=2,
        ),
        detect_sliders,
        mo.as_html(fig_peaks),
    ])
    return


@app.cell
def _(exhale_troughs, fs_rs, inhale_peaks, mo, np):
    _dt = 1.0 / fs_rs
    _ibi_inh = np.diff(inhale_peaks) * _dt
    _ibi_exh = np.diff(exhale_troughs) * _dt
    _br = 1.0 / float(np.mean(_ibi_inh)) if len(_ibi_inh) else float("nan")

    def _fmt(arr, fn):
        return f"{fn(arr):.3f} s" if len(arr) else "n/a"

    mo.md(
        f"""
        ### Breathing statistics

        | Metric | Inhale | Exhale |
        |---|---|---|
        | Events detected | {len(inhale_peaks):,} | {len(exhale_troughs):,} |
        | Mean breathing rate | **{_br:.2f} Hz** ({_br * 60:.0f} bpm) | — |
        | Median IBI | {_fmt(_ibi_inh, np.median)} | {_fmt(_ibi_exh, np.median)} |
        | p5 IBI | {_fmt(_ibi_inh, lambda a: np.percentile(a, 5))} | {_fmt(_ibi_exh, lambda a: np.percentile(a, 5))} |
        | p95 IBI | {_fmt(_ibi_inh, lambda a: np.percentile(a, 95))} | {_fmt(_ibi_exh, lambda a: np.percentile(a, 95))} |
        """
    )
    return


@app.cell
def _(exhale_troughs, fs_rs, inhale_peaks, mo, np, plt):
    _dt = 1.0 / fs_rs
    _ibi_inh = np.diff(inhale_peaks) * _dt
    _ibi_exh = np.diff(exhale_troughs) * _dt

    fig_ibi, (ax_inh, ax_exh) = plt.subplots(1, 2, figsize=(12, 3))
    for _ax, _ibi, _color, _lbl in [
        (ax_inh, _ibi_inh, "#e05c2a", "Inter-inhale interval"),
        (ax_exh, _ibi_exh, "#2eaa5e", "Inter-exhale interval"),
    ]:
        if len(_ibi) >= 2:
            _ax.hist(_ibi[_ibi <= 1.0], bins=40, range=(0, 1.0),
                     color=_color, alpha=0.7, edgecolor="none")
            _ax.axvline(float(np.median(_ibi)), color="black", lw=1.5,
                        label=f"median {np.median(_ibi):.3f} s")
            _ax.legend(fontsize=8)
        _ax.set_title(_lbl)
        _ax.set_xlabel("Interval (s)")
        _ax.set_ylabel("Count")
    plt.tight_layout()
    mo.as_html(fig_ibi)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Video + breathing alignment

    The thermistor and video parquets share a common `Time` axis (absolute
    clip-relative seconds from the same origin).  The slider scrubs through the clip:
    """)
    return


@app.cell
def _(mo, t_rs, vid_mp4_path):
    _t_min = float(t_rs[0]) if len(t_rs) else 0.0
    _t_max = float(t_rs[-1]) if len(t_rs) else 300.0

    if not vid_mp4_path.exists():
        mo.stop(True, mo.callout(mo.md("Video not found — widget unavailable."), kind="warn"))

    time_slider = mo.ui.slider(
        start=_t_min,
        stop=_t_max,
        step=0.5,
        value=round((_t_min + _t_max) / 2.0),
        # start/stop in the shared absolute time coordinate (same frame as vid_times["Time"])
        label="Clip time (s)",
        full_width=True,
        debounce=True,
    )
    return (time_slider,)


@app.cell
def _(
    exhale_troughs,
    inhale_peaks,
    mo,
    np,
    plt,
    subprocess,
    t_rs,
    time_slider,
    v_rs,
    vid_mp4_path,
    vid_times,
):
    t_cur = time_slider.value

    # ── Map slider time → frame index via the video parquet ──────────────────
    # vid_times["Time"] rows correspond 1-to-1 with MP4 frames: row i = frame i.
    # argmin gives the 0-based row index = frame number inside the clip MP4.
    _vid_t = vid_times["Time"].to_numpy(dtype=float)
    frame_idx = int(np.argmin(np.abs(_vid_t - t_cur)))
    _t_frame = float(_vid_t[frame_idx])

    # ── Extract frame by index using ffmpeg's select filter ───────────────────
    # select=eq(n\,N) picks the N-th decoded frame (0-based), bypassing any
    # PTS / seek ambiguity from the stream-copied MP4.
    _r = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(vid_mp4_path),
            "-vf", rf"select=eq(n\,{frame_idx})",
            "-vframes", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        capture_output=True, check=False,
    )
    _png = _r.stdout if _r.returncode == 0 and _r.stdout else None

    if _png:
        _frame_widget = mo.vstack([
            mo.image(src=_png, width=400),
            mo.md(f"frame **{frame_idx}** · t = {_t_frame:.3f} s"),
        ])
    else:
        _frame_widget = mo.callout(
            mo.md(
                f"**Frame extraction failed**\n\n"
                f"frame {frame_idx} · t = {_t_frame:.3f} s\n\n"
                f"```\n{_r.stderr.decode(errors='replace')[-400:]}\n```"
            ),
            kind="warn",
        )

    # ── Breathing trace: ±10 s window ────────────────────────────────────────
    _half = 10.0
    _t0 = max(0.0, t_cur - _half)
    _t1 = min(float(t_rs[-1]), t_cur + _half)
    _mask = (t_rs >= _t0) & (t_rs <= _t1)
    _inh_win = inhale_peaks[(t_rs[inhale_peaks] >= _t0) & (t_rs[inhale_peaks] <= _t1)]
    _exh_win = exhale_troughs[(t_rs[exhale_troughs] >= _t0) & (t_rs[exhale_troughs] <= _t1)]

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(t_rs[_mask], v_rs[_mask], lw=0.9, color="#1f6aa5")
    if len(_inh_win):
        _ax.plot(t_rs[_inh_win], v_rs[_inh_win], "^", ms=5,
                 color="#e05c2a", zorder=4, label="inhale")
    if len(_exh_win):
        _ax.plot(t_rs[_exh_win], v_rs[_exh_win], "v", ms=5,
                 color="#2eaa5e", zorder=4, label="exhale")
    _ax.axvline(t_cur, color="red", lw=1.5, ls="--", zorder=5, label=f"t = {t_cur:.1f} s")
    _ax.set_xlim(_t0, _t1)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Signal (60 Hz)")
    _ax.set_title("Breathing trace  (±10 s)")
    _ax.legend(loc="upper right", fontsize=8, ncol=3)
    plt.tight_layout()

    mo.vstack([
        time_slider,
        mo.hstack([_frame_widget, mo.as_html(_fig)], justify="start", gap=2),
    ])
    return


if __name__ == "__main__":
    app.run()
