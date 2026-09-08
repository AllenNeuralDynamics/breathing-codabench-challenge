# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "scipy>=1.11",
#   "pyarrow>=14.0",
#   "matplotlib>=3.8",
# ]
# ///
#
# ── Build a self-contained HTML that runs entirely in the browser ────────────
#   uv run marimo export html-wasm \
#       baseline-cnn-tcn/notebooks/01_explore_data_wasm.py \
#       -o docs/explore.html
#
# ── Or run locally (also reads from S3 over HTTPS) ──────────────────────────
#   uv run marimo edit baseline-cnn-tcn/notebooks/01_explore_data_wasm.py
#
# ── S3 prerequisite ─────────────────────────────────────────────────────────
#   The aind-scratch-data bucket must have a CORS policy that allows GET/HEAD
#   from the page origin (or "*").  Example CORS JSON:
#     [{"AllowedHeaders":["*"],"AllowedMethods":["GET","HEAD"],
#       "AllowedOrigins":["*"],"ExposeHeaders":["Content-Length"]}]
#   Without it, every browser fetch (listing + parquets + video) will be
#   blocked by the browser's same-origin policy.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


# ── Cell 1: imports + inlined signal-processing + HTTP helpers ───────────────
@app.cell
async def _():
    import asyncio
    import io
    import json
    import re
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

    # ── Async HTTP helpers ────────────────────────────────────────────────────
    # Prefer pyodide.http (WASM); fall back to asyncio+urllib (local Marimo).

    async def http_get_bytes(url: str) -> bytes:
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
        return (await http_get_bytes(url)).decode()

    # ── Inlined signal-processing (avoids dtaidistance dep; pure numpy/scipy) ─
    CANONICAL_BREATHING_SAMPLING_RATE = 60.0

    def filter_sniff_signal(values: np.ndarray, fs: float) -> np.ndarray:
        b_hp, a_hp = butter(2, 0.2, "highpass", fs=fs)
        y = filtfilt(b_hp, a_hp, values)
        b_lp, a_lp = butter(2, 40.0, "lowpass", fs=fs)
        return filtfilt(b_lp, a_lp, y)

    def resample_uniform(
        thermistor: pd.DataFrame,
        *,
        target_fs: float = CANONICAL_BREATHING_SAMPLING_RATE,
    ) -> pd.DataFrame:
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

    return (
        CANONICAL_BREATHING_SAMPLING_RATE,
        ET,
        detect_breathing_events,
        filter_sniff_signal,
        http_get_bytes,
        http_get_str,
        io,
        json,
        mo,
        np,
        pd,
        plt,
        pq,
        re,
        resample_uniform,
        urllib,
    )


# ── Cell 2: intro ─────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    # Breathing Challenge — Clip Explorer  *(S3 / WASM edition)*

    Runs entirely in the browser — **no local data download**.  Data is fetched
    on demand from the public `aind-scratch-data` S3 bucket over HTTPS.

    1. **List** — discover thermistor clips from an S3 prefix (paginated)
    2. **Load** — stream one parquet into memory via `pyarrow`
    3. **Filter** — notch + high-pass + low-pass denoising
    4. **Resample** — onto the canonical 60 Hz scoring grid
    5. **Detect** — inhalation peaks and exhalation troughs
    6. **Align** — scrub the video frame-by-frame
       *(HTML5 `<video>` streams the MP4 from S3; no ffmpeg needed)*
    """)
    return


# ── Cell 3: S3 prefix input ───────────────────────────────────────────────────
@app.cell
def _(mo):
    s3_prefix_input = mo.ui.text(
        value=(
            "vr-foraging/codabench-breathing-challenge/"
            "9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/train"
        ),
        label="S3 prefix  (bucket: aind-scratch-data)",
        full_width=True,
    )
    s3_prefix_input
    return (s3_prefix_input,)


# ── Cell 4: paginated S3 listing ──────────────────────────────────────────────
@app.cell
async def _(ET, http_get_str, mo, re, s3_prefix_input, urllib):
    _BUCKET = "aind-scratch-data"
    _prefix = s3_prefix_input.value.strip("/") + "/"

    _all_keys: list[str] = []
    _token: str | None = None

    while True:
        _url = (
            f"https://{_BUCKET}.s3.amazonaws.com/"
            f"?list-type=2&prefix={urllib.parse.quote(_prefix, safe='/:')}"
        )
        if _token:
            _url += f"&continuation-token={urllib.parse.quote(_token, safe='')}"

        try:
            _xml = await http_get_str(_url)
        except Exception as _exc:
            mo.stop(
                True,
                mo.callout(
                    mo.md(
                        f"**S3 listing failed:** `{_exc}`\n\n"
                        "Check the prefix and that the bucket has a CORS policy allowing GET."
                    ),
                    kind="danger",
                ),
            )

        _root = ET.fromstring(_xml)
        _ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        _all_keys += [el.text for el in _root.findall(".//s3:Key", _ns)]

        _trunc = _root.find("s3:IsTruncated", _ns)
        if _trunc is None or _trunc.text.lower() != "true":
            break
        _next = _root.find("s3:NextContinuationToken", _ns)
        if _next is None:
            break
        _token = _next.text

    # Build clip map: label → S3 key (thermistor parquet)
    _clip_map: dict[str, str] = {}
    for _k in _all_keys:
        _m = re.match(r".*/thermistor_(\d+)_part_(\d+)\.parquet$", _k)
        if _m:
            _split = _k.rsplit("/", 2)[-2] if _k.count("/") >= 2 else ""
            _label = f"{_split} / session {_m.group(1)}, part {_m.group(2)}"
            _clip_map[_label] = _k

    clip_map = _clip_map
    all_keys = _all_keys
    S3_BUCKET = _BUCKET
    return S3_BUCKET, all_keys, clip_map


# ── Cell 5: clip selector ─────────────────────────────────────────────────────
@app.cell
def _(clip_map, mo):
    if not clip_map:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**No clips found.** Check the S3 prefix — "
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


# ── Cell 6: fetch thermistor parquet + discover camera streams ────────────────
@app.cell
async def _(S3_BUCKET, all_keys, clip_map, clip_selector, http_get_bytes, io, mo, pq, re):
    _key = clip_map[clip_selector.value]
    _url = f"https://{S3_BUCKET}.s3.amazonaws.com/{_key}"

    try:
        _data = await http_get_bytes(_url)
        therm = pq.read_table(io.BytesIO(_data)).to_pandas()
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(mo.md(f"**Failed to fetch thermistor parquet:** `{_exc}`"), kind="danger"),
        )

    therm_key = _key

    # Discover camera MP4s that share the same N_part_M suffix
    _numeric = re.search(r"thermistor_(\d+_part_\d+)", _key)
    _suffix = _numeric.group(1) if _numeric else None
    _parent = _key.rsplit("/", 1)[0]

    _cams: dict[str, str] = {}
    if _suffix:
        for _k in all_keys:
            _m = re.match(rf".*video_(.+)_{re.escape(_suffix)}\.mp4$", _k)
            if _m and _k.startswith(_parent):
                _cams[_m.group(1)] = _k

    cam_map = _cams
    return cam_map, therm, therm_key


# ── Cell 7: camera selector ───────────────────────────────────────────────────
@app.cell
def _(cam_map, mo):
    if not cam_map:
        mo.stop(True, mo.callout(mo.md("No video files found for this clip."), kind="warn"))
    cam_selector = mo.ui.dropdown(
        options=list(cam_map.keys()),
        value=next(iter(cam_map)),
        label="Camera",
    )
    cam_selector
    return (cam_selector,)


# ── Cell 8: fetch video timestamp parquet ─────────────────────────────────────
@app.cell
async def _(S3_BUCKET, cam_map, cam_selector, http_get_bytes, io, pq):
    _mp4_key = cam_map[cam_selector.value]
    _parquet_key = _mp4_key.replace(".mp4", ".parquet")

    try:
        _data = await http_get_bytes(
            f"https://{S3_BUCKET}.s3.amazonaws.com/{_parquet_key}"
        )
        vid_times = pq.read_table(io.BytesIO(_data)).to_pandas()
    except Exception:
        vid_times = None

    vid_mp4_key = _mp4_key
    return vid_mp4_key, vid_times


# ── Cell 9: clip summary ──────────────────────────────────────────────────────
@app.cell
def _(S3_BUCKET, mo, np, therm, therm_key, vid_mp4_key, vid_times):
    _t = therm["Time"].to_numpy()
    _fs_approx = 1.0 / float(np.median(np.diff(_t))) if len(_t) > 1 else float("nan")
    _vt_status = (
        f"✅  {len(vid_times):,} frames" if vid_times is not None else "❌ not found"
    )
    mo.md(
        f"""
        ### Clip summary

        | File | Status |
        |---|---|
        | `{therm_key.split('/')[-1]}` | ✅  {len(therm):,} samples · {_t[-1]:.1f} s · ~{_fs_approx:.0f} Hz |
        | `{vid_mp4_key.replace('.mp4', '.parquet').split('/')[-1]}` | {_vt_status} |
        | `{vid_mp4_key.split('/')[-1]}` | ✅ streaming from `s3://{S3_BUCKET}/` |
        """
    )
    return


# ── Cell 10 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ---
    ## Processing pipeline

    The four steps below transform raw thermistor ADC samples into detected
    breathing events on the canonical 60 Hz scoring grid.
    """)
    return


# ── Cell 11: extract raw arrays ───────────────────────────────────────────────
@app.cell
def _(np, therm):
    t_raw = therm["Time"].to_numpy(dtype=float)
    v_raw = therm["Signal"].to_numpy(dtype=float)
    fs_raw = 1.0 / float(np.median(np.diff(t_raw)))
    return fs_raw, t_raw, v_raw


# ── Cell 12 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("### Step 1 — Raw ADC signal")
    return


# ── Cell 13: raw signal plot ──────────────────────────────────────────────────
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


# ── Cell 14 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("### Step 2 — `filter_sniff_signal`")
    return


# ── Cell 15: filter ───────────────────────────────────────────────────────────
@app.cell
def _(filter_sniff_signal, fs_raw, v_raw):
    v_filtered = filter_sniff_signal(v_raw, fs_raw)
    return (v_filtered,)


# ── Cell 16: filter plot ──────────────────────────────────────────────────────
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


# ── Cell 17 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ### Step 3 — `resample_uniform`

    Linear interpolation onto the canonical **60 Hz** scoring grid.
    Both ground-truth and participant predictions land on this common grid
    before any metric is computed.
    """)
    return


# ── Cell 18: resample ─────────────────────────────────────────────────────────
@app.cell
def _(CANONICAL_BREATHING_SAMPLING_RATE, pd, resample_uniform, t_raw, v_filtered):
    _df_in = pd.DataFrame({"time": t_raw, "adc_voltage": v_filtered})
    _df_rs = resample_uniform(_df_in, target_fs=CANONICAL_BREATHING_SAMPLING_RATE)
    t_rs = _df_rs["time"].to_numpy(dtype=float)
    v_rs = _df_rs["adc_voltage"].to_numpy(dtype=float)
    fs_rs = CANONICAL_BREATHING_SAMPLING_RATE
    return fs_rs, t_rs, v_rs


# ── Cell 19: resample plot ────────────────────────────────────────────────────
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


# ── Cell 20 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ### Step 4 — `detect_breathing_events`

    Savitzky-Golay smoothing → `scipy.signal.find_peaks`.
    """)
    return


# ── Cell 21: detection sliders ────────────────────────────────────────────────
@app.cell
def _(mo):
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
    use_defaults = mo.ui.switch(label="Use defaults")
    return detect_sliders, use_defaults


# ── Cell 22: detect ───────────────────────────────────────────────────────────
@app.cell
def _(detect_breathing_events, detect_sliders, fs_rs, use_defaults, v_rs):
    _DEFAULTS = dict(min_cycle_s=0.05, savgol_window_s=0.025, savgol_poly=3, prominence_frac=0.10)
    _params = _DEFAULTS if use_defaults.value else detect_sliders.value
    inhale_peaks, exhale_troughs = detect_breathing_events(v_rs, fs_rs, **_params)
    return exhale_troughs, inhale_peaks


# ── Cell 23: peaks plot ───────────────────────────────────────────────────────
@app.cell
def _(detect_sliders, exhale_troughs, inhale_peaks, mo, plt, t_rs, use_defaults, v_rs):
    _WIN = 30.0
    _mask = t_rs <= _WIN
    _inh_in = inhale_peaks[t_rs[inhale_peaks] <= _WIN]
    _exh_in = exhale_troughs[t_rs[exhale_troughs] <= _WIN]
    fig_peaks, ax_peaks = plt.subplots(figsize=(12, 3))
    ax_peaks.plot(t_rs[_mask], v_rs[_mask], lw=0.8, color="#1f6aa5", label="60 Hz signal")
    ax_peaks.plot(t_rs[_inh_in], v_rs[_inh_in], "^", ms=5, color="#e05c2a", zorder=4,
                  label=f"inhale peaks ({len(inhale_peaks)} total)")
    ax_peaks.plot(t_rs[_exh_in], v_rs[_exh_in], "v", ms=5, color="#2eaa5e", zorder=4,
                  label=f"exhale troughs ({len(exhale_troughs)} total)")
    ax_peaks.set_title(f"Detected breathing events  (first {_WIN:.0f} s)")
    ax_peaks.set_xlabel("Time (s)")
    ax_peaks.set_ylabel("ADC voltage")
    ax_peaks.legend(loc="upper right", fontsize=8, ncol=3)
    plt.tight_layout()
    mo.vstack([
        mo.hstack([mo.md("**Detection parameters**"), use_defaults], justify="start", gap=2),
        detect_sliders,
        mo.as_html(fig_peaks),
    ])
    return


# ── Cell 24: breathing statistics ────────────────────────────────────────────
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


# ── Cell 25: IBI histograms ───────────────────────────────────────────────────
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


# ── Cell 26 ───────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    mo.md("""
    ---
    ## Video + breathing alignment

    The thermistor and video share a common `Time` axis.  The MP4 streams from
    S3 via a hidden `<video>` element — the browser handles buffering and seeking,
    so the file is never fully downloaded.  Each slider tick seeks to the exact
    frame timestamp reported by the video parquet.
    """)
    return


# ── Cell 27: time slider ──────────────────────────────────────────────────────
@app.cell
def _(mo, t_rs, vid_times):
    if vid_times is None:
        mo.stop(True, mo.callout(mo.md("Video timestamps parquet not found — widget unavailable."), kind="warn"))
    _t_min = float(t_rs[0]) if len(t_rs) else 0.0
    _t_max = float(t_rs[-1]) if len(t_rs) else 300.0
    time_slider = mo.ui.slider(
        start=_t_min,
        stop=_t_max,
        step=0.5,
        value=round((_t_min + _t_max) / 2.0),
        label="Clip time (s)",
        full_width=True,
        debounce=True,
    )
    return (time_slider,)


# ── Cell 28: video + breathing trace ─────────────────────────────────────────
@app.cell
def _(
    S3_BUCKET,
    exhale_troughs,
    inhale_peaks,
    json,
    mo,
    np,
    plt,
    t_rs,
    time_slider,
    v_rs,
    vid_mp4_key,
    vid_times,
):
    t_cur = time_slider.value

    # Map slider time → nearest frame timestamp
    _vid_t = vid_times["Time"].to_numpy(dtype=float)
    frame_idx = int(np.argmin(np.abs(_vid_t - t_cur)))
    t_frame = float(_vid_t[frame_idx])

    # S3 HTTPS URL — the browser streams the MP4 directly
    _mp4_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{vid_mp4_key}"
    _mp4_url_js = json.dumps(_mp4_url)   # safely escaped for inline JS

    # ── Browser-native video frame extractor ─────────────────────────────────
    # Strategy: keep ONE hidden <video> element in document.body (keyed by src
    # URL so it is replaced when the clip changes).  Each reactive update just
    # calls video.currentTime = T; the 'seeked' event fires once the frame is
    # ready and repaints the visible <canvas>.  The video is never re-created
    # while the same clip is selected, so the browser keeps its buffer warm.
    _video_widget = mo.Html(f"""
    <div style="display:flex;flex-direction:column;align-items:flex-start;gap:4px">
      <canvas id="breath-frame-canvas"
              style="border:1px solid #ccc;border-radius:4px;background:#111;max-width:420px">
      </canvas>
      <span style="font-size:0.8em;color:#888">
        frame&nbsp;{frame_idx} &middot; t&nbsp;=&nbsp;{t_frame:.3f}&nbsp;s
      </span>
    </div>
    <script>
    (function () {{
      var SRC = {_mp4_url_js};
      var T   = {t_frame};

      // Reuse or create a persistent hidden <video> (one per clip URL)
      var vid = document.getElementById('_breath_vid_player');
      if (!vid || vid.dataset.src !== SRC) {{
        if (vid) vid.remove();
        vid                = document.createElement('video');
        vid.id             = '_breath_vid_player';
        vid.dataset.src    = SRC;
        vid.crossOrigin    = 'anonymous';
        vid.preload        = 'auto';
        vid.style.display  = 'none';
        vid.src            = SRC;
        document.body.appendChild(vid);
      }}

      var canvas = document.getElementById('breath-frame-canvas');
      var ctx    = canvas.getContext('2d');

      function drawFrame() {{
        var W = vid.videoWidth  || 640;
        var H = vid.videoHeight || 480;
        var maxW = 420;
        canvas.width  = Math.min(W, maxW);
        canvas.height = Math.round(H * Math.min(W, maxW) / W);
        ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
      }}

      // Only seek if we're not already there (avoids rebuffering on every redraw)
      if (Math.abs(vid.currentTime - T) > 0.04) {{
        vid.addEventListener('seeked', drawFrame, {{ once: true }});
        vid.currentTime = T;
      }} else {{
        // Already at the right position — repaint immediately
        if (vid.readyState >= 2) {{ drawFrame(); }}
        else {{ vid.addEventListener('loadeddata', drawFrame, {{ once: true }}); }}
      }}
    }})();
    </script>
    """)

    # ── Breathing trace ±10 s ─────────────────────────────────────────────────
    _half = 10.0
    _t0 = max(0.0, t_cur - _half)
    _t1 = min(float(t_rs[-1]), t_cur + _half)
    _mask = (t_rs >= _t0) & (t_rs <= _t1)
    _inh_win = inhale_peaks[(t_rs[inhale_peaks] >= _t0) & (t_rs[inhale_peaks] <= _t1)]
    _exh_win = exhale_troughs[(t_rs[exhale_troughs] >= _t0) & (t_rs[exhale_troughs] <= _t1)]

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(t_rs[_mask], v_rs[_mask], lw=0.9, color="#1f6aa5")
    if len(_inh_win):
        _ax.plot(t_rs[_inh_win], v_rs[_inh_win], "^", ms=5, color="#e05c2a", zorder=4, label="inhale")
    if len(_exh_win):
        _ax.plot(t_rs[_exh_win], v_rs[_exh_win], "v", ms=5, color="#2eaa5e", zorder=4, label="exhale")
    _ax.axvline(t_cur, color="red", lw=1.5, ls="--", zorder=5, label=f"t = {t_cur:.1f} s")
    _ax.set_xlim(_t0, _t1)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Signal (60 Hz)")
    _ax.set_title("Breathing trace  (±10 s)")
    _ax.legend(loc="upper right", fontsize=8, ncol=3)
    plt.tight_layout()

    mo.vstack([
        time_slider,
        mo.hstack([_video_widget, mo.as_html(_fig)], justify="start", gap=2),
    ])
    return


if __name__ == "__main__":
    app.run()
