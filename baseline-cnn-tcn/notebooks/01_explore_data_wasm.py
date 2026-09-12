# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.0",
#   "scipy>=1.11",
#   "pyarrow>=14.0",
#   "matplotlib>=3.8",
#   "marimo",
#   "anywidget>=0.9",
#   "traitlets>=5",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


# ── Cell 1: utility imports (hidden by default — expand to inspect) ───────────
@app.cell(hide_code=True)
async def _():
    import asyncio
    import html
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

    # In local mode, import from the sibling utils.py.
    # In WASM (Pyodide), utils.py is not bundled, so we fall back to inline defs.
    try:
        from utils import (  # noqa: F401
            CANONICAL_BREATHING_SAMPLING_RATE,
            detect_breathing_events,
            filter_sniff_signal,
            http_get_bytes,
            http_get_str,
            resample_uniform,
        )
    except (ModuleNotFoundError, ImportError):
        # ── WASM fallback: define everything inline ───────────────────────────

        CANONICAL_BREATHING_SAMPLING_RATE = 60.0

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
        html,
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
    # 🐭 Breathing Challenge — Clip Explorer

    Data streams from the public `aind-scratch-data` S3 bucket on demand.

    > **Want to see or edit the code?**
    > [Clone the repo](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge) and run `marimo edit baseline-cnn-tcn/notebooks/01_explore_data_wasm.py` —
    """)
    return


# ── Cell 3: S3 prefix (hardcoded) ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _():
    # Pyodide can't import workspace packages, so this must be kept in sync by
    # hand with scoring.data.S3_BUCKET / S3_PUBLIC_PREFIX.
    import types
    s3_prefix_input = types.SimpleNamespace(
        value=(
            "vr-foraging/codabench-breathing-challenge/"
            "3fd049f3b2d5bb39409611187918ac41ce1f8b0a0d8d113a3526e5cf5a2ebc08/public/train"
        )
    )
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

    # Build clip map: label → S3 key (thermistor parquet), plus the camera
    # MP4s sharing that clip's directory/session/part suffix.
    _videos_by_clip: dict[tuple[str, str, str], dict[str, str]] = {}
    for _k in _all_keys:
        _m = re.match(r"(.*/)?video_(.+)_(\d+)_part_(\d+)\.mp4$", _k)
        if _m:
            _parent = _m.group(1) or ""
            _camera = _m.group(2)
            # Thermistors remain under ``public/train``.  Their streaming-
            # optimized companions live in the separate remuxed mirror.
            _remuxed_key = _k.replace(
                "/public/train/", "/videos_remuxed/public/train/", 1
            )
            _videos_by_clip.setdefault((_parent, _m.group(3), _m.group(4)), {})[
                _camera
            ] = _remuxed_key

    _clip_map: dict[str, str] = {}
    _video_map: dict[str, dict[str, str]] = {}
    for _k in _all_keys:
        _m = re.match(r"(.*/)?thermistor_(\d+)_part_(\d+)\.parquet$", _k)
        if _m:
            _split = _k.rsplit("/", 2)[-2] if _k.count("/") >= 2 else ""
            _label = f"{_split} / session {_m.group(2)}, part {_m.group(3)}"
            _clip_map[_label] = _k
            _video_map[_k] = _videos_by_clip.get(
                (_m.group(1) or "", _m.group(2), _m.group(3)), {}
            )

    clip_map = _clip_map
    video_map = _video_map
    return clip_map, video_map


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


# ── Cell 6: find paired video streams for this clip ───────────────────────────
@app.cell
def _(clip_map, clip_selector, mo, video_map):
    clip_videos = video_map.get(clip_map[clip_selector.value], {})
    if not clip_videos:
        _video_status = mo.callout(
            mo.md("No MP4 video files were found alongside this thermistor clip."),
            kind="warn",
        )
    else:
        _video_status = mo.md("### Paired camera streams")
    _video_status
    return (clip_videos,)


# ── Cell 7: synchronized paired native-video widget ──────────────────────────
@app.cell
def _():
    import anywidget
    import traitlets

    class PairedCameras(anywidget.AnyWidget):
        _esm = r"""
        function render({ model, el }) {
          const root = document.createElement("div");
          root.style.cssText = "display:flex; flex-wrap:wrap; gap:16px";
          const videos = model.get("sources").map(({ label, src }) => {
            const panel = document.createElement("div");
            panel.style.cssText = "display:grid; gap:6px; min-width:0";
            const title = document.createElement("strong");
            title.textContent = `${label} camera`;
            const video = document.createElement("video");
            video.controls = true;
            video.playsInline = true;
            video.preload = "metadata";
            video.style.cssText = "width:360px; max-width:100%; background:#111";
            video.src = src;
            panel.append(title, video);
            root.appendChild(panel);
            return video;
          });
          const peer = (video) => videos.find((candidate) => candidate !== video);
          for (const video of videos) {
            video.addEventListener("play", () => {
              const other = peer(video);
              if (!other) return;
              if (Math.abs(other.currentTime - video.currentTime) > 0.02) {
                other.currentTime = video.currentTime;
              }
              if (other.paused) other.play().catch(() => {});
            });
            video.addEventListener("pause", () => {
              const other = peer(video);
              if (other && !other.paused) other.pause();
            });
            video.addEventListener("seeking", () => {
              const other = peer(video);
              if (other && Math.abs(other.currentTime - video.currentTime) > 0.02) {
                other.currentTime = video.currentTime;
              }
            });
            video.addEventListener("ratechange", () => {
              const other = peer(video);
              if (other) other.playbackRate = video.playbackRate;
            });
            video.addEventListener("timeupdate", () => {
              const other = peer(video);
              if (other && Math.abs(other.currentTime - video.currentTime) > 0.12) {
                other.currentTime = video.currentTime;
              }
            });
          }
          el.replaceChildren(root);
        }
        export default { render };
        """

        sources = traitlets.List(traitlets.Dict()).tag(sync=True)

    return (PairedCameras,)


# ── Cell 8: stream the paired remuxed S3 videos ──────────────────────────────
@app.cell
def _(PairedCameras, clip_videos, mo, urllib):
    _sources = []
    for _camera, _key in sorted(clip_videos.items()):
        _sources.append({
            "label": _camera.title(),
            "src": "https://aind-scratch-data.s3.amazonaws.com/"
            + urllib.parse.quote(_key, safe="/"),
        })
    mo.ui.anywidget(PairedCameras(sources=_sources))
    return


# ── Cell 8: fetch thermistor parquet ─────────────────────────────────────────
@app.cell
async def _(clip_map, clip_selector, http_get_bytes, io, mo, pq):
    _BUCKET = "aind-scratch-data"
    _key = clip_map[clip_selector.value]
    _url = f"https://{_BUCKET}.s3.amazonaws.com/{_key}"

    try:
        _data = await http_get_bytes(_url)
        therm = pq.read_table(io.BytesIO(_data)).to_pandas()
    except Exception as _exc:
        mo.stop(
            True,
            mo.callout(mo.md(f"**Failed to fetch thermistor parquet:** `{_exc}`"), kind="danger"),
        )

    return (therm,)


# ── Cell 9: clip summary ──────────────────────────────────────────────────────
@app.cell
def _(mo, np, therm):
    _t = therm["Time"].to_numpy()
    _dur = float(_t[-1]) if len(_t) > 1 else 0.0
    _fs  = 1.0 / float(np.median(np.diff(_t))) if len(_t) > 1 else float("nan")
    mo.hstack([
        mo.stat(value=f"{len(therm):,}", label="Thermistor samples",
                caption=f"~{_fs:.0f} Hz", bordered=True),
        mo.stat(value=f"{_dur:.1f} s", label="Duration", bordered=True),
    ], justify="start", gap=1)
    return


# ── Cell 8: extract raw arrays ───────────────────────────────────────────────
@app.cell
def _(np, therm):
    t_raw = therm["Time"].to_numpy(dtype=float)
    v_raw = therm["Signal"].to_numpy(dtype=float)
    fs_raw = 1.0 / float(np.median(np.diff(t_raw)))
    return fs_raw, t_raw, v_raw


# ── Cell 9: filter ────────────────────────────────────────────────────────────
@app.cell
def _(filter_sniff_signal, fs_raw, v_raw):
    v_filtered = filter_sniff_signal(v_raw, fs_raw)
    return (v_filtered,)


# ── Cell 10: resample ─────────────────────────────────────────────────────────
@app.cell
def _(CANONICAL_BREATHING_SAMPLING_RATE, pd, resample_uniform, t_raw, v_filtered):
    _df_in = pd.DataFrame({"time": t_raw, "adc_voltage": v_filtered})
    _df_rs = resample_uniform(_df_in, target_fs=CANONICAL_BREATHING_SAMPLING_RATE)
    t_rs = _df_rs["time"].to_numpy(dtype=float)
    v_rs = _df_rs["adc_voltage"].to_numpy(dtype=float)
    fs_rs = CANONICAL_BREATHING_SAMPLING_RATE
    return fs_rs, t_rs, v_rs


# ── Cell 11: detect (hard-coded defaults) ────────────────────────────────────
@app.cell
def _(detect_breathing_events, fs_rs, v_rs):
    inhale_peaks, exhale_troughs = detect_breathing_events(v_rs, fs_rs)
    return exhale_troughs, inhale_peaks


# ── Cell 12: view sliders ────────────────────────────────────────────────────
@app.cell
def _(mo, np, t_rs):
    _dur = float(t_rs[-1]) if len(t_rs) else 300.0
    center_slider = mo.ui.slider(
        start=float(t_rs[0]) if len(t_rs) else 0.0,
        stop=_dur,
        step=0.5,
        value=float(np.round(_dur / 2.0, 1)),
        label="Center (s)",
        full_width=True,
        debounce=True,
        show_value=True,
    )
    window_slider = mo.ui.slider(
        start=2.0, stop=60.0, step=1.0, value=20.0,
        label="Window size (s)",
        full_width=True,
        show_value=True,
    )
    return center_slider, window_slider


# ── Cell 13: pipeline — 4 stacked subplots sharing the same window ───────────
@app.cell
def _(
    center_slider,
    exhale_troughs,
    fs_raw,
    fs_rs,
    inhale_peaks,
    mo,
    plt,
    t_raw,
    t_rs,
    v_filtered,
    v_raw,
    v_rs,
    window_slider,
):
    _tc   = center_slider.value
    _half = window_slider.value / 2.0
    _t0   = max(float(t_raw[0]), _tc - _half)
    _t1   = min(float(t_raw[-1]), _tc + _half)

    # window masks (raw grid and resampled grid)
    _mr  = (t_raw >= _t0) & (t_raw <= _t1)
    _mrs = (t_rs  >= _t0) & (t_rs  <= _t1)
    _inh = inhale_peaks[(t_rs[inhale_peaks] >= _t0) & (t_rs[inhale_peaks] <= _t1)]
    _exh = exhale_troughs[(t_rs[exhale_troughs] >= _t0) & (t_rs[exhale_troughs] <= _t1)]

    _fig, (_ax1, _ax2, _ax3, _ax4) = plt.subplots(
        4, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # ── Row 1: Raw ────────────────────────────────────────────────────────────
    _ax1.plot(t_raw[_mr], v_raw[_mr], lw=0.7, color="#888888")
    _ax1.set_ylabel("Thermistor (a.u.)", fontsize=8)
    _ax1.set_title(f"Raw  (fs ≈ {fs_raw:.0f} Hz)", fontsize=9, loc="left", pad=3)
    _ax1.tick_params(labelbottom=False)

    # ── Row 2: Filtered ───────────────────────────────────────────────────────
    _ax2.plot(t_raw[_mr], v_filtered[_mr], lw=0.8, color="#1f6aa5")
    _ax2.set_ylabel("Thermistor (a.u.)", fontsize=8)
    _ax2.set_title("Filtered  (0.2 Hz HP · 40 Hz LP Butterworth)", fontsize=9, loc="left", pad=3)
    _ax2.tick_params(labelbottom=False)

    # ── Row 3: Resampled ──────────────────────────────────────────────────────
    _ax3.plot(t_raw[_mr], v_filtered[_mr], lw=0.8, color="#aaaaaa", alpha=0.6,
              label=f"filtered (~{fs_raw:.0f} Hz)")
    _ax3.plot(t_rs[_mrs], v_rs[_mrs], "o", ms=2.5, color="#e05c2a",
              label=f"resampled ({fs_rs:.0f} Hz)")
    _ax3.set_ylabel("Thermistor (a.u.)", fontsize=8)
    _ax3.set_title(f"Resampled  ({fs_rs:.0f} Hz uniform grid)", fontsize=9, loc="left", pad=3)
    _ax3.legend(loc="upper right", fontsize=7, ncol=2)
    _ax3.tick_params(labelbottom=False)

    # ── Row 4: Detected ───────────────────────────────────────────────────────
    _ax4.plot(t_rs[_mrs], v_rs[_mrs], lw=0.8, color="#1f6aa5")
    if len(_inh):
        _ax4.plot(t_rs[_inh], v_rs[_inh], "^", ms=5, color="#e05c2a", zorder=4,
                  label=f"inhale ({len(inhale_peaks)} total)")
    if len(_exh):
        _ax4.plot(t_rs[_exh], v_rs[_exh], "v", ms=5, color="#2eaa5e", zorder=4,
                  label=f"exhale ({len(exhale_troughs)} total)")
    _ax4.axvline(_tc, color="red", lw=1.0, ls="--", zorder=5, alpha=0.6)
    _ax4.set_ylabel("Thermistor (a.u.)", fontsize=8)
    _ax4.set_title(
        f"Detected  ({len(inhale_peaks)} inhale · {len(exhale_troughs)} exhale)",
        fontsize=9, loc="left", pad=3,
    )
    _ax4.set_xlabel("Time (s)")
    if len(_inh) or len(_exh):
        _ax4.legend(loc="upper right", fontsize=8, ncol=2)

    _ax4.set_xlim(_t0, _t1)
    plt.tight_layout()

    _out = mo.vstack([
        mo.hstack([center_slider, window_slider], gap=2),
        mo.as_html(_fig),
    ])
    plt.close(_fig)
    _out
    return


# ── Cell 14: analysis — stats cards + IBI histograms ─────────────────────────
@app.cell
def _(exhale_troughs, fs_rs, inhale_peaks, mo, np, plt):
    _dt      = 1.0 / fs_rs
    _ibi_inh = np.diff(inhale_peaks) * _dt
    _ibi_exh = np.diff(exhale_troughs) * _dt
    _br      = 1.0 / float(np.mean(_ibi_inh)) if len(_ibi_inh) else float("nan")

    def _s(arr, fn):
        return f"{fn(arr):.3f} s" if len(arr) else "—"

    _cards = mo.hstack([
        mo.stat(value=f"{len(inhale_peaks):,}", label="Inhale events", bordered=True),
        mo.stat(value=f"{len(exhale_troughs):,}", label="Exhale events", bordered=True),
        mo.stat(
            value=f"{_br:.2f} Hz" if not float("nan") == _br else "—",
            label="Mean breathing rate",
            caption=f"{_br * 60:.0f} bpm" if not float("nan") == _br else "",
            bordered=True,
        ),
        mo.stat(value=_s(_ibi_inh, np.median), label="Median inhale IBI", bordered=True),
        mo.stat(value=_s(_ibi_exh, np.median), label="Median exhale IBI", bordered=True),
    ], justify="start", gap=1)

    _fig, (_ax_inh, _ax_exh) = plt.subplots(1, 2, figsize=(12, 3))
    for _ax, _ibi, _color, _lbl in [
        (_ax_inh, _ibi_inh, "#e05c2a", "Inter-inhale interval"),
        (_ax_exh, _ibi_exh, "#2eaa5e", "Inter-exhale interval"),
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
    _ibi_html = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        mo.md("---\n### Analysis"),
        _cards,
        _ibi_html,
    ])
    return


if __name__ == "__main__":
    app.run()
