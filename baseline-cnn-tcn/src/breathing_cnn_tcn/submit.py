"""Package private-split predictions into the Codabench submission format.

Writes thermistor_/inhale_times_/exhale_times_ parquets under
submission/<run>/res/ (<run> = the checkpoint's parent directory name).
Onsets use the model's onset head when exactly one checkpoint is given;
offsets, and onsets otherwise, fall back to
:func:`~scoring.processing.detect_inhalation_events`.

Requires the split to already be preprocessed (see preprocess.py)::

    python -m breathing_cnn_tcn.submit --checkpoint runs/<run>/best.pt

Score the result with the scoring Docker image against submission/<run> --
see scoring/README.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scoring.processing import (
    BREATHING_SIGNAL_COLUMN,
    CANONICAL_BREATHING_SAMPLING_RATE,
    TIME_COLUMN,
    detect_inhalation_events,
)

from .clips import PRIVATE_SPLIT
from .dataset import ClipEntry, load_manifest
from .evaluate import load_checkpoint, predict_entry
from .infer import predict_clip

# Onset-head peak extraction, matching plot_diagnosis.py's convention.
ONSET_HEAD_HEIGHT = 0.4
ONSET_HEAD_REFRACTORY_S = 0.050

STREAMS = ("thermistor", "inhale_times", "exhale_times")


def clip_suffix(entry: ClipEntry) -> str:
    """The ``{session}_part_{part}`` clip_id scoring.score expects."""
    return f"{entry.session_idx}_part_{entry.part}"


def onset_head_times(onset_prob: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Discrete onset times from the model's onset-head probability curve."""
    distance = max(1, int(CANONICAL_BREATHING_SAMPLING_RATE * ONSET_HEAD_REFRACTORY_S))
    peaks, _ = find_peaks(onset_prob, height=ONSET_HEAD_HEIGHT, distance=distance)
    return times[peaks]


def package_clip(
    entry: ClipEntry,
    signal: np.ndarray,
    times: np.ndarray,
    onset_prob: np.ndarray | None,
    res_dir: Path,
) -> int:
    """Write one clip's thermistor_/inhale_times_/exhale_times_ files.

    Onsets use the onset-head probability when available; offsets, and
    onsets otherwise, come from :func:`detect_inhalation_events`. Returns
    n frames.
    """
    n = min(len(signal), len(times))
    clip_id = clip_suffix(entry)
    sig, t = signal[:n], times[:n].astype(np.float64)

    pd.DataFrame(
        {TIME_COLUMN: t, BREATHING_SIGNAL_COLUMN: sig.astype(np.float64)}
    ).to_parquet(res_dir / f"thermistor_{clip_id}.parquet", index=False)

    detected_on, detected_off = detect_inhalation_events(
        sig, CANONICAL_BREATHING_SAMPLING_RATE
    )
    onset_times = (
        onset_head_times(onset_prob[:n], t)
        if onset_prob is not None
        else t[detected_on]
    )
    offset_times = t[detected_off]

    pd.DataFrame({"time_s": onset_times.astype(np.float64)}).to_parquet(
        res_dir / f"inhale_times_{clip_id}.parquet", index=False
    )
    pd.DataFrame({"time_s": offset_times.astype(np.float64)}).to_parquet(
        res_dir / f"exhale_times_{clip_id}.parquet", index=False
    )
    return n


def _clear_stale_outputs(res_dir: Path) -> None:
    """Remove this run's own file kinds from a previous, possibly wider run."""
    for stream in STREAMS:
        for path in res_dir.glob(f"{stream}_*.parquet"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package private-split predictions into the submission format."
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--split", default=PRIVATE_SPLIT)
    parser.add_argument("--camera", default="face", choices=["face", "side"])
    parser.add_argument(
        "--sessions",
        type=int,
        nargs="+",
        help="Sessions to package. Defaults to every session in the manifest.",
    )
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Submission directory (predictions land in <out-dir>/res/). "
        "Defaults to submission/<run>, <run> = --checkpoint[0]'s parent dir.",
    )
    parser.add_argument(
        "--onset-times",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the model's onset head for onsets (needs exactly one "
        "--checkpoint); otherwise onsets fall back to "
        "detect_inhalation_events, like offsets always do.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or Path("submission") / args.checkpoint[0].parent.name

    if args.onset_times and len(args.checkpoint) != 1:
        print(
            f"note: --onset-times needs exactly one --checkpoint (got "
            f"{len(args.checkpoint)}); onsets will fall back to "
            f"detect_inhalation_events instead of the onset head."
        )
        args.onset_times = False

    device = torch.device(args.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    models = []
    for path in args.checkpoint:
        model, mean, std, state = load_checkpoint(path, device)
        models.append((model, mean, std))
        print(f"loaded {path}  epoch {state.get('epoch')}")

    _, entries = load_manifest(args.features_dir, args.split, args.camera)
    if args.sessions:
        wanted = set(args.sessions)
        entries = [e for e in entries if e.session_idx in wanted]
        missing = wanted - {e.session_idx for e in entries}
        if missing:
            raise SystemExit(f"no clips found for session(s) {sorted(missing)}")
    if not entries:
        raise SystemExit(
            f"no clips found in {args.features_dir} for split {args.split!r}"
        )

    res_dir = out_dir / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_outputs(res_dir)

    print(
        f"\npackaging {len(entries)} clip(s) with {len(models)} model(s) -> {res_dir}\n"
    )
    for entry in sorted(entries, key=lambda e: (e.session_idx, e.part)):
        times = np.load(entry.times)
        if args.onset_times:
            model, mean, std = models[0]
            signal, onset_prob = predict_clip(
                model,
                entry,
                mean,
                std,
                window=args.infer_window,
                device=device,
                frame_chunk=args.frame_chunk,
                amp_dtype=amp_dtype,
            )
        else:
            signal = predict_entry(
                models,
                entry,
                device,
                window=args.infer_window,
                frame_chunk=args.frame_chunk,
                amp_dtype=amp_dtype,
            )
            onset_prob = None
        n = package_clip(entry, signal, times, onset_prob, res_dir)
        print(f"  {clip_suffix(entry):16s} session {entry.session_idx:>2d}  {n} frames")

    print(f"\nwrote {len(entries)} clip(s) to {res_dir}")
    print(
        f"score it with the scoring Docker image against {out_dir} -- see module docstring."
    )


if __name__ == "__main__":
    main()
