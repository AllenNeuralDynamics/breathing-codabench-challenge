"""
Codabench scoring entrypoint.

Called by the platform as:
    python score.py $input $output

$input/res/ (participant submission, zip extracted), clip_id = "{session}_part_{part}":
    thermistor_{clip_id}.parquet     required: continuous predicted signal
    inhale_times_{clip_id}.parquet   optional: submitted onset times
    exhale_times_{clip_id}.parquet   optional: submitted offset times

$output/:
    scores.json     leaderboard scalars (aggregated across clips)
    detailed/       per-clip Score breakdown

Ground truth is fetched from S3 at the location named by the
``GROUND_TRUTH_S3_URI`` env var (see ``scoring/.env.example``).
"""

import json
import sys
from dataclasses import fields
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from scoring.data import ground_truth_s3_location
from scoring.metrics import Score, score_clip
from scoring.validation import (
    event_times_from_frame,
    validate_event_times_frame,
    validate_signal_frame,
)

THERMISTOR_STREAM = "thermistor"
INHALE_TIMES_STREAM = "inhale_times"
EXHALE_TIMES_STREAM = "exhale_times"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _load_truth(clip_id: str, bucket: str, prefix: str, s3) -> pd.DataFrame:
    """Download ground-truth parquet for a clip from S3."""
    key = f"{prefix}/{THERMISTOR_STREAM}_{clip_id}.parquet"
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    validate_signal_frame(df, clip_id=f"{clip_id} (ground truth)")
    return df


def _load_predicted(pred_dir: Path, clip_id: str) -> pd.DataFrame:
    path = pred_dir / f"{THERMISTOR_STREAM}_{clip_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction for clip {clip_id}: {path}")
    df = pd.read_parquet(path)
    validate_signal_frame(df, clip_id=clip_id)
    return df


def _load_predicted_event_times(
    pred_dir: Path, clip_id: str, stream: str
) -> np.ndarray | None:
    """Load a submission's optional {stream}_{clip_id}.parquet, or None."""
    path = pred_dir / f"{stream}_{clip_id}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    validate_event_times_frame(df, clip_id=clip_id, kind=stream)
    return event_times_from_frame(df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SCORE_FIELDS = [f.name for f in fields(Score)]


def main(input_dir: Path, output_dir: Path) -> None:
    pred_dir = input_dir / "res"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "detailed"
    detail_dir.mkdir(exist_ok=True)

    # Clips are identified by their required thermistor_{clip_id}.parquet --
    # the optional inhale_times_/exhale_times_ side files aren't clip lists.
    prefix = f"{THERMISTOR_STREAM}_"
    pred_clip_ids = [
        p.stem.removeprefix(prefix) for p in pred_dir.glob(f"{prefix}*.parquet")
    ]
    if not pred_clip_ids:
        sys.exit("No prediction parquets found in submission.")

    gt_bucket, gt_prefix = ground_truth_s3_location()
    s3 = _s3_client()
    all_scores: list[Score] = []

    for clip_id in pred_clip_ids:
        try:
            truth_thermistor = _load_truth(clip_id, gt_bucket, gt_prefix, s3)
            predicted_thermistor = _load_predicted(pred_dir, clip_id)
            predicted_onset = _load_predicted_event_times(
                pred_dir, clip_id, INHALE_TIMES_STREAM
            )
            predicted_offset = _load_predicted_event_times(
                pred_dir, clip_id, EXHALE_TIMES_STREAM
            )
            score = score_clip(
                truth_thermistor,
                predicted_thermistor,
                predicted_onset_times_s=predicted_onset,
                predicted_offset_times_s=predicted_offset,
            )
            all_scores.append(score)
            (detail_dir / f"{clip_id}.json").write_text(score.to_json(indent=2))
        except Exception as exc:
            print(f"[WARN] Skipping {clip_id}: {exc}", file=sys.stderr)

    if not all_scores:
        sys.exit("No clips could be scored.")

    # Aggregate each numeric field across clips (NaN-safe mean)
    aggregated = {
        name: float(np.nanmean([getattr(s, name) for s in all_scores]))
        for name in _SCORE_FIELDS
    }

    # TODO: define primary leaderboard scalar once composite metric is designed
    scores_out = {
        "score": aggregated.get("inhale_f1", float("nan")),
        **aggregated,
        "n_clips_scored": len(all_scores),
    }

    output_path = output_dir / "scores.json"
    output_path.write_text(json.dumps(scores_out, indent=2))
    print(json.dumps(scores_out, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
