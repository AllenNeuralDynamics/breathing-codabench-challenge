"""
Codabench scoring entrypoint.

Called by the platform as:
    python score.py $input $output

$input layout (set by Codabench):
    $input/res/         participant submission (zip extracted)
        {clip_id}.parquet
    $input/ref/         reference data (unused — GT fetched from S3)

$output layout (written by this script):
    $output/scores.json     leaderboard scalars (aggregated across clips)
    $output/detailed/       per-clip Score breakdown
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

from scoring.metrics import Score, score_clip

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

S3_BUCKET = "aind-breathing-challenge"
S3_GT_PREFIX = "ground-truth"  # s3://{BUCKET}/{GT_PREFIX}/{clip_id}.parquet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _load_truth(clip_id: str, s3) -> pd.DataFrame:
    """Download ground-truth parquet for a clip from S3."""
    key = f"{S3_GT_PREFIX}/{clip_id}.parquet"
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))


def _load_predicted(pred_dir: Path, clip_id: str) -> pd.DataFrame:
    path = pred_dir / f"{clip_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction for clip {clip_id}: {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SCORE_FIELDS = [f.name for f in fields(Score)]


def main(input_dir: Path, output_dir: Path) -> None:
    pred_dir = input_dir / "res"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "detailed"
    detail_dir.mkdir(exist_ok=True)

    pred_clip_ids = [p.stem for p in pred_dir.glob("*.parquet")]
    if not pred_clip_ids:
        sys.exit("No prediction parquets found in submission.")

    s3 = _s3_client()
    all_scores: list[Score] = []

    for clip_id in pred_clip_ids:
        try:
            truth_thermistor = _load_truth(clip_id, s3)
            predicted_thermistor = _load_predicted(pred_dir, clip_id)
            score = score_clip(truth_thermistor, predicted_thermistor)
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
