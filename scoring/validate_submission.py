"""
Validate a submission before it's scored.

    uv run validate_submission.py <submission_dir>

<submission_dir> is either a res/-shaped directory directly, or a parent
directory containing one (i.e. what score.py reads as $input) -- both work.
Runs the same schema checks score.py runs on each clip before scoring it,
but with no ground truth, S3, or Docker involved, so a malformed submission
is caught locally instead of surfacing mid-scoring or on Codabench.
"""

import sys
from pathlib import Path

from scoring.validation import discover_clip_ids, validate_submission_clip


def main(submission_dir: Path) -> int:
    res_dir = submission_dir / "res"
    if not res_dir.is_dir():
        res_dir = submission_dir

    clip_ids = discover_clip_ids(res_dir)
    if not clip_ids:
        print(f"No thermistor_*.parquet files found under {res_dir}")
        return 1

    n_bad = 0
    for clip_id in clip_ids:
        problems = validate_submission_clip(res_dir, clip_id)
        if problems:
            n_bad += 1
            print(f"[FAIL] {clip_id}")
            for problem in problems:
                print(f"    {problem}")
        else:
            print(f"[ OK ] {clip_id}")

    print(f"\n{len(clip_ids) - n_bad}/{len(clip_ids)} clip(s) valid")
    return 1 if n_bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <submission_dir>")
    sys.exit(main(Path(sys.argv[1])))
