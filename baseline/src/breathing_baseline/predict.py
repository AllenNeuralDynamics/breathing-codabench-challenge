"""
CLI: run baseline inference on a directory of video clips and produce a
submission zip.

Usage:
    python -m breathing_baseline.predict --video-dir ./data/private --out submission.zip
"""

import argparse
import io
import zipfile
from pathlib import Path

from .inference import infer_breathing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline inference and package submission."
    )
    parser.add_argument(
        "--video-dir", type=Path, required=True, help="Directory of .mp4 clips"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("submission.zip"), help="Output zip path"
    )
    parser.add_argument(
        "--fs", type=float, default=1000.0, help="Target sampling rate (Hz)"
    )
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.video_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for video_path in videos:
            clip_id = video_path.stem
            print(f"  Processing {clip_id} …", end=" ", flush=True)
            try:
                df = infer_breathing(str(video_path), target_fs=args.fs)
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                zf.writestr(f"{clip_id}.parquet", buf.getvalue())
                print("done")
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {exc}")

    print(f"\nSubmission written to {args.out}")


if __name__ == "__main__":
    main()
