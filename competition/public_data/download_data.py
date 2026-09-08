import argparse
import subprocess
from pathlib import Path

S3_URI = "s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public/"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download packaged competition data from S3."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("./data"),
        help="Destination directory (default: ./data)",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", "--no-sign-request", S3_URI, str(args.dest)],
        check=True,
    )


if __name__ == "__main__":
    main()
