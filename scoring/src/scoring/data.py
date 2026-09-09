"""Location of the competition dataset in S3.

Single source of truth for every script/notebook that needs to find the data.
The one exception is `baseline-cnn-tcn/notebooks/01_explore_data_wasm.py`,
which runs in-browser via Pyodide and cannot import workspace packages --
keep its hardcoded bucket and prefix in sync with this file by hand.

The public (train) split is a fixed, checked-in path. The ground-truth
location is not -- it's expected to move -- so it's read from
``GROUND_TRUTH_S3_URI`` instead (see ``scoring/.env.example``). Populate it
locally with ``uv run --env-file scoring/.env``; the Docker image bakes in
its own default via ``ENV``.
"""

import os
from urllib.parse import urlparse

S3_BUCKET = "aind-scratch-data"
S3_PUBLIC_PREFIX = (
    "vr-foraging/codabench-breathing-challenge/"
    "9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public"
)
S3_PUBLIC_URI = f"s3://{S3_BUCKET}/{S3_PUBLIC_PREFIX}/"

GROUND_TRUTH_S3_URI_ENV_VAR = "GROUND_TRUTH_S3_URI"


def ground_truth_s3_location() -> tuple[str, str]:
    """Return ``(bucket, prefix)`` parsed from ``GROUND_TRUTH_S3_URI``.

    Raises ``RuntimeError`` if the variable is unset, empty, or not a valid
    ``s3://`` URI.
    """
    uri = os.environ.get(GROUND_TRUTH_S3_URI_ENV_VAR, "").strip()
    if not uri:
        raise RuntimeError(
            f"{GROUND_TRUTH_S3_URI_ENV_VAR} is not set. Copy "
            "scoring/.env.example to scoring/.env, then run with "
            "`uv run --env-file scoring/.env ...` (or set the environment "
            "variable directly) -- e.g. s3://bucket/folder"
        )
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise RuntimeError(
            f"{GROUND_TRUTH_S3_URI_ENV_VAR}={uri!r} is not a valid s3:// URI"
        )
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")
