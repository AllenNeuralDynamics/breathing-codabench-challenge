"""Location of the public competition dataset in S3.

Single source of truth for every script/notebook that needs to find the data
(`download_data.py`, docs, etc). The one exception is
`baseline-cnn-tcn/notebooks/01_explore_data_wasm.py`, which runs in-browser
via Pyodide and cannot import workspace packages -- keep its hardcoded bucket
and prefix in sync with this file by hand.
"""

S3_BUCKET = "aind-scratch-data"
S3_PUBLIC_PREFIX = (
    "vr-foraging/codabench-breathing-challenge/"
    "9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public"
)
S3_PUBLIC_URI = f"s3://{S3_BUCKET}/{S3_PUBLIC_PREFIX}/"
