# Baseline

A simple baseline for the [Breathing from Video Challenge](https://www.codabench.org) plus a worked example showing how to load the data and run inference.

## Setup

```bash
# From repo root
uv sync --all-packages

# Or, from this directory only
uv sync
```

## Download data

```bash
python ../competition/public_data/download_data.py --dest ./data
```

This downloads all packaged clips from
`s3://aind-scratch-data/vr-foraging/breathing-codabench-challenge/demo/packaged`
into `./data/`, preserving the `test/` and `train/` sub-directories.
AWS credentials must be configured (`aws configure` or the standard
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables).

## Run the baseline

```bash
# Predict breathing signals for the test split
uv run python -m breathing_baseline.predict \
    --video-dir ./data/test \
    --out ./submission.zip
```

## Notebooks

Open with marimo (from the repo root):

```bash
uv run marimo edit baseline/notebooks/01_explore_data.py
```

## Baseline algorithm

The baseline uses **dense optical flow** (Farnebäck) averaged over a
nose/snout ROI to extract a proxy respiratory signal. It is intentionally
simple — the point is to illustrate the data pipeline, not to win.

See `src/breathing_baseline/inference.py` for the implementation.

## Docker (optional)

Build and run the baseline in a container (build from repo root for
workspace context):

```bash
docker build -f baseline/Dockerfile -t breathing-baseline .
docker run --rm -v $(pwd)/data:/data breathing-baseline \
    --video-dir /data/test --out /data/submission.zip
```
