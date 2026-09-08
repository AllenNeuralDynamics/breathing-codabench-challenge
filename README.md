# Breathing from Video Challenge

A [Codabench](https://www.codabench.org) competition for inferring breathing signals from video snippets of mice.

Participants receive short video clips and — for the public (`train`) split — paired thermistor breathing signals. The goal is to predict the breathing trace from video alone. Submissions are scored against held-out thermistor recordings on the private (`test`) split.

The data schema, download instructions, and worked examples for participants live in
[`competition/pages/overview.md`](competition/pages/overview.md) — this README covers the repo layout and the developer / organizer workflow around it.

---

## Table of contents

- [Breathing from Video Challenge](#breathing-from-video-challenge)
  - [Table of contents](#table-of-contents)
  - [For participants](#for-participants)
    - [1. Set up your environment](#1-set-up-your-environment)
    - [2. Download the data](#2-download-the-data)
    - [3. Explore the data](#3-explore-the-data)
    - [4. Run a baseline](#4-run-a-baseline)
    - [5. Score locally](#5-score-locally)
    - [6. Package and submit](#6-package-and-submit)
- [TODO/TBD](#todotbd)
- [TODO/TBD](#todotbd-1)
  - [For organizers / developers](#for-organizers--developers)
    - [Scoring program](#scoring-program)
    - [CI / CD](#ci--cd)
    - [Adding tests](#adding-tests)

---

## For participants

### 1. Set up your environment

This repo uses [uv](https://docs.astral.sh/uv/) as its package manager and is a workspace of three packages: `scoring`, `baseline`, `baseline-cnn-tcn`.

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: winget install astral-sh.uv                  # Windows

# Clone the repo
git clone https://github.com/AllenNeuralDynamics/breathing-codabench-challenge
cd breathing-codabench-challenge

# Install everything
uv sync --all-packages
```

### 2. Download the data

```bash
uv run competition/public_data/download_data.py --dest ./data
```

No AWS account required — the bucket is public. See
[Download the data](competition/pages/overview.md#download-the-data) and
[Data layout](competition/pages/overview.md#data-layout) in the overview for
the S3 path, directory structure, and file schema.

### 3. Explore the data

```bash
uv run marimo edit baseline-cnn-tcn/notebooks/01_explore_data.py
```

See [Explore the data](competition/pages/overview.md#explore-the-data) for a
walkthrough, plus the reusable signal-processing helpers exposed by the
`scoring` package.

### 4. Run a baseline

Two are provided:

| Baseline                                          | What it is                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------- |
| [`baseline/`](baseline/README.md)                 | TODO.                                                               |
| [`baseline-cnn-tcn/`](baseline-cnn-tcn/README.md) | A learned CNN + TCN model: crop, preprocess, train, evaluate, plot. |

Each README covers its own setup, training/inference, and Docker image.

### 5. Score locally

[`scoring/score.py`](scoring/score.py) is the exact entrypoint Codabench runs:
it fetches ground-truth parquets from S3 and scores each submitted clip with
`scoring.metrics.score_clip`. That same function is what to score against
locally — see the `Score` dataclass in
[`scoring/src/scoring/metrics.py`](scoring/src/scoring/metrics.py) for every
metric it computes. `baseline-cnn-tcn`'s
[`evaluate.py`](baseline-cnn-tcn/src/breathing_cnn_tcn/evaluate.py) is a
working example of scoring predictions this way against local held-out clips.

The composite leaderboard scalar is still being finalised — see
[`competition/pages/overview.md`](competition/pages/overview.md) for the
current participant-facing docs.

### 6. Package and submit

# TODO/TBD


```bash
uv run python -m breathing_baseline.predict \
    --video-dir ./data/test \
    --out ./submission.zip
```

Then upload `submission.zip` on the [Codabench competition page](https://www.codabench.org/competitions/9975/).

**Submission format:**

# TODO/TBD

```
submission.zip
├── {clip_id}.parquet    # columns: time (float64), adc_voltage (float64)
└── ...
```

One parquet per clip, named by clip ID. Missing clips are skipped with a warning.

---

## For organizers / developers

### Scoring program

The scoring program lives in [`scoring/`](scoring/). It runs on Codabench for
every participant submission (image reference in
[`competition/competition.yaml`](competition/competition.yaml)):

1. Reads predicted parquets from the submission zip (`$input/res/`)
2. Fetches ground-truth parquets from S3
3. Computes all metrics and writes `$output/scores.json` (+ per-clip detail)

To test it locally:

```bash
cd scoring
uv sync

# Simulate a Codabench scoring run
mkdir -p /tmp/test_input/res /tmp/test_output
cp path/to/predictions/*.parquet /tmp/test_input/res/
uv run python score.py /tmp/test_input /tmp/test_output
cat /tmp/test_output/scores.json
```

### CI / CD

| Workflow           | Trigger                   | What it does                                                                             |
| ------------------ | ------------------------- | ---------------------------------------------------------------------------------------- |
| `ci.yml`           | Every PR / push to `main` | `uv sync` + `ruff check .` across the whole workspace                                    |
| `build-images.yml` | GitHub release published  | Builds `scoring/Dockerfile`, pushes to `ghcr.io/{repo}/scoring` (semver + `latest` tags) |

The image tag Codabench pulls is set by `container_image` in
`competition.yaml` — after cutting a release, check it still points at the tag
`build-images.yml` just pushed.

### Adding tests

```bash
cd scoring
mkdir tests
# add tests/test_metrics.py, then:
uv run pytest tests/ -v
```
