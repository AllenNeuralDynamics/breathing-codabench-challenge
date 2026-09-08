# Breathing from Video Challenge

Can you recover a mouse's breathing signal from video alone?

Participants get paired video clips and thermistor breathing recordings. The task
is to predict the breathing trace from the video frames only. This page covers
what the data looks like, how to download it, and how to start exploring it.

- **Input:** a 5-minute clip from one or both cameras (`face`, `side`), 240 fps.
- **Output:** a predicted breathing signal over the clip's time axis.

Evaluation details and the submission format are still being finalised and will
be published here before the development phase opens.

---

## Getting set up

The challenge tooling lives in the
[breathing-codabench-challenge](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge)
repo and uses [uv](https://docs.astral.sh/uv/) as its package manager.

```bash
git clone https://github.com/AllenNeuralDynamics/breathing-codabench-challenge
cd breathing-codabench-challenge
uv sync --all-packages
```

We also recommend having the following two external tools on your `PATH`:

| Tool | Used for |
|---|---|
| [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) | downloading the data |
| [ffmpeg](https://ffmpeg.org/download.html) | decoding video frames |

While not necessary, we will assume these are installed and available on your `PATH` for the rest of the document.

---

## Download the data

The data sits in a **public** S3 prefix — no AWS account or credentials needed.

```
s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public/
```

### Option 1 — download script

```bash
uv run competition/public_data/download_data.py --dest ./data
```

This is a thin wrapper around `aws s3 sync --no-sign-request` and fetches
everything, preserving the `train/` and `test/` sub-directories.

### Option 2 — AWS CLI directly

Useful when you want a subset. The full dataset is **~49 GB** (train ~21 GB,
test ~28 GB), so grabbing a couple of clips first is usually the right move.

```bash
S3=s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public

# Everything
aws s3 sync --no-sign-request $S3/ ./data/

# Just the train split
aws s3 sync --no-sign-request $S3/train/ ./data/train/
```

---

## Data layout

```
data/
├── train/                                 # 12 sessions × 2 parts = 24 clips
│   ├── thermistor_1_part_1.parquet        # breathing signal (ground truth)
│   ├── video_face_1_part_1.mp4            # face camera video
│   ├── video_face_1_part_1.parquet        # face camera frame timestamps
│   ├── video_side_1_part_1.mp4            # side camera video
│   ├── video_side_1_part_1.parquet        # side camera frame timestamps
│   └── ...
└── test/                                  # 16 sessions × 2 parts = 32 clips
    ├── video_face_1_part_1.mp4            # face camera video
    ├── video_face_1_part_1.parquet        # face camera frame timestamps
    ├── video_side_1_part_1.mp4            # side camera video
    ├── video_side_1_part_1.parquet        # side camera frame timestamps
    └── ...                                # no thermistor_*.parquet — ground truth is withheld
```

Files are named `{stream}_{session}_part_{part}.{ext}`. Session indices restart
at 1 within each split, so `train/…_1_…` and `test/…_1_…` are **different**
recordings. Each session contributes two disjoint 300-second parts.

### Thermistor parquet — `thermistor_{S}_part_{P}.parquet` (train only)

| Column | Type | Description |
|---|---|---|
| `Time` | float64 | Sample timestamp, in the clip's shared time base (see [Time alignment](#time-alignment)) |
| `Signal` | float64 | Thermistor ADC reading (a.u.) |

Sampled at **~250 Hz** (≈75,000 rows per clip). The raw signal is unfiltered after acquisition.

### Video — `video_{face,side}_{S}_part_{P}.mp4`

H.264, 720 × 540, `yuv420p`, **~240 fps**, ~72,000 frames per clip.

### Frame timestamps — `video_{face,side}_{S}_part_{P}.parquet`

| Column | Type | Description |
|---|---|---|
| `Time` | float64 | Timestamp of each frame, in the clip's shared time base |

One row per video frame, in order: **row `i` is frame `i`** of the matching MP4.
Always read the timestamp from this column — do not compute it as
`frame_index / fps`. Frame intervals are not uniform, and the column is the only
record of when each frame was actually acquired.

### Time alignment

**Every parquet belonging to one clip is expressed in the same time base.** That
is the guarantee to build on: a thermistor sample and a face- or side-camera row
carrying the same `Time` value refer to the same instant. Align streams by
looking up nearest timestamps between their `Time` columns.

The time base is the session's acquisition clock with a constant offset removed
(the offset is the primary/face camera's first frame in the clip window). It is
**not** an index and **not** a "seconds since clip start" counter:

- No stream is guaranteed to start at `Time == 0`. The thermistor typically
  starts a few hundred microseconds to a few milliseconds in.
- `Time` can be **negative**. In 7 of the 56 packaged clips the side camera's
  first frame lands at about `-1.03 s`. The clip is cut with a stream copy, so
  the seek snaps back to the preceding keyframe — up to one 250-frame GOP
  (~1.04 s at 240 fps) of extra footage before the requested window, carrying
  the timestamps that go with it.
- Sampling is not perfectly regular in any stream, so intervals between
  consecutive `Time` values vary within nominal bounds.

Treat `Time` as the authoritative timestamp for every stream and never infer it
from row position or a nominal sampling rate.

---

## Explore the data

### Interactive notebook

The repo ships a [marimo](https://marimo.io) notebook that walks a single clip
through the full pipeline — load, filter, resample, detect breathing events, and
scrub the video frame-by-frame against the breathing trace:

```bash
# From the repo root (the notebook defaults to ./data/packaged)
uv run marimo edit baseline-cnn-tcn/notebooks/01_explore_data.py
```

Pick a clip and a camera from the dropdowns at the top. The detection sliders let
you see how the event-detection parameters change what gets picked up.

### Loading a clip yourself

```python
import numpy as np
import pandas as pd

therm = pd.read_parquet("data/packaged/train/thermistor_1_part_1.parquet")
frames = pd.read_parquet("data/packaged/train/video_face_1_part_1.parquet")

t = therm["Time"].to_numpy()
fs = 1.0 / np.median(np.diff(t))  # ≈ 250 Hz

# Video frame nearest a given clip time
frame_idx = int(np.argmin(np.abs(frames["Time"].to_numpy() - 12.5)))
```

### Reusable signal-processing helpers

The `scoring` package (installed by `uv sync --all-packages`) exposes the same
primitives the notebook and the scoring program use:

```python
from scoring.processing import (
    CANONICAL_BREATHING_SAMPLING_RATE,  # 60 Hz common grid
    detect_breathing_events,  # inhale peaks / exhale troughs
    filter_sniff_signal,  # notch + band-pass denoising
    resample_uniform,  # interpolate onto a uniform grid
)

v = filter_sniff_signal(therm["Signal"].to_numpy(), fs)
```

---

## Organizers

[Allen Institute for Neural Dynamics](https://alleninstitute.org/division/neural-dynamics/)
If you have issues with the dataset or provided files, open an issue on the [competition's GitHub repository](https://github.com/AllenNeuralDynamics/breathing-codabench-challenge).