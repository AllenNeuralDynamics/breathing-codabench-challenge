# CNN + TCN baseline

A learned baseline for the [Breathing from Video Challenge](https://www.codabench.org):
a per-frame CNN over a snout crop feeds a dilated TCN that reconstructs the
thermistor breathing trace.

This package sits alongside `baseline/`, which stays the intentionally simple
optical-flow reference that participants read first. Nothing here changes that
package or its Docker image.

## Architecture

```
720x540 --downsample--> WxH --crop--> box --> 4 channels --> CNN --> TCN --> 60 Hz waveform
                                              gray                    \--> inhale-onset heatmap
                                              diff  I(t) - I(t-4)
                                              flow_x, flow_y  (DIS, stride 4)
```

Both geometry choices are made by hand in
[`annotate.py`](src/breathing_cnn_tcn/annotate.py): the **downsample target**
sets how much detail survives, and the **crop box** sets how much of the frame
the model sees. The box is placed on the downsampled frame, so preprocessing
scales before cropping and there is no second rescale — what you crop is the
model's input shape.

## Getting started

All commands run from the repo root.

```bash
uv sync --all-packages --extra train
```

**1. Data** (~11 GiB, public bucket, no credentials).

```bash
aws s3 sync --no-sign-request s3://aind-scratch-data/vr-foraging/codabench-breathing-challenge/9bd7d45e35cdfea74ae9c5897a336bb6dcc972288db862b523635ab5a397657a/public/train/ data/train/
```

**2. Crop boxes** ship with the repo as `baseline-cnn-tcn/artifacts/session_boxes_face.json`.
Nothing to do unless you want to change them — see [Choosing crops](#choosing-crops).

**3. Preprocess** — decode, downsample, crop, channelise. ~45 min, ~20 GiB.

```bash
uv run python -m breathing_cnn_tcn.preprocess --boxes-json baseline-cnn-tcn/artifacts/session_boxes_face.json
```

**4. Train.** Checkpoints land in `runs_all/baseline-cnn-tcn/<timestamp>/` as
`best.pt`, `last.pt`, `history.json`, `best_per_clip.json`. Every non-reserved
session trains together; defaults match the settings behind the shipped model.

```bash
uv run python -m breathing_cnn_tcn.train
```

**5. Evaluate** on the reserved sessions, scored the way the competition scores.
Several `--checkpoint` paths are ensembled; add `--plot` for the rate-breakdown
and reserved-grid diagnostic plots too (needs exactly one `--checkpoint`, since
the reserved-grid panel reads one model's own onset head).

```bash
uv run python -m breathing_cnn_tcn.evaluate --checkpoint runs_all/baseline-cnn-tcn/<timestamp>/best.pt --plot
```

Those sessions are consumable: every look influences what you try next, so score
a model you are ready to commit to, not every intermediate.

### Choosing crops

One box per session, placed by hand. There is no detector and no fallback:
`preprocess` refuses to run without a box for every clip, because a guessed box
fails silently — the arrays look normal and the model never learns. Needs a
display; everything else here runs headless.

```bash
uv run python -m breathing_cnn_tcn.annotate --prepare      # cache one frame per clip, ~10 s
uv run python -m breathing_cnn_tcn.annotate --width 360 --height 270 --box-size 96
```

Click places the box centre. Boxes auto-save on every edit, so the window can be
closed and reopened at any point.

| Key | |
|---|---|
| arrows / shift+arrows | nudge 1 px / 10 px |
| `[` `]` | resize the box on every session |
| `n` `p` | next / previous session |
| `1` `2` | switch between the session's two part frames |
| `c` | copy the previous session's box |
| `r` | re-centre on the frame |

The width/height flags set the starting downsample target; it is adjustable in
the window and rescales boxes already placed. The ROI column reads `set` once a
session is placed and `auto` while it still sits at the frame centre.

After re-annotating, re-run preprocess for that session alone:

```bash
uv run python -m breathing_cnn_tcn.preprocess --boxes-json baseline-cnn-tcn/artifacts/session_boxes_face.json --sessions 7
```
