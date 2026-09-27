# Challenge baseline scoring

The CNN + TCN model, data preparation, training, validation, and local model
diagnostics live in [Zephyr](zephyr/README.md). Follow Zephyr's README to
download the data with the AWS CLI, prepare features, and train a checkpoint.
This package applies the challenge's authoritative scoring metrics to that
trained checkpoint; it does not implement or train the network.

From the challenge repository root, initialize the submodule and install the
scoring dependencies:

```bash
git submodule update --init --recursive
uv sync --all-packages
```

If you trained from the checked-out `baseline/zephyr` submodule, run the
official held-out score from the challenge repository root:

The scorer needs the raw labelled thermistor parquet files under
`baseline/zephyr/data/train/` and Zephyr's preprocessed feature manifests and
arrays under `baseline/zephyr/data/features/`. These are the paths supplied
below with `--packaged-root` and `--features-dir`; if your data lives elsewhere,
pass its locations explicitly. The raw thermistor files are used for scoring,
while the features are used to generate predictions.

```bash
uv run --package breathing-baseline \
  python -m baseline.evaluate \
  --checkpoint baseline/zephyr/runs_all/zephyr/<run>/best.pt \
  --features-dir baseline/zephyr/data/features \
  --packaged-root baseline/zephyr/data \
  --holdout-json baseline/artifacts/holdout_sessions.json \
  --plot
```

The command scores the reserved sessions and writes diagnostic plots beside
the checkpoint. Replace the paths if your Zephyr data or checkpoint lives
elsewhere. See `python -m baseline.evaluate --help` for options.

For Codabench submission packaging, this repository also provides
`baseline.submit`; see the [repository README](../README.md#6-package-and-submit).
