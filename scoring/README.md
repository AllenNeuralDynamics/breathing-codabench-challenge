# scoring

The Codabench scoring program for the Breathing from Video Challenge.

## Pull

```bash
docker pull ghcr.io/allenneuraldynamics/breathing-codabench-challenge/scoring:latest
# or a pinned release (tag matches the GitHub release exactly, e.g. v0.0.0rc0):
docker pull ghcr.io/allenneuraldynamics/breathing-codabench-challenge/scoring:v0.0.0rc0
```

Or build it locally (from the repo root, so the build sees `uv.lock`):
`docker build -t scoring -f scoring/Dockerfile .`

## Run

```bash
docker run --rm --env-file scoring/.env \
    -v /path/to/submission:/input \
    -v /path/to/output:/output \
    ghcr.io/allenneuraldynamics/breathing-codabench-challenge/scoring:latest
```

PowerShell (`\` line continuation is `` ` ``, and quote paths containing spaces):

```powershell
docker run --rm --env-file scoring/.env `
    -v C:\path\to\submission:/input `
    -v C:\path\to\output:/output `
    ghcr.io/allenneuraldynamics/breathing-codabench-challenge/scoring:latest
```

### The contract

| | |
|---|---|
| **`/input`** (bind mount, read) | A submission directory containing `res/` -- `thermistor_{clip_id}.parquet`, `inhale_times_{clip_id}.parquet`, and `exhale_times_{clip_id}.parquet` are all required per clip. See the [submission format](../README.md#6-package-and-submit) and [`validate_submission.py`](validate_submission.py). |
| **`/output`** (bind mount, write) | Where results land: `scores.json` (aggregated) and `detailed/{clip_id}.json` (per clip). Created automatically if it doesn't exist. |
| **`GROUND_TRUTH_S3_URI`** (env var) | S3 location of the ground-truth thermistor parquets, e.g. `s3://bucket/prefix`. Required -- the container exits immediately with a clear error if it's unset. Supply it with `--env-file scoring/.env` (copy from [`.env.example`](.env.example) and fill in the value) or `-e GROUND_TRUTH_S3_URI=...`. |

Positional args (`<input_dir> <output_dir>`) default to `/input`/`/output`
via the image's `CMD`, matching the mounts above -- pass them explicitly
only if you want different in-container paths. This is also exactly how
Codabench itself invokes the image (`command: python /app/score.py $input
$output` in [`competition.yaml`](../competition/competition.yaml)), just
with `$input`/`$output` bind-mounted at those same paths instead of `/input`/
`/output` by name.

## Validate before scoring

No ground truth or network access needed -- checks the submission's file
format only:

```bash
uv run validate_submission.py /path/to/submission
```

## Local development

```bash
cd scoring
uv sync
cp .env.example .env   # fill in GROUND_TRUTH_S3_URI
uv run --env-file .env score.py /path/to/submission /path/to/output
```

See the main [README](../README.md#scoring-program) for the metrics
computed, the submission format, and the CI/release workflow.
