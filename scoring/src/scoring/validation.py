"""Submission validation for the breathing-from-video challenge.

``score.py`` runs these checks before handing a clip to
:func:`scoring.metrics.score_clip`, so a malformed submission fails with a
specific message instead of a stack trace or silently-wrong scoring.

All three files are required per clip: ``thermistor_{clip_id}.parquet``
(:func:`validate_signal_frame`) and ``inhale_times_{clip_id}.parquet`` /
``exhale_times_{clip_id}.parquet`` (:func:`validate_event_times_frame`).
A participant with no better source for onset/offset times can run
:func:`scoring.processing.detect_inhalation_events` on their own predicted
signal and submit that. All raise :class:`SubmissionValidationError` on
failure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scoring.processing import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

MIN_SAMPLES = 2
"""Fewest samples a signal parquet may have (need at least 2 for interpolation)."""

EVENT_TIME_COLUMN = "time_s"

THERMISTOR_STREAM = "thermistor"
INHALE_TIMES_STREAM = "inhale_times"
EXHALE_TIMES_STREAM = "exhale_times"


class SubmissionValidationError(ValueError):
    """Raised when a submitted file fails schema/sanity checks."""


def _require_columns(
    df: pd.DataFrame, columns: set[str], *, clip_id: str, kind: str
) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise SubmissionValidationError(
            f"{clip_id}: {kind} is missing required column(s) {sorted(missing)}; "
            f"has {list(df.columns)}"
        )


def validate_signal_frame(df: pd.DataFrame, *, clip_id: str) -> None:
    """Validate a continuous-signal submission (or ground-truth) frame.

    Checks columns present and numeric, at least :data:`MIN_SAMPLES` rows,
    no NaN/Inf, and ``Time`` strictly increasing.
    """
    kind = "signal file"
    _require_columns(
        df, {TIME_COLUMN, BREATHING_SIGNAL_COLUMN}, clip_id=clip_id, kind=kind
    )

    if len(df) < MIN_SAMPLES:
        raise SubmissionValidationError(
            f"{clip_id}: {kind} has {len(df)} row(s); need at least {MIN_SAMPLES}"
        )

    for col in (TIME_COLUMN, BREATHING_SIGNAL_COLUMN):
        values = df[col].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            raise SubmissionValidationError(
                f"{clip_id}: {kind} column {col!r} is not numeric "
                f"(dtype {values.dtype})"
            )
        if not np.isfinite(values).all():
            raise SubmissionValidationError(
                f"{clip_id}: {kind} column {col!r} contains NaN/Inf values"
            )

    t = df[TIME_COLUMN].to_numpy(dtype=float)
    if not np.all(np.diff(t) > 0):
        raise SubmissionValidationError(
            f"{clip_id}: {kind} column {TIME_COLUMN!r} is not strictly increasing "
            "(duplicate or out-of-order timestamps)"
        )


def validate_event_times_frame(df: pd.DataFrame, *, clip_id: str, kind: str) -> None:
    """Validate an ``inhale_times`` / ``exhale_times`` submission frame.

    Checks ``time_s`` present, numeric, no NaN/Inf, and strictly increasing.
    *kind* (``"inhale_times"`` or ``"exhale_times"``) is used only in error
    messages.
    """
    file_kind = f"{kind} file"
    _require_columns(df, {EVENT_TIME_COLUMN}, clip_id=clip_id, kind=file_kind)

    times = df[EVENT_TIME_COLUMN].to_numpy()
    if not np.issubdtype(times.dtype, np.number):
        raise SubmissionValidationError(
            f"{clip_id}: {file_kind} column {EVENT_TIME_COLUMN!r} is not numeric "
            f"(dtype {times.dtype})"
        )
    if not np.isfinite(times).all():
        raise SubmissionValidationError(
            f"{clip_id}: {file_kind} column {EVENT_TIME_COLUMN!r} contains "
            "NaN/Inf values"
        )

    t = times.astype(float)
    if not np.all(np.diff(t) > 0):
        raise SubmissionValidationError(
            f"{clip_id}: {file_kind} times are not strictly increasing "
            "(sort by time_s, drop duplicates)"
        )


def event_times_from_frame(df: pd.DataFrame) -> np.ndarray:
    """Extract the sorted ``time_s`` array from a validated events frame."""
    return df[EVENT_TIME_COLUMN].to_numpy(dtype=float)


def discover_clip_ids(res_dir: Path) -> list[str]:
    """Clip ids present in a submission's ``res/`` directory.

    Clips are identified by ``thermistor_{clip_id}.parquet`` specifically,
    not by any of the three required files, so a submission missing one of
    the other two is still discovered as a (then invalid) clip rather than
    silently skipped.
    """
    prefix = f"{THERMISTOR_STREAM}_"
    return sorted(
        p.stem.removeprefix(prefix) for p in res_dir.glob(f"{prefix}*.parquet")
    )


def validate_submission_clip(res_dir: Path, clip_id: str) -> list[str]:
    """Validate one clip's submission files, with no ground truth or network
    access needed -- unlike ``score.py``, which validates as a side effect of
    loading each file for scoring, this only checks and reports.

    All three files are required. Returns a list of problem descriptions;
    empty means the clip is valid.
    """
    problems: list[str] = []

    thermistor_path = res_dir / f"{THERMISTOR_STREAM}_{clip_id}.parquet"
    if not thermistor_path.exists():
        problems.append(f"missing required file {thermistor_path.name}")
    else:
        try:
            validate_signal_frame(pd.read_parquet(thermistor_path), clip_id=clip_id)
        except SubmissionValidationError as exc:
            problems.append(str(exc))

    for stream in (INHALE_TIMES_STREAM, EXHALE_TIMES_STREAM):
        path = res_dir / f"{stream}_{clip_id}.parquet"
        if not path.exists():
            problems.append(f"missing required file {path.name}")
            continue
        try:
            validate_event_times_frame(
                pd.read_parquet(path), clip_id=clip_id, kind=stream
            )
        except SubmissionValidationError as exc:
            problems.append(str(exc))

    return problems
