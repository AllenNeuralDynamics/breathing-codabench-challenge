"""Submission validation for the breathing-from-video challenge.

``score.py`` runs these checks before handing a clip to
:func:`scoring.metrics.score_clip`, so a malformed submission fails with a
specific message instead of a stack trace or silently-wrong scoring.

Validates ``thermistor_{clip_id}.parquet`` (required, via
:func:`validate_signal_frame`) and ``inhale_times_{clip_id}.parquet`` /
``exhale_times_{clip_id}.parquet`` (both optional, via
:func:`validate_event_times_frame`). All raise
:class:`SubmissionValidationError` on failure.
"""

import numpy as np
import pandas as pd

from scoring.processing import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

MIN_SAMPLES = 2
"""Fewest samples a signal parquet may have (need at least 2 for interpolation)."""

EVENT_TIME_COLUMN = "time_s"


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
    no NaN/Inf, and ``time`` strictly increasing.
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
    messages -- neither file is required, and each is validated on its own.
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
