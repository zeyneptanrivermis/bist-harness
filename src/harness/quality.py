"""Point-in-time data quality gate used before analytical tools may run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import MAX_ALLOWED_MISSING_RATIO, MIN_REQUIRED_HISTORY_DAYS


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    status: str
    as_of_date: Optional[str]
    rows_evaluated: int
    data_fingerprint: Optional[str]
    issues: List[QualityIssue]
    filtered_data: pd.DataFrame = field(repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        # Do not deepcopy the potentially large DataFrame merely to omit it.
        return {
            "status": self.status,
            "as_of_date": self.as_of_date,
            "rows_evaluated": self.rows_evaluated,
            "data_fingerprint": self.data_fingerprint,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _fingerprint(frame: pd.DataFrame) -> Optional[str]:
    if frame.empty:
        return None
    stable = frame.copy()
    stable.index = stable.index.astype(str)
    serialized = stable.to_csv(index=True, float_format="%.12g").encode("utf-8")
    return sha256(serialized).hexdigest()


def _resolve_status(issues: List[QualityIssue]) -> str:
    severities = {issue.severity for issue in issues}
    if "HARD_BLOCK" in severities:
        return "HARD_BLOCK"
    if "ANALYSIS_UNSAFE" in severities:
        return "ANALYSIS_UNSAFE"
    if "INVESTIGATE" in severities:
        return "INVESTIGATE"
    if "WARNING" in severities:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def check_data_quality(
    data: pd.DataFrame,
    as_of_date: Optional[str],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    required_history_days: int = MIN_REQUIRED_HISTORY_DAYS,
    max_missing_ratio: float = MAX_ALLOWED_MISSING_RATIO,
) -> QualityReport:
    """Validate OHLCV data and enforce a strict point-in-time cutoff.

    Rows after ``as_of_date`` are removed before any downstream computation. A
    future-looking feature declaration or a non-positive signal shift is a hard
    safety failure; the harness must not continue in either case.
    """
    metadata = metadata or {}
    issues: List[QualityIssue] = []
    required_columns = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}

    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append(QualityIssue("INVALID_INDEX", "HARD_BLOCK", "Price data must have a DatetimeIndex."))
        return QualityReport(_resolve_status(issues), as_of_date, 0, None, issues, data.iloc[0:0].copy())

    if as_of_date is None:
        issues.append(QualityIssue("AS_OF_REQUIRED", "ANALYSIS_UNSAFE", "An as_of_date is required for reproducible point-in-time analysis."))
        cutoff = data.copy()
    else:
        cutoff_timestamp = pd.Timestamp(as_of_date)
        cutoff = data.loc[data.index <= cutoff_timestamp].copy()
        excluded = int((data.index > cutoff_timestamp).sum())
        if excluded:
            issues.append(QualityIssue("FUTURE_ROWS_EXCLUDED", "WARNING", "Rows after as_of_date were excluded before analysis.", {"count": excluded}))

    missing_columns = sorted(required_columns - set(cutoff.columns))
    if missing_columns:
        issues.append(QualityIssue("MISSING_COLUMNS", "HARD_BLOCK", "Required OHLCV columns are missing.", {"columns": missing_columns}))
    if cutoff.empty:
        issues.append(QualityIssue("NO_DATA_AT_AS_OF", "HARD_BLOCK", "No price observations exist on or before as_of_date."))
    if not cutoff.index.is_monotonic_increasing:
        issues.append(QualityIssue("UNSORTED_DATES", "INVESTIGATE", "Dates are not sorted ascending; data must be normalized before analysis."))
    if cutoff.index.has_duplicates:
        issues.append(QualityIssue("DUPLICATE_DATES", "HARD_BLOCK", "Duplicate trading dates were found."))

    if not missing_columns and not cutoff.empty:
        numeric = cutoff[list(required_columns)]
        missing_ratio = float(numeric.isna().mean().mean())
        if missing_ratio > max_missing_ratio:
            issues.append(QualityIssue("EXCESS_MISSING_DATA", "ANALYSIS_UNSAFE", "Missing OHLCV values exceed the configured threshold.", {"missing_ratio": missing_ratio, "threshold": max_missing_ratio}))
        elif missing_ratio > 0:
            issues.append(QualityIssue("MISSING_DATA", "WARNING", "Some OHLCV values are missing.", {"missing_ratio": missing_ratio}))

        if len(cutoff) < required_history_days:
            issues.append(QualityIssue("INSUFFICIENT_HISTORY", "HARD_BLOCK", "Insufficient history for the configured indicators.", {"rows": len(cutoff), "minimum": required_history_days}))

        price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        if (numeric[price_columns] <= 0).any().any() or (numeric["Volume"] < 0).any():
            issues.append(QualityIssue("NON_POSITIVE_VALUES", "HARD_BLOCK", "Prices must be positive and volume cannot be negative."))
        invalid_ohlc = (numeric["High"] < numeric[["Open", "Close", "Low"]].max(axis=1)) | (numeric["Low"] > numeric[["Open", "Close", "High"]].min(axis=1))
        if invalid_ohlc.fillna(False).any():
            issues.append(QualityIssue("INVALID_OHLC_RANGE", "HARD_BLOCK", "High/low values are inconsistent with open or close."))

        returns = numeric["Adj Close"].pct_change().abs()
        if (returns > 0.50).fillna(False).any():
            issues.append(QualityIssue("EXTREME_RETURN", "INVESTIGATE", "A daily adjusted-price move above 50% requires corporate-action review."))

    if metadata.get("uses_forward_data"):
        issues.append(QualityIssue("FORWARD_DATA_DECLARED", "HARD_BLOCK", "The request declares use of unavailable future data."))
    if metadata.get("signal_shift", 1) < 1:
        issues.append(QualityIssue("INVALID_SIGNAL_SHIFT", "ANALYSIS_UNSAFE", "Signal execution must be delayed by at least one trading session."))
    if metadata.get("source") == "curated_local_dataset":
        issues.append(QualityIssue("CURATED_SOURCE", "WARNING", "Curated fundamentals need source identifiers for external verification."))

    return QualityReport(
        status=_resolve_status(issues),
        as_of_date=as_of_date,
        rows_evaluated=len(cutoff),
        data_fingerprint=_fingerprint(cutoff),
        issues=issues,
        filtered_data=cutoff,
    )
