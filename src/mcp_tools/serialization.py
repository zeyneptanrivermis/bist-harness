"""Conversion helpers that prevent pandas/numpy values leaking into MCP JSON."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict
import math

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Convert common analysis values to deterministic JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def frame_payload(frame: pd.DataFrame, max_rows: int = 500) -> Dict[str, Any]:
    """Return a bounded tabular response suitable for a language-model tool call."""
    clipped = frame.head(max_rows).copy()
    if isinstance(clipped.index, pd.DatetimeIndex):
        clipped = clipped.reset_index().rename(columns={clipped.index.name or "index": "Date"})
    return {
        "row_count": int(len(frame)),
        "returned_row_count": int(len(clipped)),
        "truncated": bool(len(frame) > max_rows),
        "columns": [str(column) for column in clipped.columns],
        "records": json_safe(clipped),
    }
