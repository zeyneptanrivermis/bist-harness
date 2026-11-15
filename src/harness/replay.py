"""Tamper-evident JSON decision logs and replay validation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from src.config import ANALYSIS_LOG_DIR
from src.mcp_tools.serialization import json_safe


def _canonical_json(value: Dict[str, Any]) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def log_decision(record: Dict[str, Any], log_dir: Path = ANALYSIS_LOG_DIR) -> Path:
    """Persist an immutable analysis record with a content hash."""
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = json_safe(record)
    payload.pop("replay_sha256", None)
    payload["replay_sha256"] = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    path = log_dir / f"analysis_{payload.get('analysis_id', uuid4().hex)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def replay_analysis(path: str | Path) -> Dict[str, Any]:
    """Load a record only if its stored content hash still matches."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_hash = record.pop("replay_sha256", None)
    computed_hash = sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    if stored_hash != computed_hash:
        raise ValueError("Replay log integrity check failed: content hash does not match.")
    record["replay_sha256"] = stored_hash
    return record
