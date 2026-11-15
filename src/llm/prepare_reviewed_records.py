"""Convert reviewed CSV rows into validated records for local-LLM training.

Workflow:
1. Open candidates_review.csv in Excel/LibreOffice.
2. For each approved row, set human_reviewed to true, add reviewer, and write
   the final English text in reviewed_summary.
3. Run this module. It keeps only complete, approved rows and writes the JSONL
   consumed by notebooks/05_local_llm_finetuning.ipynb.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.llm.training_data import validate_records


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _nonempty(value: Any) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_reviewed_records(
    candidates_path: str | Path,
    review_csv_path: str | Path,
    output_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build validated records from complete review rows; return excluded rows."""
    candidates = _read_jsonl(candidates_path)
    by_case_id = {record["metadata"]["case_id"]: record for record in candidates}
    review_frame = pd.read_csv(review_csv_path, dtype=str, keep_default_na=False)
    required_columns = {"case_id", "human_reviewed", "reviewed_summary", "reviewer"}
    missing = required_columns - set(review_frame.columns)
    if missing:
        raise ValueError(f"Review CSV is missing required columns: {', '.join(sorted(missing))}")

    approved: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in review_frame.iterrows():
        case_id = str(row["case_id"]).strip()
        if case_id not in by_case_id:
            excluded.append({"case_id": case_id, "reason": "case_id is absent from candidates.jsonl"})
            continue
        if case_id in seen:
            excluded.append({"case_id": case_id, "reason": "duplicate review row"})
            continue
        seen.add(case_id)
        if not _is_true(row["human_reviewed"]):
            excluded.append({"case_id": case_id, "reason": "human_reviewed is not true"})
            continue
        if not _nonempty(row["reviewed_summary"]):
            excluded.append({"case_id": case_id, "reason": "reviewed_summary is empty"})
            continue
        if not _nonempty(row["reviewer"]):
            excluded.append({"case_id": case_id, "reason": "reviewer is empty"})
            continue

        source = by_case_id[case_id]
        record = json.loads(json.dumps(source))  # JSON-compatible deep copy
        record["messages"][-1]["content"] = str(row["reviewed_summary"]).strip()
        record["metadata"]["human_reviewed"] = True
        record["metadata"]["reviewer"] = str(row["reviewer"]).strip()
        if "review_notes" in row and _nonempty(row["review_notes"]):
            record["metadata"]["review_notes"] = str(row["review_notes"]).strip()
        approved.append(record)

    validate_records(approved, require_human_review=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in approved) + ("\n" if approved else ""), encoding="utf-8")
    return approved, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Create validated reviewed_records.jsonl from the candidates review CSV.")
    parser.add_argument("--candidates", default="data/llm/candidates/candidates.jsonl")
    parser.add_argument("--review-csv", default="data/llm/candidates/candidates_review.csv")
    parser.add_argument("--output", default="data/llm/reviewed_records.jsonl")
    args = parser.parse_args()
    approved, excluded = prepare_reviewed_records(args.candidates, args.review_csv, args.output)
    print(f"Wrote {len(approved)} reviewed records to {args.output}")
    print(f"Excluded {len(excluded)} incomplete or invalid review rows.")
    if excluded:
        print("First exclusions:")
        for item in excluded[:10]:
            print(f"- {item['case_id']}: {item['reason']}")


if __name__ == "__main__":
    main()
