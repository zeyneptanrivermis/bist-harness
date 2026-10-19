"""Machine-readable artifacts and compact comparison tables for the final report."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.experiments.models import ExperimentRun
from src.mcp_tools.serialization import json_safe


_BOOLEAN_METRICS = [
    "passed",
    "numeric_hallucination",
    "invalid_tool_call",
    "look_ahead_violation",
    "unsupported_conclusion",
    "direct_trade_advice",
    "state_bypass",
    "quality_gate_bypass",
    "missing_replay",
    "missing_human_review",
]


def runs_to_frame(runs: Iterable[ExperimentRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        evaluation = run.evaluation.model_dump()
        row = {
            "case_id": run.case.case_id,
            "trap": run.case.trap.value,
            "version": run.version.value,
            "provider": run.provider_name,
            "latency_ms": run.response.latency_ms,
            "input_tokens": run.response.input_tokens,
            "output_tokens": run.response.output_tokens,
            "estimated_cost_usd": run.response.estimated_cost_usd,
            **{metric: evaluation[metric] for metric in _BOOLEAN_METRICS},
            "issue_count": len(evaluation["issues"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(runs: Iterable[ExperimentRun]) -> pd.DataFrame:
    frame = runs_to_frame(runs)
    if frame.empty:
        return frame
    grouped = frame.groupby("version", as_index=False)[_BOOLEAN_METRICS + ["issue_count", "latency_ms", "estimated_cost_usd"]].mean(numeric_only=True)
    for metric in _BOOLEAN_METRICS:
        grouped[metric] = grouped[metric] * 100
    return grouped.rename(columns={metric: f"{metric}_pct" for metric in _BOOLEAN_METRICS})


def write_report_artifacts(runs: Iterable[ExperimentRun], output_dir: str | Path) -> dict[str, Path]:
    """Write replayable JSONL and report-ready CSV/PNG files."""
    run_list = list(runs)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    jsonl_path = target / "ae_experiment_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for run in run_list:
            handle.write(json.dumps(json_safe(run.model_dump()), ensure_ascii=False, sort_keys=True) + "\n")

    raw_path = target / "ae_experiment_runs.csv"
    runs_to_frame(run_list).to_csv(raw_path, index=False)
    summary_path = target / "ae_comparison_summary.csv"
    summary = comparison_table(run_list)
    summary.to_csv(summary_path, index=False)

    chart_path = target / "ae_comparison_pass_rate.png"
    if not summary.empty:
        # Keep Matplotlib's cache within the selected artifact folder. This is
        # required in sandboxed Windows environments where the user home is not writable.
        matplotlib_cache = target / ".matplotlib"
        matplotlib_cache.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
        import matplotlib.pyplot as plt

        ax = summary.plot.bar(x="version", y="passed_pct", legend=False, ylim=(0, 100), color="#2E86AB")
        ax.set_ylabel("Passed cases (%)")
        ax.set_xlabel("Architecture version")
        ax.set_title("A–E Harness Comparison")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
    return {"jsonl": jsonl_path, "runs_csv": raw_path, "summary_csv": summary_path, "chart": chart_path}
