"""Local entry point for a reproducible simulated A–E experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.catalog import build_experiment_catalog
from src.experiments.providers import DemonstrationProvider
from src.experiments.reporting import write_report_artifacts
from src.experiments.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible A–E harness comparison demo.")
    parser.add_argument("--output-dir", default="reports/ae_harness_demo", help="Directory for JSONL, CSV, and PNG artifacts.")
    parser.add_argument("--repetitions", type=int, default=1, help="Repeat each case/version pair.")
    args = parser.parse_args()

    runner = ExperimentRunner(DemonstrationProvider())
    runs = runner.run_matrix(build_experiment_catalog(), repetitions=args.repetitions)
    artifacts = write_report_artifacts(runs, Path(args.output_dir))
    print(f"Completed {len(runs)} simulated runs.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
