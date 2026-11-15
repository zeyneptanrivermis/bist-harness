"""A–E harness-comparison experiment framework."""

from src.experiments.catalog import build_experiment_catalog
from src.experiments.runner import ExperimentRunner

__all__ = ["ExperimentRunner", "build_experiment_catalog"]
