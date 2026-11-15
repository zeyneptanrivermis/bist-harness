from __future__ import annotations

import unittest

from src.experiments.catalog import build_experiment_catalog
from src.experiments.evaluator import evaluate_response
from src.experiments.models import ExperimentVersion, LLMResponse, ToolCallRecord
from src.experiments.policy import policy_for
from src.experiments.providers import DemonstrationProvider
from src.experiments.reporting import comparison_table
from src.experiments.runner import ExperimentRunner


class ExperimentFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = build_experiment_catalog()

    def test_catalog_has_normal_and_adversarial_cases(self) -> None:
        self.assertGreaterEqual(len(self.cases), 10)
        self.assertTrue(any(case.should_block for case in self.cases))
        self.assertTrue(any(not case.should_block for case in self.cases))

    def test_demo_matrix_runs_all_versions_and_has_expected_safety_separation(self) -> None:
        runs = ExperimentRunner(DemonstrationProvider()).run_matrix(self.cases)
        self.assertEqual(len(runs), len(self.cases) * len(ExperimentVersion))
        summary = comparison_table(runs).set_index("version")
        self.assertLess(summary.loc[ExperimentVersion.RAW_LLM.value, "passed_pct"], 100)
        self.assertEqual(summary.loc[ExperimentVersion.EVIDENCE.value, "passed_pct"], 100)
        self.assertEqual(summary.loc[ExperimentVersion.HUMAN_REVIEW.value, "passed_pct"], 100)

    def test_evaluator_detects_unauthorized_tool_and_future_cutoff(self) -> None:
        case = next(item for item in self.cases if item.case_id == "TECHNICAL_NORMAL")
        response = LLMResponse(
            content="WATCH.",
            tool_calls=[ToolCallRecord(name="not_a_tool", arguments={"as_of_date": "2025-10-01"})],
        )
        evaluation = evaluate_response(case, response, policy_for(ExperimentVersion.TOOLS))
        self.assertTrue(evaluation.invalid_tool_call)
        self.assertTrue(evaluation.look_ahead_violation)

    def test_evidence_version_requires_committed_evidence(self) -> None:
        case = next(item for item in self.cases if item.case_id == "TECHNICAL_NORMAL")
        response = LLMResponse(
            content="WATCH.",
            state_history=["START", "SELECT_UNIVERSE", "LOAD_DATA", "CHECK_DATA_QUALITY"],
            quality_status="PASS",
        )
        evaluation = evaluate_response(case, response, policy_for(ExperimentVersion.EVIDENCE))
        self.assertTrue(evaluation.unsupported_conclusion)


if __name__ == "__main__":
    unittest.main()
