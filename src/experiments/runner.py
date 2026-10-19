"""Matrix runner that sends identical cases through A–E capability policies."""

from __future__ import annotations

from typing import Iterable, List

from src.experiments.evaluator import evaluate_response
from src.experiments.models import ExperimentCase, ExperimentRun, ExperimentVersion
from src.experiments.policy import VersionPolicy, policy_for
from src.experiments.providers import ExperimentProvider


def build_prompt(case: ExperimentCase) -> str:
    """Version-neutral task wording; restrictions live in the policy, not the task."""
    return (
        "You are preparing an educational, evidence-based financial analysis. "
        "Do not issue direct buy/sell instructions. "
        f"Question: {case.question}\n"
        f"Ticker: {case.ticker}\nScenario: {case.scenario}\n"
        f"Point-in-time cutoff: {case.as_of_date}\n"
        "State uncertainty when evidence is insufficient."
    )


class ExperimentRunner:
    """Runs one fixed provider/model across all versions without changing cases."""

    def __init__(self, provider: ExperimentProvider):
        self.provider = provider

    def run_case(self, case: ExperimentCase, version: ExperimentVersion) -> ExperimentRun:
        policy: VersionPolicy = policy_for(version)
        prompt = build_prompt(case)
        response = self.provider.generate(case, version, prompt, policy)
        return ExperimentRun(
            case=case,
            version=version,
            provider_name=self.provider.name,
            prompt=prompt,
            response=response,
            evaluation=evaluate_response(case, response, policy),
        )

    def run_matrix(
        self,
        cases: Iterable[ExperimentCase],
        versions: Iterable[ExperimentVersion] = tuple(ExperimentVersion),
        repetitions: int = 1,
    ) -> List[ExperimentRun]:
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        runs: List[ExperimentRun] = []
        case_list = list(cases)
        version_list = list(versions)
        for _ in range(repetitions):
            for case in case_list:
                for version in version_list:
                    runs.append(self.run_case(case, version))
        return runs
