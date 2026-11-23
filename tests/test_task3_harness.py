from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.harness.harness import HarnessSession, HarnessState
from src.harness.quality import check_data_quality
from src.harness.replay import log_decision, replay_analysis
from src.mcp_tools.models import AnalysisRequest, HumanReview
from src.mcp_tools.serialization import frame_payload
from src.backtest.engine import run_backtest
from src.fundamentals import get_fundamental_metrics
from src.indicators import calculate_all_indicators
from src.scenarios.sector_laggard import rank_sector_laggards
from src.scenarios.technical_reversals import detect_reversal_events
from src.scenarios.weekday_patterns import test_weekday_pattern


def make_ohlcv(periods: int = 260) -> pd.DataFrame:
    """Deterministic, valid daily OHLCV input independent of external market data."""
    index = pd.bdate_range("2025-01-02", periods=periods)
    close = 100 + np.linspace(0, 30, periods) + np.sin(np.arange(periods) / 5)
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": np.full(periods, 1_000_000),
        },
        index=index,
    )


class QualityGateTests(unittest.TestCase):
    def test_cutoff_excludes_future_rows_and_detects_forward_data(self) -> None:
        data = make_ohlcv()
        as_of = str(data.index[200].date())
        report = check_data_quality(data, as_of, metadata={"signal_shift": 1})
        self.assertEqual(report.status, "PASS_WITH_WARNINGS")
        self.assertEqual(len(report.filtered_data), 201)
        self.assertIn("FUTURE_ROWS_EXCLUDED", {issue.code for issue in report.issues})

        unsafe = check_data_quality(data, as_of, metadata={"uses_forward_data": True})
        self.assertEqual(unsafe.status, "HARD_BLOCK")


class HarnessTests(unittest.TestCase):
    def test_state_machine_prevents_skipping_quality_and_produces_a_reviewable_log(self) -> None:
        data = make_ohlcv()
        request = AnalysisRequest(
            ticker="ASELS",
            scenario="technical_reversal",
            as_of_date=str(data.index[220].date()),
        )
        session = HarnessSession(request)
        with self.assertRaises(RuntimeError):
            session.calculate()

        session.select_universe()
        session.load_data(data, data.copy())
        report = session.check_quality()
        self.assertIn(report.status, {"PASS", "PASS_WITH_WARNINGS", "INVESTIGATE"})
        session.calculate()
        session.run_scenario()
        session.run_backtest()
        session.build_evidence()
        self.assertEqual(session.state, HarnessState.BUILD_EVIDENCE)
        self.assertGreaterEqual(len(session.evidence), 2)
        self.assertIn(session.action, {"TECHNICAL_REVERSAL_CANDIDATE", "WATCH"})

        session.generate_explanation()
        self.assertIn("PREVIEW_STRATEGIES", session.transitions)
        self.assertIn("RISK_GATE", session.transitions)
        session.submit_human_review(HumanReview(decision="APPROVED", reviewer="test-reviewer", rationale="Evidence and quality gate were reviewed."))
        with patch("src.harness.harness.log_decision", return_value=Path("test-replay.json")):
            path = session.log_and_close()
        self.assertEqual(path, "test-replay.json")
        self.assertEqual(session.state, HarnessState.CLOSED)


class ReplayAndSerializationTests(unittest.TestCase):
    def test_replay_is_tamper_evident_and_frame_payload_is_bounded(self) -> None:
        # Workspace-local instead of the sandboxed OS temp folder, which is read-only.
        path = log_decision({"analysis_id": "replay_test", "value": 42}, Path("data") / "test_analysis_logs")
        self.assertEqual(replay_analysis(path)["value"], 42)
        path.write_text('{"analysis_id":"replay_test","value":43,"replay_sha256":"bad"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            replay_analysis(path)

        payload = frame_payload(make_ohlcv(20), max_rows=5)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["returned_row_count"], 5)


class ExistingEngineRegressionTests(unittest.TestCase):
    def test_existing_indicators_scenarios_fundamentals_and_backtest_still_work(self) -> None:
        price = make_ohlcv()
        benchmark = make_ohlcv() * 1.03
        benchmark["Volume"] = 1_000_000
        enriched = calculate_all_indicators(price)
        self.assertIn("RSI_14", enriched.columns)
        self.assertIn("Supertrend_Direction", enriched.columns)

        self.assertFalse(test_weekday_pattern(price).empty)
        events = detect_reversal_events(enriched, market_df=benchmark)
        self.assertIsInstance(events, pd.DataFrame)
        ranks = rank_sector_laggards(
            date=str(price.index[-1].date()),
            price_dict={"THYAO": price, "PGSUS": price * 1.01},
        )
        self.assertEqual(set(ranks["Ticker"]), {"THYAO", "PGSUS"})

        result = run_backtest((enriched["Supertrend_Direction"] == 1).astype(int), price, benchmark_df=benchmark)
        self.assertIn("Sharpe_Ratio", result.metrics)
        self.assertFalse(get_fundamental_metrics("TUPRS").empty)


if __name__ == "__main__":
    unittest.main()
