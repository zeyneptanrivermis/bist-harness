import unittest

import pandas as pd

from src.fundamentals import add_derived_metrics
from src.scenarios.fundamental_reaction import calculate_quarterly_changes
from src.scenarios.technical_reversals import MANDATORY_EVENT_FAMILIES, detect_reversal_events, summarize_reversals_by_regime
from tests.test_task3_harness import make_ohlcv


class ReversalFamilyTests(unittest.TestCase):
    def test_new_families_and_regimes(self) -> None:
        data = make_ohlcv()
        events = detect_reversal_events(data, market_df=data.copy())
        for col in ("Volatility_Regime", "Reversal_Strength_5d", "Market_Regime"):
            self.assertIn(col, events.columns)
        self.assertTrue(set(events["Event_Name"]) <= set(MANDATORY_EVENT_FAMILIES))
        self.assertIn("Volatility_Regime", summarize_reversals_by_regime(events).columns)

    def test_support_level_uses_only_prior_bars(self) -> None:
        data = make_ohlcv()
        cut = data.iloc[:150]
        a = detect_reversal_events(cut, event_families=["SUPPORT_TOUCH"])
        b = detect_reversal_events(data, event_families=["SUPPORT_TOUCH"])
        limit = cut.index[-12]
        self.assertEqual(len(a[a["Event_Date"] <= limit]), len(b[b["Event_Date"] <= limit]))


class FundamentalMetricTests(unittest.TestCase):
    def _frame(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame({"ticker": ticker, "kap_disclosure_date": pd.date_range("2025-01-01", periods=5, freq="QS"),
                             "revenue": [100.0] * 5, "ebitda": [10.0] * 5, "net_income": [5.0] * 5, "net_debt": [20.0] * 5,
                             "roe": 0.2, "ebitda_margin": 0.1})

    def test_derived_metrics_never_invent_external_values(self) -> None:
        out = add_derived_metrics(self._frame("TUPRS"), "TUPRS")
        self.assertAlmostEqual(out["net_margin"].iloc[-1], 0.05)
        self.assertAlmostEqual(out["net_debt_ebitda"].iloc[-1], 0.5)
        self.assertTrue(out["pe"].isna().all() and out["current_ratio"].isna().all())
        self.assertIn("net_margin_Change_QoQ", calculate_quarterly_changes(out).columns)

    def test_bank_industrial_ratios_are_disabled(self) -> None:
        out = add_derived_metrics(self._frame("AKBNK"), "AKBNK")
        self.assertTrue(out["net_debt_ebitda"].isna().all())
        self.assertIn("sector_note", out.columns)


if __name__ == "__main__":
    unittest.main()


class HarnessPermissionTests(unittest.TestCase):
    def test_state_tool_permissions_are_enforced(self) -> None:
        from src.harness.permissions import assert_tool_allowed
        assert_tool_allowed("RUN_BACKTEST", "run_backtest")
        with self.assertRaises(PermissionError):
            assert_tool_allowed("RUN_BACKTEST", "get_macro_context")
        with self.assertRaises(PermissionError):
            assert_tool_allowed("CHECK_DATA_QUALITY", "run_backtest")

    def test_every_planable_tool_is_permitted_in_scenario_state(self) -> None:
        from src.harness.permissions import STATE_TOOLS
        from src.llm.tool_loop import _TOOLS
        self.assertFalse(set(_TOOLS) - STATE_TOOLS["RUN_SCENARIO"] - STATE_TOOLS["CHECK_DATA_QUALITY"])

    def test_optional_load_states_are_ordered_and_logged(self) -> None:
        from src.harness.harness import HarnessSession
        from src.mcp_tools.models import AnalysisRequest
        data = make_ohlcv()
        s = HarnessSession(AnalysisRequest(ticker="ASELS", scenario="technical_reversal", as_of_date=str(data.index[220].date()), require_human_review=False))
        s.select_universe(); s.load_data(data, data.copy()); s.check_quality(); s.calculate()
        with self.assertRaises(RuntimeError):
            s.load_fundamentals()  # too early
        s.run_scenario(); s.load_fundamentals(); s.load_macro_context()
        with self.assertRaises(RuntimeError):
            s.load_fundamentals()  # cannot go back
        s.run_backtest(); s.build_evidence()
        self.assertEqual(s.transitions[6:9], ["LOAD_FUNDAMENTALS", "LOAD_MACRO_CONTEXT", "PREVIEW_STRATEGIES"])


class SectorBenchmarkTests(unittest.TestCase):
    def test_equal_weight_peer_index_excludes_the_stock_itself(self) -> None:
        from src.backtest.sector_benchmark import build_sector_benchmark, sector_benchmark_metrics
        idx = pd.date_range("2025-01-01", periods=10)
        up = pd.DataFrame({"Adj Close": [100.0 * 1.01 ** i for i in range(10)]}, index=idx)
        flat = pd.DataFrame({"Adj Close": [100.0] * 10}, index=idx)
        bench = build_sector_benchmark("BIMAS", {"BIMAS": flat, "MGROS": up, "MAVI": up})
        self.assertAlmostEqual(bench["Adj Close"].iloc[-1], 1.01 ** 9, places=6)
        self.assertIsNone(build_sector_benchmark("AKBNK", {"AKBNK": flat}))
        metrics = sector_benchmark_metrics(pd.Series([1.0] * 10, index=idx), bench)
        self.assertLess(metrics["Alpha_vs_Sector"], 0)


class SectorCarTests(unittest.TestCase):
    def test_sector_car_uses_peer_basket_and_is_nan_without_peers(self) -> None:
        from src.scenarios.fundamental_reaction import align_financial_disclosure_dates
        idx = pd.bdate_range("2024-09-01", periods=400)
        px = lambda g: pd.DataFrame({"Adj Close": [100.0 * g ** i for i in range(len(idx))]}, index=idx)
        stock, market, peer = px(1.001), px(1.0005), px(1.002)
        with_peer = align_financial_disclosure_dates("BIMAS", price_df=stock, market_df=market, peer_prices={"MGROS": peer})
        row = with_peer.iloc[0]
        self.assertAlmostEqual(row["Fwd_5d_Sector_Return"], 1.002 ** 5 - 1, places=6)
        self.assertAlmostEqual(row["Fwd_5d_Sector_CAR"], row["Fwd_5d_Stock_Return"] - row["Fwd_5d_Sector_Return"], places=9)
        without = align_financial_disclosure_dates("BIMAS", price_df=stock, market_df=market, peer_prices={})
        self.assertTrue(without["Fwd_5d_Sector_CAR"].isna().all())
