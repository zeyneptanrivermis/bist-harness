from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import ANALYSIS_AS_OF_DATE as AS_OF, PROCESSED_DATA_DIR
from src.llm import evaluate as E
from src.llm.agent import _expected_status, check_answer, execute_plan, ollama_chat_fn, run_agent, unsupported_numbers
from src.llm.dataset import FAMILIES, HOLDOUT_TICKERS, build_cases, case_records
from src.llm.summaries import draft_summary
from src.llm.tool_loop import SafeProjectToolRouter

WEEKDAY = {
    "tool": "test_weekday_pattern", "available": True, "ticker": "THYAO", "pattern": "mon_tue_wed_down", "warnings": ["Small sample: 3."], "suggested_status": "REJECT_SIGNAL",
    "period": {"start": "2024-01-01", "end": "2025-12-31"}, "multiple_testing": {"tests_run": 1, "bonferroni_alpha": 0.05},
    "in_sample": {"occurrences": 19, "weeks": 87, "thu_fri_both_up": 6, "thu": {"n": 19, "mean": 0.0121, "median": 0.0097}, "fri": {"n": 19, "mean": 0.0016, "median": 0.0},
                  "unconditional_thu_mean": 0.002, "unconditional_fri_mean": 0.0033, "p_value_vs_other_weeks": 0.2585, "tests_run": 1},
}
BACKTEST = {"tool": "run_backtest", "available": True, "ticker": "THYAO", "strategy": "supertrend_long_cash_t_plus_1", "warnings": [], "suggested_status": "WATCH",
            "metrics": {"Total_Trades": 8, "Total_Return": -0.011, "Benchmark_Total_Return": 0.477, "Sharpe_Ratio": -1.71, "Max_Drawdown": -0.286}}
PLAN = json.dumps({"tool_calls": [{"tool_name": "test_weekday_pattern", "arguments": {"ticker": "THYAO", "pattern": "mon_tue_wed_down", "start_date": "2024-01-01", "end_date": "2025-12-31"}}]})


class StubRouter(SafeProjectToolRouter):
    """Validation is the real one; tool execution returns canned outputs so no price data is needed."""

    def __init__(self, gate: str = "PASS") -> None:
        super().__init__()
        self.gate, self.ran = gate, []

    def execute(self, tool_name, arguments, *, as_of_date):
        self.ran.append(tool_name)
        if tool_name == "check_data_quality":
            return {"tool": tool_name, "available": True, "ticker": arguments["ticker"], "gate_status": self.gate, "warnings": [], "suggested_status": "WATCH"}
        return {"test_weekday_pattern": WEEKDAY, "run_backtest": BACKTEST}[tool_name]


def _model(second: str, first: str = PLAN):
    replies = iter([first, second])
    return lambda messages: next(replies)


class GuardTests(unittest.TestCase):
    EVIDENCE = [{"evidence_id": "E1", "source_tool": "x", "values": {"mean": 0.0557, "name": "XU100", "period": "2024Q3"}}]

    def test_numbers_must_trace_to_evidence_or_the_question(self) -> None:
        self.assertEqual(unsupported_numbers("E1: return 5.6% versus XU100 in 2024Q3 (2026-09-10)", self.EVIDENCE), [])
        self.assertEqual(unsupported_numbers("return will be 77.7%", self.EVIDENCE), ["77.7"])
        self.assertEqual(unsupported_numbers("over 33.3 days", self.EVIDENCE, question="the last 33.3 days"), [])

    def test_check_answer_flags_each_rule(self) -> None:
        bad = check_answer("Status: MOON. Buy it, 2026-12-01 gain 77.7%", self.EVIDENCE, "WATCH", AS_OF)
        self.assertEqual(set(bad), {"bad_status", "trade_advice", "future_date", "missing_evidence_id", "numeric_hallucination"})
        self.assertIn("unsupported_conclusion", check_answer("Status: WATCH. E1", self.EVIDENCE, "INVESTIGATE", AS_OF))
        self.assertIn("missing_evidence_id", check_answer("Status: WATCH. See E10.", self.EVIDENCE, "WATCH", AS_OF))  # E10 must not satisfy E1
        self.assertIn("missing_advice_refusal", check_answer("Status: WATCH. E1", self.EVIDENCE, "WATCH", AS_OF, question="Should I buy it?"))

    def test_deterministic_draft_passes_its_own_guard(self) -> None:
        evidence = [{"evidence_id": "E1", "source_tool": "test_weekday_pattern", "values": WEEKDAY}, {"evidence_id": "E2", "source_tool": "run_backtest", "values": BACKTEST}]
        question = "Should I buy THYAO? Does the pattern hold?"
        self.assertEqual(check_answer(draft_summary("REJECT_SIGNAL", evidence, question), evidence, "REJECT_SIGNAL", AS_OF, question), [])


class AgentLoopTests(unittest.TestCase):
    def _run(self, chat_fn, router=None, question="Does THYAO show the pattern?"):
        with patch("src.llm.agent.log_decision", return_value=Path("log.json")):
            return run_agent(question, chat_fn, as_of_date=AS_OF, router=router or StubRouter())

    def test_states_run_in_order_gate_first_and_backtest_is_added_by_the_harness(self) -> None:
        router = StubRouter()
        result = self._run(_model("x"), router)
        self.assertEqual(result["state_history"], ["START", "SELECT_UNIVERSE", "LOAD_DATA", "CHECK_DATA_QUALITY", "CALCULATE", "RUN_SCENARIO", "PREVIEW_STRATEGIES", "RUN_BACKTEST", "RISK_GATE", "BUILD_EVIDENCE", "GENERATE_EXPLANATION", "LOG_AND_CLOSE"])
        self.assertEqual(router.ran, ["check_data_quality", "test_weekday_pattern", "run_backtest"])
        self.assertEqual(result["action"], "REJECT_SIGNAL")

    @unittest.skipUnless((PROCESSED_DATA_DIR / "MAVI.parquet").exists(), "needs the price cache")
    def test_sector_plan_gets_macro_context_and_pattern_reports_costs(self) -> None:
        sector = [("rank_sector_relative_strength", {"tickers": ["BIMAS", "MGROS", "MAVI"], "horizon_days": 20, "start_date": "2024-01-01", "end_date": AS_OF})]
        run = execute_plan(sector, AS_OF, SafeProjectToolRouter())
        self.assertIn("LOAD_MACRO_CONTEXT", run["states"])
        self.assertEqual([e["source_tool"] for e in run["evidence"]], ["check_data_quality", "rank_sector_relative_strength", "get_macro_context", "run_backtest"])
        self.assertEqual(run["evidence"][2]["values"]["fx_relationship"]["ticker"], "MAVI")
        pattern = SafeProjectToolRouter().execute("test_weekday_pattern", {"ticker": "THYAO", "pattern": "mon_tue_wed_down", "start_date": "2024-01-01", "end_date": "2025-12-31", "oos_start_date": "2026-01-01"}, as_of_date=AS_OF)
        p = pattern["in_sample"]
        self.assertAlmostEqual(p["wed_fri_net_mean"], p["wed_fri_mean"] - 0.003, places=3)
        self.assertEqual(pattern["multiple_testing"]["tests_run"], 32)
        evidence = [{"evidence_id": "E1", "values": pattern}]
        text = draft_summary("REJECT_SIGNAL", evidence)
        self.assertIn("round-trip cost", text)
        self.assertIn("median", text)
        self.assertEqual(check_answer(text, evidence, "REJECT_SIGNAL", AS_OF), [])  # the template must survive its own guard

    def test_good_answer_is_kept_and_bad_answer_is_replaced_but_raw_is_kept(self) -> None:
        run = execute_plan(StubRouter().validate_plan(PLAN, as_of_date=AS_OF), AS_OF, StubRouter())
        good = draft_summary(run["expected"], run["evidence"], "Does THYAO show the pattern?")
        kept = self._run(_model(good))
        self.assertEqual((kept["answer"], kept["violations"]), (good, []))
        replaced = self._run(_model("Status: WATCH. Buy now for 77.7%."))
        self.assertIn("trade_advice", replaced["violations"])
        self.assertEqual(replaced["raw_answer"], "Status: WATCH. Buy now for 77.7%.")
        self.assertEqual(replaced["answer"], good)

    def test_blocked_gate_stops_before_any_tool_and_yields_analysis_unsafe(self) -> None:
        router = StubRouter(gate="HARD_BLOCK")
        result = self._run(_model("never used"), router)
        self.assertEqual(router.ran, ["check_data_quality"])
        self.assertEqual(result["state_history"][-1], "BLOCKED")
        self.assertTrue(result["answer"].startswith("Status: ANALYSIS_UNSAFE"))

    def test_invalid_plans_never_reach_a_tool(self) -> None:
        for reply in ("not json", '{"tool_calls": [{"tool_name": "test_weekday_pattern", "arguments": {}}]}',
                      '{"tool_calls": [{"tool_name": "test_weekday_pattern", "arguments": {"ticker": "THYAO", "end_date": "2026-09-11"}}]}'):
            router = StubRouter()
            result = self._run(_model("x", first=reply), router)
            self.assertEqual((result["tool_valid"], result["violations"], router.ran, result["action"]), (False, ["invalid_tool_call"], [], "ANALYSIS_UNSAFE"))

    def test_prose_refusal_is_allowed_and_is_not_a_tool_call(self) -> None:
        result = self._run(_model("x", first="Status: ANALYSIS_UNSAFE. The request needs data after the analysis date."))
        self.assertEqual((result["tool_valid"], result["violations"], result["action"]), (None, [], "ANALYSIS_UNSAFE"))

    def test_advice_question_needs_the_refusal_sentence(self) -> None:
        result = self._run(_model("Status: REJECT_SIGNAL. E1 E2 E3"), question="Should I buy THYAO tomorrow? Give me a price target.")
        self.assertIn("missing_advice_refusal", result["violations"])
        self.assertIn("cannot give investment advice", result["answer"])

    def test_same_question_gives_the_same_log_id(self) -> None:
        ids = []
        with patch("src.llm.agent.log_decision", side_effect=lambda record: ids.append(record["analysis_id"]) or Path("x")):
            for _ in range(2):
                run_agent("Does THYAO show the pattern?", _model("x"), as_of_date=AS_OF, router=StubRouter())
        self.assertEqual(ids[0], ids[1])


class DatasetTests(unittest.TestCase):
    def test_plans_are_valid_cover_every_family_and_never_use_holdout_tickers(self) -> None:
        router = SafeProjectToolRouter()
        cases = build_cases(per_family=12)
        self.assertEqual({c["family"] for c in cases}, set(FAMILIES))
        for case in cases:
            text = json.dumps(case["target"])
            self.assertFalse(any(t in text or t in case["question"] for t in HOLDOUT_TICKERS), case["case_id"])
            if not isinstance(case["target"], str):
                router.validate_plan(json.dumps({"tool_calls": case["target"]}), as_of_date=case["as_of"])
            else:
                self.assertTrue(case["target"].startswith("Status: ANALYSIS_UNSAFE"))

    def test_case_records_pair_a_plan_with_a_report_that_matches_the_evidence(self) -> None:
        case = {"case_id": "WEEKDAY_0001", "family": "weekday", "question": "Does THYAO show the pattern?", "as_of": AS_OF, "target": json.loads(PLAN)["tool_calls"]}
        records, reason = case_records(case, StubRouter(), with_report=True)
        self.assertIsNone(reason)
        self.assertEqual([r["metadata"]["task"] for r in records], ["tool_selection", "evidence_report"])
        self.assertTrue(records[1]["messages"][2]["content"].startswith("Status: REJECT_SIGNAL."))


class EvaluateTests(unittest.TestCase):
    def test_plan_matches_requires_tool_tickers_and_dates(self) -> None:
        want = [E._c("rank_sector_relative_strength", tickers=["A", "B"], start_date="2024-01-01")]
        ok = [{"tool_name": "rank_sector_relative_strength", "arguments": {"tickers": ["B", "A"], "start_date": "2024-01-01", "horizon_days": 20}}]
        self.assertTrue(E.plan_matches(ok, want))
        self.assertFalse(E.plan_matches([{"tool_name": "rank_sector_relative_strength", "arguments": {"tickers": ["A"], "start_date": "2024-01-01"}}], want))
        self.assertFalse(E.plan_matches([{"tool_name": "run_backtest", "arguments": {}}], want))

    @unittest.skipUnless((PROCESSED_DATA_DIR / "THYAO.parquet").exists(), "needs the cached prices")
    def test_the_eval_set_runs_on_real_data_and_a_perfect_model_scores_zero(self) -> None:
        def oracle(messages):
            user = messages[1]["content"]
            question = user.split("Question: ", 1)[1].split("\n")[0]
            if "Committed evidence (JSON):" in user:
                evidence = json.loads(user.split("Committed evidence (JSON):\n")[1].split("\n\nWrite")[0])
                return draft_summary(_expected_status(evidence), evidence, question)
            item = next(i for i in E.EVAL_ITEMS if i["q"].startswith(question))
            return json.dumps({"tool_calls": item["expect"]}) if item["expect"] else "Status: ANALYSIS_UNSAFE. Refused."

        with patch("src.llm.agent.log_decision", return_value=Path("x")):
            rates = E.evaluate(oracle, "oracle")["rates"]
        self.assertEqual(sum(rates.values()), 0, rates)

    @unittest.skipUnless((PROCESSED_DATA_DIR / "KCHOL.parquet").exists(), "needs the cached prices")
    def test_only_the_gated_versions_block_broken_data_and_a_has_no_plan(self) -> None:
        def planner(messages):
            user = messages[1]["content"]
            if "Committed evidence (JSON):" in user:
                return "Status: WATCH. E1"
            item = next(i for i in E.CORRUPT_ITEMS if i["q"] in user)
            return json.dumps({"tool_calls": item["expect"]})

        for item in E.CORRUPT_ITEMS:
            with self.subTest(item["corrupt"]):
                status = {v: E._ask(item, planner, v).get("action") for v in "BDE"}
                self.assertEqual(status["D"], "ANALYSIS_UNSAFE")
                self.assertEqual(status["E"], "ANALYSIS_UNSAFE")
                self.assertNotEqual(status["B"], "ANALYSIS_UNSAFE")  # without the gate the broken data is analysed
        a = run_agent(E.CORRUPT_ITEMS[0]["q"], lambda m: "Status: WATCH. E1", log=False, version="A")
        self.assertEqual((a["tool_call"], [e["source_tool"] for e in a["evidence"]]), (None, ["get_stock_history"]))
        self.assertIsNone(a["human_review"])
        self.assertEqual(run_agent(E.EVAL_ITEMS[2]["q"], lambda m: json.dumps({"tool_calls": E.EVAL_ITEMS[2]["expect"]}) if "Committed evidence" not in m[1]["content"] else "x",
                                   log=False, version="E")["human_review"]["status"], "PENDING")


class OllamaTests(unittest.TestCase):
    def test_ollama_chat_fn_posts_to_localhost_and_returns_the_message(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"], captured["body"] = request.full_url, json.loads(request.data)
            return io.BytesIO(json.dumps({"message": {"content": " hi "}}).encode())

        with patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(ollama_chat_fn("m")([{"role": "user", "content": "q"}]), "hi")
        self.assertTrue(captured["url"].startswith("http://localhost:11434/"))
        self.assertEqual((captured["body"]["model"], captured["body"]["options"]["temperature"], captured["body"]["stream"]), ("m", 0, False))


if __name__ == "__main__":
    unittest.main()
