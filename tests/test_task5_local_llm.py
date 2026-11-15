from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.llm.build_corpus import _ensure_market_cache, build_corpus, select_common_cutoffs
from src.llm.prepare_reviewed_records import prepare_reviewed_records
from src.llm.tool_loop import SafeProjectToolRouter, ToolRequestError
from src.llm.training_data import generate_seed_records, stable_split, validate_records, write_jsonl_splits


class LocalLlmTrainingDataTests(unittest.TestCase):
    def test_seed_records_cover_tool_and_reporting_tasks(self) -> None:
        records = generate_seed_records()
        self.assertGreaterEqual(len(records), 20)
        self.assertEqual(
            {record["metadata"]["task"] for record in records},
            {"tool_selection", "safe_refusal", "evidence_report"},
        )
        validate_records(records, require_human_review=False)

    def test_split_keeps_case_together(self) -> None:
        splits = stable_split(generate_seed_records())
        locations = {}
        for split, records in splits.items():
            for record in records:
                case_id = record["metadata"]["case_id"]
                locations.setdefault(case_id, split)
                self.assertEqual(locations[case_id], split)
        self.assertTrue(all(splits.values()))

    def test_training_requires_human_review(self) -> None:
        with self.assertRaises(ValueError):
            validate_records(generate_seed_records())

    def test_jsonl_writer_emits_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_jsonl_splits(temp_dir, generate_seed_records())
            manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 24)
            self.assertTrue(all(Path(paths[name]).exists() for name in ("train", "validation", "test")))


class LocalToolRouterTests(unittest.TestCase):
    def test_router_rejects_malformed_and_future_calls_without_running_tools(self) -> None:
        router = SafeProjectToolRouter()
        with self.assertRaises(ToolRequestError):
            router.execute_json("not json", as_of_date="2025-09-30")
        with self.assertRaises(ToolRequestError):
            router.execute_json(
                '{"tool_name":"check_data_quality","arguments":{"ticker":"ASELS","as_of_date":"2025-10-01"}}',
                as_of_date="2025-09-30",
            )
        with self.assertRaises(ToolRequestError):
            router.execute_json(
                '{"tool_name":"get_stock_history","arguments":{"ticker":"ASELS","allow_download":true}}',
                as_of_date="2025-09-30",
            )
        with self.assertRaises(ToolRequestError):
            router.execute_json(
                '{"tool_name":"check_data_quality","arguments":{"ticker":"TESTX"}}',
                as_of_date="2025-09-30",
            )


class CorpusBuilderTests(unittest.TestCase):
    @staticmethod
    def _frame() -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=700)
        close = np.linspace(100.0, 180.0, len(dates))
        return pd.DataFrame(
            {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Adj Close": close, "Volume": 10_000},
            index=dates,
        )

    def test_common_cutoffs_are_shared_and_have_warmup(self) -> None:
        frame = self._frame()
        cutoffs = select_common_cutoffs({"ASELS": frame, "THYAO": frame.copy(), "XU100": frame.copy()}, 3)
        self.assertEqual(len(cutoffs), 3)
        self.assertEqual(cutoffs, sorted(cutoffs))

    def test_corpus_builder_writes_unreviewed_candidates_and_skip_manifest(self) -> None:
        record = {
            "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "z"}],
            "metadata": {"case_id": "ASELS_TECHNICAL_REVERSAL_20250930", "task": "evidence_report", "human_reviewed": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir, patch("src.llm.build_corpus._run_case", return_value=(record, None)):
            result = build_corpus(
                price_data={"ASELS": self._frame(), "MAVI": self._frame(), "XU100": self._frame()},
                tickers=["ASELS", "MAVI"],
                scenarios=["technical_reversal", "fundamental_reaction"],
                as_of_dates=["2025-09-30"],
                output_dir=temp_dir,
            )
            self.assertEqual(len(result.records), 3)
            self.assertEqual(len(result.skipped), 1)
            self.assertTrue((Path(temp_dir) / "candidates.jsonl").exists())
            self.assertTrue((Path(temp_dir) / "candidates_review.csv").exists())

    @patch("src.llm.build_corpus.get_stock_history")
    @patch("src.llm.build_corpus.get_index_history")
    def test_targeted_download_avoids_full_universe_without_sector_scenario(self, mock_index, mock_stock) -> None:
        _ensure_market_cache(["ASELS", "THYAO"], includes_sector_scenario=False, force_refresh=False)
        mock_index.assert_called_once_with()
        self.assertEqual([call.args[0] for call in mock_stock.call_args_list], ["ASELS", "THYAO"])


class ReviewedRecordPreparationTests(unittest.TestCase):
    def test_converter_keeps_only_complete_approved_rows(self) -> None:
        candidates = [
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "draft"},
                ],
                "metadata": {"case_id": "ASELS_TECHNICAL_REVERSAL_20250930", "task": "evidence_report", "human_reviewed": False},
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidates_path = base / "candidates.jsonl"
            candidates_path.write_text(json.dumps(candidates[0]) + "\n", encoding="utf-8")
            review_path = base / "review.csv"
            review_path.write_text(
                "case_id,human_reviewed,reviewed_summary,reviewer,review_notes\n"
                "ASELS_TECHNICAL_REVERSAL_20250930,true,Status: WATCH. Evidence: E1.,student,checked\n",
                encoding="utf-8",
            )
            output_path = base / "reviewed_records.jsonl"
            approved, excluded = prepare_reviewed_records(candidates_path, review_path, output_path)
            self.assertEqual(len(approved), 1)
            self.assertFalse(excluded)
            self.assertTrue(approved[0]["metadata"]["human_reviewed"])
            self.assertEqual(approved[0]["messages"][-1]["content"], "Status: WATCH. Evidence: E1.")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()


class SectorActionTests(unittest.TestCase):
    def test_single_stock_sector_is_not_a_catch_up_candidate(self) -> None:
        from src.harness.evidence import select_action

        self.assertEqual(select_action("sector_laggard", {"rank_in_sector": 1, "Lag_Score": 0.0}), "WATCH")
        self.assertEqual(select_action("sector_laggard", {"rank_in_sector": 1, "Lag_Score": -0.05}), "POTENTIAL_CATCH_UP_CANDIDATE")


class ScoreTests(unittest.TestCase):
    def test_score_flags_advice_bad_status_and_wrong_tool(self) -> None:
        from src.llm.infer import score

        prompt = "As-of date: 2025-01-02"
        bad = score("evidence_report", prompt, "You should buy. Status: MOON on 2025-03-01 gain 9.9", "Status: WATCH. E1")
        self.assertEqual((bad["trade_advice"], bad["future_date"], bad["bad_status"], bad["no_evidence_id"], bad["number_not_in_reference"]), (1, 1, 1, 1, 1))
        good = score("evidence_report", prompt, "Status: WATCH. E1 holds 9.9", "Status: WATCH. E1 9.9")
        self.assertEqual(sum(good.values()), 0)
        ref = '{"arguments": {"as_of_date": "2025-01-02"}, "tool_name": "rank_sector_laggards"}'
        self.assertEqual(score("tool_selection", prompt, ref, ref)["wrong_tool_call"], 0)
        self.assertEqual(score("tool_selection", prompt, "not json", ref)["wrong_tool_call"], 1)


class FirstJsonTests(unittest.TestCase):
    def test_runaway_tool_output_is_cut_after_first_object(self) -> None:
        from src.llm.infer import first_json

        self.assertEqual(first_json('{"a": 1}\n</think>\n{"fake": null}'), '{"a": 1}')
        self.assertEqual(first_json("no json"), "no json")
