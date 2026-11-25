from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.context.news import extract_news_events, get_news_events, save_news_events
from src.context.speech import (extract_speech_features, get_speech_features, save_speech_features,
                                transcribe_finance_video_segment)
from src.harness.harness import HarnessSession, HarnessState
from src.mcp_tools.models import AnalysisRequest
from src.strategies.variants import VARIANTS, compare_strategy_variants, generate_strategy_variants
from tests.test_task3_harness import make_ohlcv

ITEMS = [
    {"headline": "THYAO net profit record in Q3, revenue growth", "published_at": "2026-03-10 09:30", "source": "wire-a", "url": "https://x/1"},
    {"headline": "THYAO fined by regulator after investigation", "published_at": "2026-03-12 18:40", "source": "wire-a"},
    {"headline": "USD/TRY rises after policy rate decision", "published_at": "2026-03-13 10:00", "source": "wire-b"},
    {"headline": "no source here", "published_at": "2026-03-13 10:00", "source": ""},
    {"headline": "bad time", "published_at": "not a date", "source": "wire-b"},
]


class NewsTests(unittest.TestCase):
    def test_extract_tags_and_rejects_items_without_provenance(self) -> None:
        frame = extract_news_events(ITEMS)
        self.assertEqual(len(frame.attrs["rejected"]), 2)
        thy = frame[frame["ticker"] == "THYAO"]
        self.assertEqual(set(thy["stance"]), {"positive", "negative"})
        self.assertIn("earnings", thy.iloc[0]["topics"])
        self.assertEqual(frame[frame["ticker"] == ""].iloc[0]["topics"], "fx;rates")
        self.assertTrue(frame["event_id"].is_unique)

    def test_after_close_news_is_known_only_from_the_next_day_and_cutoff_holds(self) -> None:
        frame = extract_news_events(ITEMS)
        late = frame[frame["headline"].str.contains("fined")].iloc[0]
        self.assertEqual(late["effective_date"], pd.Timestamp("2026-03-13"))  # published 18:40 on 12 March
        with tempfile.TemporaryDirectory() as tmp:
            path = save_news_events(frame, Path(tmp) / "news.csv")
            save_news_events(frame, path)  # de-duplicated
            self.assertEqual(len(get_news_events(path=path)), len(frame))
            self.assertEqual(len(get_news_events(ticker="THYAO", as_of="2026-03-12", path=path)), 1)
            self.assertEqual(len(get_news_events(ticker="THYAO", as_of="2026-03-13", path=path)), 2)
            self.assertEqual(len(get_news_events(topic="fx", path=path)), 1)

    def test_missing_file_means_no_evidence_not_an_error(self) -> None:
        self.assertTrue(get_news_events(path=Path("does-not-exist.csv")).empty)


class SpeechTests(unittest.TestCase):
    URL = "https://youtu.be/approved"

    def segment(self, url: str = URL) -> dict:
        return transcribe_finance_video_segment(
            "a.wav", url, "Speaker", "Channel", "2026-03-12 19:00", 0, 30, approved=[self.URL],
            transcriber=lambda *_: "THYAO margins might improve. Inflation is 45 percent. Fed rate cut soon.")

    def test_unapproved_source_is_refused(self) -> None:
        with self.assertRaises(PermissionError):
            self.segment("https://youtu.be/other")

    def test_features_flag_claims_unverified_and_respect_cutoff(self) -> None:
        frame = extract_speech_features([self.segment()])
        row = frame.iloc[0]
        self.assertEqual(row["tickers"], "THYAO")
        self.assertTrue(row["uncertainty"])
        self.assertIn("45 percent", row["numeric_claims"])
        self.assertEqual(row["effective_date"], pd.Timestamp("2026-03-13"))  # after the 18:00 close
        with tempfile.TemporaryDirectory() as tmp:
            path = save_speech_features(frame, Path(tmp) / "s.csv")
            save_speech_features(frame, path)  # de-duplicated
            self.assertEqual(len(get_speech_features(path=path)), 1)
            self.assertTrue(get_speech_features(as_of="2026-03-12", path=path).empty)
            self.assertEqual(len(get_speech_features(source_id=self.URL, as_of="2026-03-13", path=path)), 1)

    def test_missing_file_means_no_evidence(self) -> None:
        self.assertTrue(get_speech_features(path=Path("does-not-exist.csv")).empty)


class StrategyVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.price = make_ohlcv(300)
        self.market = make_ohlcv(300)
        self.as_of = str(self.price.index[-1].date())

    def test_variants_are_nested_and_flag_missing_layers(self) -> None:
        signals, flags = generate_strategy_variants("THYAO", self.price)
        self.assertEqual(set(flags), {"no_peer_group", "no_fundamentals", "no_macro", "no_news_records"})
        for weaker, stronger in zip(VARIANTS, VARIANTS[1:]):
            self.assertTrue((signals[stronger] <= signals[weaker]).all())

    def test_each_layer_can_only_remove_signal_days(self) -> None:
        peers = {"THYAO": self.price, "PGSUS": self.price * np.linspace(1, 3, len(self.price))[:, None]}  # peer rallies, THYAO lags
        fx = pd.Series(np.linspace(30, 60, len(self.price)), index=self.price.index)  # steep lira fall
        negative = extract_news_events([{"headline": "THYAO probe and loss", "published_at": str(self.price.index[150].date()) + " 10:00", "source": "s"}])
        base, _ = generate_strategy_variants("THYAO", self.price)
        full, flags = generate_strategy_variants("THYAO", self.price, peers, None, fx, negative)
        self.assertEqual(set(flags), {"no_fundamentals"})
        self.assertLess(full["B"].sum(), base["B"].sum())
        self.assertLess(full["D"].sum(), base["D"].sum())
        self.assertEqual(full["E"].iloc[150:155].sum(), 0)  # blocked right after the negative item

    def test_compare_uses_same_backtest_and_reports_out_of_sample(self) -> None:
        result = compare_strategy_variants("THYAO", self.as_of, oos_start=str(self.price.index[200].date()), price=self.price, market=self.market)
        self.assertEqual(set(result["variants"]), set(VARIANTS))
        for row in result["variants"].values():
            self.assertIn("Buy_and_Hold_Return", row)
            self.assertIn("out_of_sample", row)
        self.assertIn("no_news_records", result["flags"])


class HarnessContextTests(unittest.TestCase):
    def test_context_step_is_optional_ordered_and_logged(self) -> None:
        data = make_ohlcv()
        request = AnalysisRequest(ticker="ASELS", scenario="technical_reversal", as_of_date=str(data.index[220].date()), require_human_review=False)
        session = HarnessSession(request)
        session.select_universe()
        session.load_data(data, data.copy())
        session.check_quality()
        session.calculate()
        with self.assertRaises(RuntimeError):
            session.load_context()  # too early
        session.run_scenario()
        news = extract_news_events([{"headline": "ASELS contract awarded", "published_at": "2025-06-02 10:00", "source": "s"},
                                    {"headline": "ASELS future item", "published_at": "2030-01-01 10:00", "source": "s"}])
        session.load_context(news=news)
        self.assertEqual(session.state, HarnessState.LOAD_CONTEXT)
        self.assertEqual(session.context["news_count"], 1)  # the 2030 item is after the analysis date
        session.run_backtest()
        session.build_evidence()
        with patch("src.harness.harness.log_decision", return_value=Path("x.json")) as log:
            session.log_and_close()
        self.assertIn("LOAD_CONTEXT", log.call_args[0][0]["state_history"])
        self.assertEqual(log.call_args[0][0]["context"]["news_count"], 1)


if __name__ == "__main__":
    unittest.main()
