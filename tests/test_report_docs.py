"""The written deliverables must not drift from the code and data they describe."""

import re
import unittest
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.context.news import NEWS_COLUMNS
from src.universe import get_universe_df

ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = (ROOT / "reports" / "01_veri_sozlugu.md").read_text(encoding="utf-8")


def undocumented(columns) -> list[str]:
    return [c for c in columns if f"`{c}`" not in DICTIONARY]


class DataDictionaryTest(unittest.TestCase):
    def test_universe_and_news_columns_are_documented(self) -> None:
        self.assertEqual(undocumented(list(get_universe_df().columns) + NEWS_COLUMNS), [])

    def test_macro_and_fundamental_headers_are_documented(self) -> None:
        files = sorted((DATA_DIR / "macro").glob("*.csv")) + sorted((DATA_DIR / "fundamentals").glob("*.csv"))
        if not files:
            self.skipTest("no local data files")
        for path in files:
            with self.subTest(file=path.name):
                self.assertEqual(undocumented(pd.read_csv(path, nrows=0).columns), [])
                self.assertIn(path.name, DICTIONARY)

    def test_price_and_indicator_columns_are_documented(self) -> None:
        from src.data_pipeline import get_stock_history
        from src.indicators import calculate_all_indicators

        try:
            frame = get_stock_history("ASELS")
        except Exception:
            self.skipTest("no local price cache")
        self.assertEqual(undocumented(calculate_all_indicators(frame).columns), [])


class SubmissionCleanupTest(unittest.TestCase):
    draft = (ROOT / "roadmap" / "TASK_5_RAPOR_TASLAGI.md").read_text(encoding="utf-8")

    def test_task5_draft_has_no_placeholders_and_one_dataset_name(self) -> None:
        self.assertNotIn("[ ]", self.draft)
        self.assertNotIn("[yapıldı", self.draft)
        self.assertNotIn("v4 üretimi", self.draft)

    def test_task5_limitations_are_lettered_once_each(self) -> None:
        letters = re.findall(r"\(([a-z])\) ", self.draft.split("6. **Bilinen")[1])
        self.assertEqual(letters, sorted(set(letters)))

    def test_ollama_route_is_described_in_one_place(self) -> None:
        notebooks = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
        self.assertIn("Route A", notebooks)
        self.assertIn("Route B", notebooks)
        self.assertNotIn("not run in this repo", notebooks)

    def test_test_results_show_the_current_adapter(self) -> None:
        text = (ROOT / "reports" / "test_sonuclari.md").read_text(encoding="utf-8")
        self.assertIn("r=32", text)
        self.assertNotIn("0,583 | 0,000", text)  # the old adapter's guard rate


if __name__ == "__main__":
    unittest.main()
