"""Backtest strategy variants A-E on all 30 tickers, cut at the analysis date, and save one row per ticker and variant.

    python scripts/run_strategy_variants.py     (writes reports/evidence/strategy_variants.csv, or the path given as the first argument)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ANALYSIS_AS_OF_DATE as A
from src.strategies.variants import compare_strategy_variants
from src.universe import get_all_tickers

OOS = "2026-01-01"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/evidence/strategy_variants.csv")


def main() -> None:
    rows = []
    for ticker in get_all_tickers():
        result = compare_strategy_variants(ticker, A, oos_start=OOS)
        for name, m in result["variants"].items():
            oos = m.pop("out_of_sample", {})
            rows.append({"ticker": ticker, "variant": name, "flags": ";".join(result["flags"]), **m,
                         **{f"oos_{k}": v for k, v in oos.items() if k in ("Total_Return", "Sharpe_Ratio", "Max_Drawdown", "Total_Trades", "Buy_and_Hold_Return", "Alpha_vs_Benchmark", "Alpha_vs_Sector")}})
        print(ticker, result["flags"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows), "rows")


if __name__ == "__main__":
    main()
