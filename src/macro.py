"""Macro context series with publication dates.

FX comes from Yahoo (``python -m src.macro --download``). Policy rate, CPI and Fed funds
must be filled by the team from the official sources (TCMB/EVDS, TUIK, Federal Reserve) as
``data/macro/<NAME>.csv`` with columns ``date,value[,published_date]``. A missing file is
reported as unavailable; the agent never invents a value.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import MACRO_DATA_DIR

FX_SYMBOLS = {"USDTRY": "USDTRY=X", "EURTRY": "EURTRY=X"}
OFFICIAL_SERIES = ("TCMB_POLICY_RATE", "TUIK_CPI", "FED_FUNDS")
ALL_SERIES = (*FX_SYMBOLS, *OFFICIAL_SERIES)


def load_macro_series(name: str) -> pd.DataFrame | None:
    """Series indexed by publication date (when the value became public), or None if no file."""
    path = MACRO_DATA_DIR / f"{name}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"] + (["published_date"] if "published_date" in pd.read_csv(path, nrows=0).columns else []))
    frame["published_date"] = frame["published_date"] if "published_date" in frame else frame["date"]
    return frame.set_index("published_date").sort_index()[["date", "value"]]


def download_fx(start: str = "2024-01-01") -> None:
    import yfinance as yf

    MACRO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, symbol in FX_SYMBOLS.items():
        close = yf.download(symbol, start=start, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        pd.DataFrame({"date": close.index, "value": close.values}).to_csv(MACRO_DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(close)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FX macro series into data/macro.")
    parser.add_argument("--download", action="store_true", required=True)
    parser.add_argument("--start", default="2024-01-01")
    download_fx(parser.parse_args().start)
