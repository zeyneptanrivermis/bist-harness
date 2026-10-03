"""
BIST 100 Study Universe Definition
CSE-481 Engineering Economics Project
Consists of 30 diversified BIST 100 stocks across multiple sectors.
"""

from typing import Dict, List, Optional
import pandas as pd

# The 30 BIST 100 stocks specified in Section 5 of the project syllabus
BIST_30_UNIVERSE: List[Dict[str, str]] = [
    {"ticker": "ASELS", "yahoo_symbol": "ASELS.IS", "sector": "Technology / Defense"},
    {"ticker": "TUPRS", "yahoo_symbol": "TUPRS.IS", "sector": "Petroleum / Refining"},
    {"ticker": "BIMAS", "yahoo_symbol": "BIMAS.IS", "sector": "Retail Trade"},
    {"ticker": "THYAO", "yahoo_symbol": "THYAO.IS", "sector": "Transportation / Airlines"},
    {"ticker": "AKBNK", "yahoo_symbol": "AKBNK.IS", "sector": "Banking"},
    {"ticker": "KCHOL", "yahoo_symbol": "KCHOL.IS", "sector": "Holding / Investment"},
    {"ticker": "EREGL", "yahoo_symbol": "EREGL.IS", "sector": "Basic Metals / Steel"},
    {"ticker": "TCELL", "yahoo_symbol": "TCELL.IS", "sector": "Telecommunications"},
    {"ticker": "CCOLA", "yahoo_symbol": "CCOLA.IS", "sector": "Food & Beverage"},
    {"ticker": "MGROS", "yahoo_symbol": "MGROS.IS", "sector": "Retail Trade"},
    {"ticker": "SASA", "yahoo_symbol": "SASA.IS", "sector": "Chemicals / Plastics"},
    {"ticker": "TRALT", "yahoo_symbol": "TRALT.IS", "sector": "Mining"},
    {"ticker": "FROTO", "yahoo_symbol": "FROTO.IS", "sector": "Automotive"},
    {"ticker": "ENKAI", "yahoo_symbol": "ENKAI.IS", "sector": "Construction"},
    {"ticker": "PGSUS", "yahoo_symbol": "PGSUS.IS", "sector": "Transportation / Airlines"},
    {"ticker": "MPARK", "yahoo_symbol": "MPARK.IS", "sector": "Healthcare"},
    {"ticker": "EKGYO", "yahoo_symbol": "EKGYO.IS", "sector": "Real Estate Investment Trust"},
    {"ticker": "GUBRF", "yahoo_symbol": "GUBRF.IS", "sector": "Chemicals / Fertilizer"},
    {"ticker": "TURSG", "yahoo_symbol": "TURSG.IS", "sector": "Insurance"},
    {"ticker": "ENJSA", "yahoo_symbol": "ENJSA.IS", "sector": "Electric Utilities"},
    {"ticker": "TTKOM", "yahoo_symbol": "TTKOM.IS", "sector": "Telecommunications"},
    {"ticker": "OYAKC", "yahoo_symbol": "OYAKC.IS", "sector": "Cement"},
    {"ticker": "AKSEN", "yahoo_symbol": "AKSEN.IS", "sector": "Electric Utilities"},
    {"ticker": "ISMEN", "yahoo_symbol": "ISMEN.IS", "sector": "Brokerage / Capital Markets"},
    {"ticker": "ARCLK", "yahoo_symbol": "ARCLK.IS", "sector": "Durables / Machinery"},
    {"ticker": "ECILC", "yahoo_symbol": "ECILC.IS", "sector": "Healthcare / Pharmaceuticals"},
    {"ticker": "MAVI", "yahoo_symbol": "MAVI.IS", "sector": "Retail / Apparel"},
    {"ticker": "PETKM", "yahoo_symbol": "PETKM.IS", "sector": "Petrochemicals"},
    {"ticker": "TOASO", "yahoo_symbol": "TOASO.IS", "sector": "Automotive"},
    {"ticker": "KRDMD", "yahoo_symbol": "KRDMD.IS", "sector": "Basic Metals / Steel"},
]

BENCHMARK_SYMBOL = "XU100.IS"
BENCHMARK_NAME = "BIST 100 Index"


def get_universe_df() -> pd.DataFrame:
    """Returns the universe as a pandas DataFrame."""
    return pd.DataFrame(BIST_30_UNIVERSE)


def get_all_tickers() -> List[str]:
    """Returns a list of all 30 tickers (e.g. ['ASELS', 'TUPRS', ...])."""
    return [item["ticker"] for item in BIST_30_UNIVERSE]


def get_all_yahoo_symbols() -> List[str]:
    """Returns a list of all 30 Yahoo Finance symbols (e.g. ['ASELS.IS', ...])."""
    return [item["yahoo_symbol"] for item in BIST_30_UNIVERSE]


def get_ticker_to_yahoo_map() -> Dict[str, str]:
    """Returns dictionary mapping ticker to Yahoo symbol."""
    return {item["ticker"]: item["yahoo_symbol"] for item in BIST_30_UNIVERSE}


def get_yahoo_to_ticker_map() -> Dict[str, str]:
    """Returns dictionary mapping Yahoo symbol to ticker."""
    return {item["yahoo_symbol"]: item["ticker"] for item in BIST_30_UNIVERSE}


def get_sector_map() -> Dict[str, str]:
    """Returns dictionary mapping ticker to sector."""
    return {item["ticker"]: item["sector"] for item in BIST_30_UNIVERSE}


def get_sectors() -> List[str]:
    """Returns unique sorted list of sectors."""
    return sorted(list(set(item["sector"] for item in BIST_30_UNIVERSE)))


def get_stocks_in_sector(sector: str) -> List[str]:
    """Returns list of tickers belonging to a given sector."""
    return [item["ticker"] for item in BIST_30_UNIVERSE if item["sector"] == sector]


def get_peer_group(ticker: str) -> List[str]:
    """Returns all peer tickers in the same sector (including the given ticker)."""
    sector_map = get_sector_map()
    target_sector = sector_map.get(ticker)
    if not target_sector:
        return [ticker]
    return get_stocks_in_sector(target_sector)


# Suggested Section 32 API bindings
def load_universe() -> pd.DataFrame:
    """Syllabus Section 32 function: loads the 30-stock universe with sector metadata."""
    return get_universe_df()


def build_sector_peer_groups() -> Dict[str, List[str]]:
    """Syllabus Section 32 function: builds dictionary mapping each sector to its constituent tickers."""
    sectors = get_sectors()
    return {sector: get_stocks_in_sector(sector) for sector in sectors}

