"""Timestamped news metadata layer (proposal sections 16 and 22: get_news_events, extract_news_events).

Only legally obtained metadata is stored: headline, source, URL and publication time. No article body
is kept and nothing is scraped here. Records come from a file the team fills (``data/news/news_events.csv``)
or from ``extract_news_events`` on items the team collected. A missing or empty file means "no news
evidence", which is a soft warning, never a reason to invent events.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from src.config import DATA_DIR
from src.context.text_rules import tag_stance, tag_tickers, tag_topics
from src.universe import get_sector_map

NEWS_PATH = DATA_DIR / "news" / "news_events.csv"
NEWS_COLUMNS = ["event_id", "published_at", "effective_date", "ticker", "sector", "topics", "stance", "headline", "source", "url"]
MARKET_CLOSE_HOUR = 18  # BIST closes at 18:00 (Istanbul time); later items are known only from the next day


def effective_date(published_at: pd.Timestamp) -> pd.Timestamp:
    """First calendar day on which a decision at the close may use this item."""
    day = published_at.normalize()
    return day + pd.Timedelta(days=1) if published_at.hour >= MARKET_CLOSE_HOUR else day


def extract_news_events(items: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Tag raw items ``{headline, published_at, source[, url]}`` by ticker, sector, topic and stance.

    Items without a headline, source or parseable timestamp are rejected (listed in ``.attrs["rejected"]``)
    because evidence without provenance cannot be replayed. One row per tagged ticker; an item that names
    no universe ticker becomes a market-wide row (empty ticker).
    """
    sectors, rows, rejected = get_sector_map(), [], []
    for item in items:
        headline, source = str(item.get("headline") or "").strip(), str(item.get("source") or "").strip()
        stamp = pd.to_datetime(item.get("published_at"), errors="coerce")
        if not headline or not source or pd.isna(stamp):
            rejected.append({"item": item, "reason": "headline, source and a valid published_at are required"})
            continue
        stamp = stamp.tz_localize(None) if stamp.tzinfo else stamp
        key = hashlib.sha1(f"{source}|{stamp.isoformat()}|{headline}".encode()).hexdigest()[:12]
        for ticker in tag_tickers(headline) or [""]:
            rows.append({"event_id": f"{key}-{ticker or 'MKT'}", "published_at": stamp, "effective_date": effective_date(stamp), "ticker": ticker,
                         "sector": sectors.get(ticker, ""), "topics": ";".join(tag_topics(headline)), "stance": tag_stance(headline),
                         "headline": headline, "source": source, "url": str(item.get("url") or "")})
    frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    frame.attrs["rejected"] = rejected
    return frame


def save_news_events(frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Append to the news file, de-duplicated by ``event_id``."""
    path = Path(path or NEWS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([_read(path), frame[NEWS_COLUMNS]], ignore_index=True).drop_duplicates("event_id", keep="first")
    merged.sort_values("published_at").to_csv(path, index=False)
    return path


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    frame = pd.read_csv(path, parse_dates=["published_at", "effective_date"], keep_default_na=False)
    frame["ticker"] = frame["ticker"].astype(str)
    return frame


def get_news_events(ticker: Optional[str] = None, topic: Optional[str] = None, start_date: Optional[str] = None,
                    end_date: Optional[str] = None, as_of: Optional[str] = None, path: Optional[Path] = None) -> pd.DataFrame:
    """Events known by ``as_of`` (by effective date, never by later timestamps), optionally filtered."""
    frame = _read(Path(path or NEWS_PATH))
    if as_of:
        frame = frame[frame["effective_date"] <= pd.Timestamp(as_of)]
    if ticker:
        frame = frame[frame["ticker"] == ticker]
    if topic:
        frame = frame[frame["topics"].str.split(";").apply(lambda t: topic in t)]
    if start_date:
        frame = frame[frame["effective_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["effective_date"] <= pd.Timestamp(end_date)]
    return frame.sort_values("published_at").reset_index(drop=True)
