"""Write the "RAG file" of the submission: the knowledge the agent uses and its source metadata (JSON).

    python scripts/make_rag_file.py  ->  reports/rag_knowledge_base.json

The system has no retrieval-augmented generation: the model never searches documents, every number
comes from a deterministic tool over the files below. This file lists those files (spec section 24:
temporal, episodic and procedural memory) with source, publication rule, coverage and SHA-256, plus
the exact system prompt and tool definitions the model receives. Nothing here is invented: counts,
dates and hashes are read from the files at run time.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ANALYSIS_AS_OF_DATE, DATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from src.fundamentals import CURATED_FUNDAMENTAL_RECORDS  # noqa: E402
from src.harness.permissions import STATE_TOOLS  # noqa: E402
from src.llm.training_data import DEFAULT_SYSTEM_PROMPT, STATUSES, TOOL_CALL_FORMAT, project_tool_definitions, tool_system_prompt  # noqa: E402
from src.universe import get_universe_df  # noqa: E402

OUT = ROOT / "reports" / "rag_knowledge_base.json"
MACRO = {
    "USDTRY": ("Yahoo Finance (USDTRY=X)", "https://finance.yahoo.com/quote/USDTRY=X/", "Day-end value, usable the same day."),
    "EURTRY": ("Yahoo Finance (EURTRY=X)", "https://finance.yahoo.com/quote/EURTRY=X/", "Day-end value, usable the same day."),
    "TCMB_POLICY_RATE": ("TCMB EVDS, one-week repo rate (%)", "https://evds2.tcmb.gov.tr/", "Usable from published_date (decision day)."),
    "TUIK_CPI": ("TUIK / TCMB EVDS, CPI", "https://data.tuik.gov.tr/", "Usable from published_date (early next month)."),
    "FED_FUNDS": ("FRED FEDFUNDS (%)", "https://fred.stlouisfed.org/series/FEDFUNDS", "Usable from published_date (start of next month)."),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def price_entry() -> dict:
    files = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))
    entry = {"id": "daily_prices", "memory_type": "temporal", "path": "data/processed/<TICKER>.parquet",
             "description": "Daily OHLCV of the 30-stock universe and XU100, with indicators computed by src/indicators.py.",
             "source": {"name": "Yahoo Finance via yfinance", "url": "https://finance.yahoo.com/", "retrieval": "python -m src.data_pipeline"},
             "publication_rule": "A bar is known at that day's close; signals act from t+1.",
             "in_repository": False, "note": "Gitignored; rebuilt from the source with the command above. The cache may run past the "
                                              "analysis date; every tool cuts the data at analysis_as_of_date."}
    if not files:
        return {**entry, "available": False}
    dates = [pd.read_parquet(f, columns=[]).index for f in files]
    return {**entry, "available": True, "files": len(files), "tickers": [f.stem for f in files],
            "first_date": str(min(d.min() for d in dates).date()), "last_date": str(max(d.max() for d in dates).date()),
            "sha256": {f.stem: sha256(f) for f in files}}


def macro_entry(name: str) -> dict:
    path = DATA_DIR / "macro" / f"{name}.csv"
    source, url, rule = MACRO[name]
    entry = {"id": f"macro_{name.lower()}", "memory_type": "temporal", "path": rel(path), "description": f"Macro series {name}.",
             "source": {"name": source, "url": url, "retrieval": "python -m src.macro --download (FX); official series entered by the team",
                        "query_and_download_date": None}, "publication_rule": rule, "in_repository": True}
    if not path.exists():
        return {**entry, "available": False}
    frame = pd.read_csv(path)
    entry |= {"available": True, "rows": len(frame), "first_date": frame["date"].min(), "last_date": frame["date"].max(), "sha256": sha256(path)}
    if name == "TUIK_CPI":
        entry["note"] = "Values rise from about 62 to 135: an index level, not an annual inflation %; base year not stated in the file."
    return entry


def build() -> dict:
    fundamentals = ROOT / "src" / "fundamentals.py"
    news = DATA_DIR / "news" / "news_events.csv"
    speech = DATA_DIR / "speech" / "speech_segments.csv"
    news_rows = len(pd.read_csv(news)) if news.exists() else 0
    universe = ROOT / "src" / "universe.py"
    return {
        "retrieval_augmented_generation": False,
        "design_note": ("The system does not use RAG. The model never searches or reads documents to find facts; it only (1) plans which "
                        "deterministic tools to call and (2) words a summary of the committed tool evidence (E1, E2, ...). Every number comes "
                        "from a tool over the files below, and a guard rejects numbers that are not in the evidence. This is deliberate: it "
                        "limits unsupported numbers. The knowledge the system uses is listed here with its source metadata (spec section 24)."),
        "analysis_as_of_date": ANALYSIS_AS_OF_DATE,
        "documents": [
            {"id": "study_universe", "memory_type": "temporal", "path": rel(universe), "description": "Fixed 30-stock BIST 100 study universe and sector map.",
             "source": {"name": "Team-curated list", "url": None}, "rows": len(get_universe_df()), "sha256": sha256(universe), "in_repository": True},
            price_entry(),
            *(macro_entry(name) for name in MACRO),
            {"id": "quarterly_fundamentals", "memory_type": "temporal", "path": rel(fundamentals) + " (CURATED_FUNDAMENTAL_RECORDS)",
             "description": "Quarterly revenue, EBITDA, net income, net debt, ROE, EBITDA margin with disclosure dates.",
             "source": {"name": "Entered by hand by the team", "url": None, "kap_reference": None},
             "publication_rule": "A quarter is usable only after kap_disclosure_date.",
             "rows": len(CURATED_FUNDAMENTAL_RECORDS), "tickers": sorted({r["ticker"] for r in CURATED_FUNDAMENTAL_RECORDS}),
             "sha256": sha256(fundamentals), "in_repository": True,
             "note": "Sample data: no KAP notification id or source document, so the numbers cannot be verified."},
            {"id": "news_events", "memory_type": "temporal", "path": rel(news), "description": "Timestamped news metadata (headline, source, URL; no article body).",
             "source": {"name": "Team-collected, legally accessible sources", "url": None},
             "publication_rule": "Published at or after 18:00 (BIST close) -> usable the next day.",
             "rows": news_rows, "in_repository": True,
             "note": "Empty: the tool (get_news_events) exists and returns a warning instead of inventing context."},
            {"id": "speech_segments", "memory_type": "temporal", "path": rel(speech), "description": "Short timestamped transcript segments of instructor-approved videos.",
             "source": {"name": "Instructor-approved public videos", "url": None}, "rows": len(pd.read_csv(speech)) if speech.exists() else 0,
             "available": speech.exists(), "in_repository": False,
             "note": "No file: the tool (get_speech_features) exists and returns a warning instead of inventing commentary."},
            {"id": "decision_logs", "memory_type": "episodic", "path": "data/analysis_logs/analysis_<id>.json",
             "description": "Past analyses: question, plan, evidence, answer, raw model draft, guard violations, trace; SHA-256 sealed (src/harness/replay.py).",
             "source": {"name": "Produced by the agent", "url": None}, "in_repository": False,
             "note": "Written by the agent; not read back into the model's prompt."},
        ],
        "procedural_memory": {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "tool_call_format": TOOL_CALL_FORMAT,
            "tool_definitions": project_tool_definitions(),
            "tool_planning_system_prompt": tool_system_prompt(),
            "allowed_statuses": sorted(STATUSES),
            "state_tool_permissions": {state: sorted(tools) for state, tools in STATE_TOOLS.items()},
            "code": {"prompts": "src/llm/training_data.py", "permissions": "src/harness/permissions.py", "router": "src/llm/tool_loop.py", "guard": "src/llm/agent.py"},
        },
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"wrote {rel(OUT)}")
