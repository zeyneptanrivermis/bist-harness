"""Deterministic keyword rules used by the news layer.

Topics and stance are labelled by plain keyword lists (English + Turkish). They are context tags,
never facts: the original text, source and timestamp are always kept next to them.
"""

from __future__ import annotations

import re
from typing import List

from src.universe import get_all_tickers

TOPIC_KEYWORDS = {
    "earnings": ["earnings", "net profit", "net income", "ebitda", "revenue", "quarter", "bilanço", "net kâr", "net kar", "ciro", "çeyrek", "ceyrek"],
    "regulation": ["regulation", "regulator", "spk", "bddk", "rekabet", "ban", "fine", "düzenleme", "duzenleme", "ceza", "yasak"],
    "fx": ["usd/try", "eur/try", "exchange rate", "lira", "dollar", "devaluation", "kur", "dolar", "euro"],
    "rates": ["interest rate", "policy rate", "rate hike", "rate cut", "fed ", "fomc", "tcmb", "faiz", "politika faizi"],
    "inflation": ["inflation", "cpi", "enflasyon", "tüfe", "tufe"],
    "commodity": ["oil", "brent", "gold", "steel", "commodity", "petrol", "altın", "altin", "çelik", "celik"],
    "contract_award": ["contract", "order", "tender", "awarded", "sözleşme", "sozlesme", "sipariş", "siparis", "ihale"],
    "geopolitical": ["war", "sanction", "conflict", "geopolit", "savaş", "savas", "yaptırım", "yaptirim", "gerilim"],
}
POSITIVE = ["beat", "growth", "rise", "record", "upgrade", "surge", "improve", "profit increase", "artış", "artis", "rekor", "yükseldi", "yukseldi"]
NEGATIVE = ["miss", "loss", "fall", "drop", "downgrade", "plunge", "decline", "fine", "probe", "investigation", "default", "zarar", "düştü", "dustu", "soruşturma", "sorusturma", "ceza"]


def _has(text: str, word: str) -> bool:
    # short ASCII words must match as whole words so "ban" does not fire on "bank"
    return re.search(rf"\b{re.escape(word.strip())}\b", text) is not None if len(word.strip()) <= 4 and word.isascii() else word in text


def tag_topics(text: str) -> List[str]:
    low = text.casefold()
    return [topic for topic, words in TOPIC_KEYWORDS.items() if any(_has(low, w) for w in words)]


def tag_stance(text: str) -> str:
    low = text.casefold()
    pos, neg = sum(_has(low, w) for w in POSITIVE), sum(_has(low, w) for w in NEGATIVE)
    if pos and neg:
        return "mixed"
    return "positive" if pos else "negative" if neg else "neutral"


def tag_tickers(text: str) -> List[str]:
    upper = text.upper()
    return [t for t in get_all_tickers() if re.search(rf"\b{t}\b", upper)]

