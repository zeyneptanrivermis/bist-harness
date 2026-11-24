"""Speech-intelligence layer (proposal sections 16 and 22: transcribe_finance_video_segment, get_speech_features).

Only instructor-approved public videos are processed, and only as short timestamped segments: source URL,
speaker/channel, publication time and the transcript time range are mandatory. Topics and stance reuse the
news keyword rules. Numeric claims are only *extracted* and flagged ``verified=False``: commentary is context,
never ground truth, and a claim may enter the committed evidence only after a deterministic tool confirms it.
A missing or empty file means "no speech evidence"; nothing is invented.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from src.config import DATA_DIR
from src.context.news import effective_date
from src.context.text_rules import tag_stance, tag_tickers, tag_topics

SPEECH_PATH = DATA_DIR / "speech" / "speech_segments.csv"
APPROVED_PATH = DATA_DIR / "speech" / "approved_sources.txt"  # one video URL per line, fixed before experiments
SPEECH_COLUMNS = ["segment_id", "source_id", "video_url", "speaker", "channel", "published_at", "effective_date",
                  "start_s", "end_s", "text", "topics", "stance", "tickers", "numeric_claims", "uncertainty"]
HEDGES = ["maybe", "might", "could", "i think", "uncertain", "not sure", "belki", "sanırım", "emin değil", "olabilir"]


def approved_sources(path: Optional[Path] = None) -> set[str]:
    path = Path(path or APPROVED_PATH)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path.exists() else set()


def _default_transcriber(audio_path: str, start_s: float, end_s: float) -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # optional dependency, not in requirements.txt
        raise RuntimeError("Speech-to-text needs `pip install faster-whisper`, or pass your own `transcriber`.") from exc
    segments, _ = WhisperModel("small").transcribe(audio_path, clip_timestamps=[start_s, end_s])
    return " ".join(s.text.strip() for s in segments)


def transcribe_finance_video_segment(audio_path: str, video_url: str, speaker: str, channel: str, published_at: str,
                                     start_s: float, end_s: float, approved: Optional[Iterable[str]] = None,
                                     transcriber: Optional[Callable[[str, float, float], str]] = None) -> dict:
    """Transcribe ``[start_s, end_s]`` of an approved video and return one provenance-complete segment."""
    allowed = set(approved) if approved is not None else approved_sources()
    if video_url not in allowed:
        raise PermissionError(f"{video_url} is not on the instructor-approved source list.")
    stamp = pd.to_datetime(published_at, errors="coerce")
    if not (speaker and channel) or pd.isna(stamp) or not 0 <= start_s < end_s:
        raise ValueError("speaker, channel, a valid published_at and 0 <= start_s < end_s are required.")
    text = (transcriber or _default_transcriber)(audio_path, start_s, end_s).strip()
    if not text:
        raise ValueError("Empty transcript: nothing to store.")
    return {"video_url": video_url, "speaker": speaker, "channel": channel, "published_at": stamp, "start_s": float(start_s),
            "end_s": float(end_s), "text": text}


def extract_speech_features(segments: Iterable[dict]) -> pd.DataFrame:
    """Tag segments with topics, stance, tickers, numeric claims (unverified) and an uncertainty flag."""
    rows = []
    for seg in segments:
        text, low = seg["text"], seg["text"].casefold()
        source_id = hashlib.sha1(seg["video_url"].encode()).hexdigest()[:10]
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\d", s)]
        rows.append({"segment_id": f"{source_id}-{int(seg['start_s'])}-{int(seg['end_s'])}", "source_id": source_id,
                     "video_url": seg["video_url"], "speaker": seg["speaker"], "channel": seg["channel"],
                     "published_at": seg["published_at"], "effective_date": effective_date(pd.Timestamp(seg["published_at"])),
                     "start_s": seg["start_s"], "end_s": seg["end_s"], "text": text, "topics": ";".join(tag_topics(text)),
                     "stance": tag_stance(text), "tickers": ";".join(tag_tickers(text)),
                     "numeric_claims": " | ".join(claims), "uncertainty": any(h in low for h in HEDGES)})
    return pd.DataFrame(rows, columns=SPEECH_COLUMNS)


def save_speech_features(frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
    path = Path(path or SPEECH_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([_read(path), frame[SPEECH_COLUMNS]], ignore_index=True).drop_duplicates("segment_id", keep="first")
    merged.sort_values("published_at").to_csv(path, index=False)
    return path


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SPEECH_COLUMNS)
    return pd.read_csv(path, parse_dates=["published_at", "effective_date"], keep_default_na=False)


def get_speech_features(source_id: Optional[str] = None, as_of: Optional[str] = None, path: Optional[Path] = None) -> pd.DataFrame:
    """Segments known by ``as_of`` (by effective date), optionally for one video ``source_id`` or its URL."""
    frame = _read(Path(path or SPEECH_PATH))
    if as_of:
        frame = frame[frame["effective_date"] <= pd.Timestamp(as_of)]
    if source_id:
        frame = frame[(frame["source_id"] == source_id) | (frame["video_url"] == source_id)]
    return frame.sort_values(["published_at", "start_s"]).reset_index(drop=True)
