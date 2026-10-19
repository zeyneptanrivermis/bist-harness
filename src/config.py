"""
Configuration and Parameters
CSE-481 Engineering Economics Project
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MACRO_DATA_DIR = DATA_DIR / "macro"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ANALYSIS_LOG_DIR = DATA_DIR / "analysis_logs"

# Ensure all subdirectories exist
for d in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MACRO_DATA_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
    ANALYSIS_LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Date Configurations
# Warmup start date ensures rolling indicators (like 200 SMA, Ichimoku) are valid by Jan 2025
WARMUP_START_DATE = "2024-01-01"
# TMS 29 inflation-adjusted reporting period recommended by instructor
ANALYSIS_START_DATE = "2025-01-01"
ANALYSIS_END_DATE = "2026-11-01"

# Trading & Backtest Assumptions
DEFAULT_INITIAL_CAPITAL = 100_000.0  # TL
DEFAULT_COMMISSION_RATE = 0.001       # 0.10% (Binde 1)
DEFAULT_SLIPPAGE_RATE = 0.0005        # 0.05%
SIGNAL_SHIFT_DAYS = 1                 # Strict t+1 execution to prevent look-ahead bias

# Finite Educational Action Space (from Specification Section 17)
ALLOWED_ACTIONS = [
    "HOLD",
    "WATCH",
    "INVESTIGATE",
    "POTENTIAL_CATCH_UP_CANDIDATE",
    "TECHNICAL_REVERSAL_CANDIDATE",
    "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
    "REJECT_SIGNAL",
    "ANALYSIS_UNSAFE",
]

# Risk & Data Quality Gate Statuses
GATE_STATUSES = [
    "PASS",
    "PASS_WITH_WARNINGS",
    "INVESTIGATE",
    "ANALYSIS_UNSAFE",
    "HARD_BLOCK",
]

# Data Quality Thresholds
MAX_ALLOWED_MISSING_RATIO = 0.05  # 5% max missing points
MIN_REQUIRED_HISTORY_DAYS = 60    # Minimum trading days required for indicator calculation
