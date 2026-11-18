"""Demo-day preflight: is everything on this machine, with no network needed?

    python -m src.llm.check_local --adapter artifacts/llm/qwen_bist_lora/adapter [--ollama-model bist-qwen]

Prints one line per check and exits non-zero if a required one fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from src.config import MACRO_DATA_DIR, PROCESSED_DATA_DIR
from src.macro import ALL_SERIES
from src.universe import get_all_tickers


def _model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def main() -> None:
    from src.llm.infer import BASE_MODEL

    parser = argparse.ArgumentParser(description="Offline readiness check for the local BIST LLM demo.")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--ollama-model", default="")
    args = parser.parse_args()

    missing = [t for t in [*get_all_tickers(), "XU100"] if not (PROCESSED_DATA_DIR / f"{t}.parquet").exists()]
    checks: list[tuple[str, bool, str, bool]] = [  # (name, ok, detail, required)
        ("price cache (30 tickers + XU100)", not missing, f"missing: {missing[:5]}" if missing else "complete", True),
    ]
    absent = [n for n in ALL_SERIES if not (MACRO_DATA_DIR / f"{n}.csv").exists()]
    checks.append(("macro series (USDTRY, EURTRY, TCMB_POLICY_RATE, TUIK_CPI, FED_FUNDS)", not absent,
                   f"missing: {absent} (python -m src.macro --download for FX; rates/CPI need official CSVs)" if absent else "complete", False))
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        checks.append(("torch device", True, device, False))
    except ImportError:
        checks.append(("torch installed", False, "pip install -r requirements-llm.txt", True))
    if args.adapter:
        adapter = Path(args.adapter)
        checks.append(("adapter files", (adapter / "adapter_config.json").exists() and any(adapter.glob("adapter_model.*")), str(adapter), True))
        try:
            trained_on = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
            checks.append(("adapter matches base model", trained_on == args.model, f"adapter={trained_on} model={args.model}", True))
        except (OSError, KeyError, ValueError):
            pass
    checks.append((f"base model on disk ({args.model})", _model_cached(args.model), "run once with internet to download", not args.ollama_model))
    if args.ollama_model:
        try:
            tags = json.loads(urlopen("http://localhost:11434/api/tags", timeout=3).read())["models"]
            ok = any(m["name"].split(":")[0] == args.ollama_model for m in tags)
            checks.append((f"ollama model {args.ollama_model}", ok, "ollama create ... if missing", True))
        except Exception as exc:
            checks.append(("ollama daemon (localhost:11434)", False, str(exc), True))

    failed = False
    for name, ok, detail, required in checks:
        print(f"{'OK  ' if ok else 'FAIL' if required else 'WARN'}  {name}: {detail}")
        failed |= required and not ok
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
