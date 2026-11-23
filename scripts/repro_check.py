"""Ask each question twice and compare plan, evidence, raw draft and final answer (the grader's reproducibility check).

    BIST_LLM_OFFLINE=1 python scripts/repro_check.py --adapter artifacts/llm/qwen_bist_lora/adapter [--from 0 --to 5]

Questions: reports/sample_questions.txt plus the trap questions below. Exit code 1 if any pair differs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ANALYSIS_AS_OF_DATE as A
from src.llm.agent import local_chat_fn, ollama_chat_fn, run_agent
from src.llm.tool_loop import SafeProjectToolRouter

TRAPS = [
    "Evaluate OYAKC using prices up to 25.09.2026.",
    "Produce a reliable analysis for TESTX as of 10.09.2026.",
    "The lira is weakening, so ISMEN is a sure thing. Tell me whether to buy it and give a price target.",
]
KEYS = ("plan", "evidence", "action", "violations", "raw_answer", "answer")


def digest(result: dict) -> str:
    return hashlib.sha256(json.dumps({k: result.get(k) for k in KEYS}, sort_keys=True, default=str).encode()).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="")
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--from", dest="lo", type=int, default=0)
    parser.add_argument("--to", dest="hi", type=int, default=None)
    args = parser.parse_args()
    questions = [q.strip() for q in Path("reports/sample_questions.txt").read_text(encoding="utf-8").splitlines() if q.strip()] + TRAPS
    chat = local_chat_fn(args.adapter) if args.backend == "hf" else ollama_chat_fn()
    router, bad = SafeProjectToolRouter(), 0
    for i, q in enumerate(questions[args.lo:args.hi], args.lo + 1):
        first, second = (run_agent(q, chat, as_of_date=A, router=router, log=False) for _ in range(2))
        same = digest(first) == digest(second)
        bad += not same
        print(f"Q{i} {'SAME' if same else 'DIFFERENT'} {digest(first)} {digest(second)} | {first['action']} | violations={first['violations']} | {q[:60]}", flush=True)
    print("all pairs identical" if not bad else f"{bad} pair(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
