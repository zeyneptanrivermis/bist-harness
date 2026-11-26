"""Run the five course scenarios with the model and save every question-answer record with its full trace (JSON).

    BIST_LLM_OFFLINE=1 python scripts/run_scenarios.py --adapter artifacts/llm/qwen_bist_lora/adapter
    -> reports/execution_traces.json

Each record is the complete run_agent result: model name, version and decoding settings (temperature 0, greedy,
no seed), the system and user prompts of every model call, every tool call with its arguments, date ranges and
output, the guard result, who wrote the final answer (answer_source) and UTC timestamps. The questions are the
same five as the web page's examples (src/llm/web.py SAMPLES = reports/sample_questions.txt).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ANALYSIS_AS_OF_DATE  # noqa: E402
from src.mcp_tools.serialization import json_safe  # noqa: E402

OUT = ROOT / "reports" / "execution_traces.json"


def main() -> int:
    from src.llm.agent import local_chat_fn, ollama_chat_fn, run_agent
    from src.llm.tool_loop import SafeProjectToolRouter
    from src.llm.web import SAMPLES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="", help="hf backend: LoRA adapter directory")
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--ollama-model", default="bist-qwen")
    parser.add_argument("--as-of", default=ANALYSIS_AS_OF_DATE)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    chat = local_chat_fn(args.adapter) if args.backend == "hf" else ollama_chat_fn(args.ollama_model)
    router, records = SafeProjectToolRouter(), []
    for number, (scenario, question) in enumerate(SAMPLES, 1):
        result = run_agent(question, chat, as_of_date=args.as_of, router=router)
        records.append({"scenario": scenario, **result})
        print(f"{number}/{len(SAMPLES)} {scenario}: {result['action']} (answer by {result['answer_source']})", flush=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "analysis_as_of_date": args.as_of,
               "model": getattr(chat, "meta", None), "records": records}
    Path(args.output).write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
