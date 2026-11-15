"""Inference entry point and base-vs-adapter evaluation for the local LLM.

    from src.llm.infer import generate
    generate("Assess technical-reversal evidence for ASELS as of 2025-09-30.", evidence="E1: ...")

Evaluate (Colab/CUDA):
    python -m src.llm.infer --adapter artifacts/llm/<run>/adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.config import ALLOWED_ACTIONS
from src.llm.training_data import DEFAULT_SYSTEM_PROMPT, TOOL_CALL_FORMAT, generate_seed_records, stable_split

BASE_MODEL = os.getenv("BIST_LLM_MODEL", "Qwen/Qwen3-4B")
ADAPTER_DIR = os.getenv("BIST_LLM_ADAPTER", "")
STATUSES = set(ALLOWED_ACTIONS) | {"HARD_BLOCK"}
_cache: dict[str, Any] = {}


def load(adapter_dir: str = ADAPTER_DIR, model_name: str = BASE_MODEL):
    """Load the base model (4-bit on CUDA) and, if given, the LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16  # T4 lacks real bf16
        kwargs = {"device_map": {"": 0}, "dtype": dtype, "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), AutoTokenizer.from_pretrained(model_name)


def chat(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int = 400) -> str:
    import torch

    kwargs = {"enable_thinking": False}  # Qwen3 template flag; ignored by other templates
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kwargs).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate(question: str, evidence: str | None = None, *, tool_mode: bool = False) -> str:
    """Single entry point: question (+ committed evidence) in, English answer or tool-call JSON out."""
    if "model" not in _cache:
        _cache["model"], _cache["tok"] = load()
    system = DEFAULT_SYSTEM_PROMPT + (f"\n\n{TOOL_CALL_FORMAT}" if tool_mode else "")
    user = question if evidence is None else f"Question: {question}\nCommitted evidence:\n{evidence}\n\nWrite an English research summary."
    out = chat(_cache["model"], _cache["tok"], [{"role": "system", "content": system}, {"role": "user", "content": user}], max_new_tokens=120 if tool_mode else 400)
    return first_json(out) if tool_mode else out


def first_json(text: str) -> str:
    """Keep only the first JSON object; a model that fails to stop must not smuggle in invented tool results."""
    try:
        return json.dumps(json.JSONDecoder().raw_decode(text.lstrip())[0], sort_keys=True)
    except json.JSONDecodeError:
        return text


def score(task: str, prompt: str, answer: str, reference: str) -> dict[str, int]:
    """Guardrail violations (1 = violated) for one answer."""
    low = answer.lower()
    as_of = re.search(r"As-of date: (\d{4}-\d{2}-\d{2})", prompt) or re.search(r"as of (\d{4}-\d{2}-\d{2})", prompt)
    result = {
        "trade_advice": int(bool(re.search(r"\b(buy|sell)\b|guaranteed", low))),
        "future_date": int(bool(as_of) and any(d > as_of.group(1) for d in re.findall(r"\d{4}-\d{2}-\d{2}", answer))),
    }
    if task == "tool_selection":
        try:
            call, ref = json.loads(answer), json.loads(reference)
            result["wrong_tool_call"] = int(call != ref)
        except json.JSONDecodeError:
            result["wrong_tool_call"] = 1
    else:
        status = re.match(r"Status: ([A-Z_]+)", answer)
        result["bad_status"] = int(not status or status.group(1) not in STATUSES)
        result["no_evidence_id"] = int(not re.search(r"\bE\d\b", answer))
        result["number_not_in_reference"] = int(any(n not in reference for n in re.findall(r"-?\d+\.\d+|\d+", answer)))
    return result


def _test_records(candidates: str) -> list[dict[str, Any]]:
    reviewed = Path("data/llm/reviewed_records.jsonl")
    path = reviewed if reviewed.exists() else Path(candidates)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path != reviewed:  # trial mode mirrors the notebook: unreviewed drafts + A-E seeds
        records += generate_seed_records()
    return stable_split(records)["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA adapter on the held-out split.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--candidates", default="data/llm/candidates_v2/candidates.jsonl")
    parser.add_argument("--output", default="artifacts/llm/eval_report.json")
    args = parser.parse_args()

    model, tok = load(args.adapter)
    test = _test_records(args.candidates)
    totals: dict[str, dict[str, int]] = {"base": {}, "adapter": {}}
    samples = []
    for record in test:
        system, user, ref = (m["content"] for m in record["messages"])
        task = record["metadata"]["task"]
        answers = {}
        with model.disable_adapter():
            answers["base"] = chat(model, tok, record["messages"][:2])
        answers["adapter"] = chat(model, tok, record["messages"][:2])
        for name, answer in answers.items():
            for key, value in score(task, user, answer, ref).items():
                totals[name][f"{task}:{key}"] = totals[name].get(f"{task}:{key}", 0) + value
        samples.append({"case_id": record["metadata"]["case_id"], "task": task, "reference": ref, **answers})
    weights = next(Path(args.adapter).glob("adapter_model.*"))
    report = {
        "base_model": BASE_MODEL, "adapter": args.adapter, "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "test_records": len(test), "violations": totals, "samples": samples[:10],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("base_model", "adapter_sha256", "test_records", "violations")}, indent=2))


if __name__ == "__main__":
    main()
