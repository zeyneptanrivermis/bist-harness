"""Model loading and chat helpers for the local LLM (Hugging Face Transformers + LoRA adapter).

Full agent loop and CLI: src/llm/agent.py. Evaluation: python -m src.llm.evaluate --adapter <dir>.
Offline: set BIST_LLM_OFFLINE=1 (model and adapter must already be on disk).
"""

from __future__ import annotations

import json
import os
from typing import Any

BASE_MODEL = os.getenv("BIST_LLM_MODEL", "Qwen/Qwen3-4B")
ADAPTER_DIR = os.getenv("BIST_LLM_ADAPTER", "")
# Greedy decoding: no sampling, so temperature and seed do not apply. Recorded in every trace (src/llm/agent.py).
DECODING = {"temperature": 0, "do_sample": False, "seed": None, "note": "greedy decoding (argmax), no sampling: temperature 0, seed not used"}


def load(adapter_dir: str = ADAPTER_DIR, model_name: str = BASE_MODEL):
    """Load the base model (4-bit on CUDA) and, if given, the LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    offline = os.getenv("BIST_LLM_OFFLINE") == "1"  # never touch the network; fail if files are missing
    kwargs: dict[str, Any] = {"local_files_only": offline}
    if not torch.cuda.is_available():
        # 4-bit bitsandbytes needs CUDA; bf16 weights instead (4B is about 8 GB). Apple GPU (MPS) when present; BIST_LLM_DEVICE=cpu overrides.
        device = os.getenv("BIST_LLM_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        kwargs |= {"dtype": torch.bfloat16, "device_map": {"": device}}
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16  # T4 lacks real bf16
        kwargs |= {"device_map": {"": 0}, "dtype": dtype, "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), AutoTokenizer.from_pretrained(model_name, local_files_only=offline)


def chat(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int = 900) -> str:
    import torch

    kwargs = {"enable_thinking": False}  # Qwen3 template flag; ignored by other templates
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kwargs).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=DECODING["do_sample"])
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def first_json(text: str) -> str:
    """Keep only the first JSON object; a model that fails to stop must not smuggle in invented tool results."""
    try:
        return json.dumps(json.JSONDecoder().raw_decode(text.lstrip())[0], sort_keys=True)
    except json.JSONDecodeError:
        return text


def main() -> None:
    from src.llm.evaluate import main as evaluate_main

    evaluate_main()


if __name__ == "__main__":
    main()
