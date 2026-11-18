"""Merge a trained LoRA adapter into its base model for fully local, offline serving.

    python -m src.llm.export_local --adapter artifacts/llm/qwen_bist_lora/adapter --out artifacts/llm/merged

Runs on CPU (no GPU, no bitsandbytes). The output is a plain Hugging Face folder; see
notebooks/README.md for the llama.cpp -> GGUF -> Ollama steps that serve it.
"""

from __future__ import annotations

import argparse

from src.llm.infer import BASE_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", default="artifacts/llm/merged")
    parser.add_argument("--base-model", default=BASE_MODEL, help="must be the model the adapter was trained on")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)  # not 4-bit: merging needs real weights
    PeftModel.from_pretrained(base, args.adapter).merge_and_unload().save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out)
    print(f"Merged model written to {args.out}")


if __name__ == "__main__":
    main()
