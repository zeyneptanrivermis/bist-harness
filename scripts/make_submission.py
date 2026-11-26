"""Write the submission manifest (submission.json) and the fine-tuning configuration file.

    python scripts/make_submission.py --adapter artifacts/llm/qwen_bist_lora/adapter

Writes reports/fine_tuning_config.json (read from the adapter's own files, so it always describes the adapter
being submitted) and submission.json (every attachment with path, SHA-256 and size). Exit code 1 lists what is
still missing, and warns when the adapter is not the one the reports were measured with.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.llm.infer import BASE_MODEL, DECODING  # noqa: E402

CONFIG = ROOT / "reports" / "fine_tuning_config.json"
MANIFEST = ROOT / "submission.json"
DATASET = ROOT / "data" / "llm" / "dataset_v3"
TRAINING_KEYS = ("num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
                 "warmup_steps", "weight_decay", "optim", "max_grad_norm", "max_length", "seed", "bf16", "fp16", "gradient_checkpointing",
                 "completion_only_loss", "eval_strategy", "save_strategy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def attachment(path: Path, role: str) -> dict:
    if not path.exists():
        return {"path": rel(path), "role": role, "present": False}
    return {"path": rel(path), "role": role, "present": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def base_revision() -> str | None:
    """Commit of the base model in the local Hugging Face cache (the weights the adapter is loaded on)."""
    hub = Path(os.getenv("HF_HUB_CACHE") or Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    ref = hub / f"models--{BASE_MODEL.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def training_args(adapter: Path) -> dict | None:
    path = adapter / "training_args.bin"
    if not path.exists():
        return None
    try:
        import torch

        values = torch.load(path, weights_only=False).to_dict()  # our own Colab artefact (a pickled SFTConfig)
    except Exception as exc:  # torch/trl missing: say so instead of guessing values
        return {"unreadable": f"{type(exc).__name__}: {exc}"}
    return {key: values.get(key) for key in TRAINING_KEYS}


def fine_tuning_config(adapter: Path) -> dict:
    lora = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    dataset_manifest, provenance = DATASET / "manifest.json", DATASET / "provenance.json"
    training = training_args(adapter)
    return {
        "base_model": lora.get("base_model_name_or_path"),
        "method": "QLoRA: 4-bit NF4 base (double quantization), LoRA adapter trained with TRL SFTTrainer, loss on the answer only",
        "lora": {key: lora.get(key) for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type", "peft_version")},
        "training": training,
        "quantization_during_training": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                                         "bnb_4bit_compute_dtype": "bfloat16" if isinstance(training, dict) and training.get("bf16") else "float16 (T4) or bfloat16 (compute capability >= 8)"},
        "inference_decoding": DECODING,
        "dataset": {"path": rel(DATASET / "records.jsonl"), "builder": "python -m src.llm.dataset --output-dir data/llm/dataset_v3 --per-family 120 --reports-per-family 90",
                    "builder_commit": "cfc6b46",  # the adapter was trained on data built by this commit; later code builds different records
                    "provenance": json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None,
                    "split": "src.llm.training_data.stable_split (train+test used for training, validation for eval)",
                    "manifest": json.loads(dataset_manifest.read_text(encoding="utf-8")) if dataset_manifest.exists() else None},
        "sources": {"lora": rel(adapter / "adapter_config.json"), "training": rel(adapter / "training_args.bin"), "notebook": "notebooks/bist_llm_training.ipynb"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", default="artifacts/llm/qwen_bist_lora/adapter")
    args = parser.parse_args()
    adapter = (ROOT / args.adapter).resolve()
    weights = adapter / "adapter_model.safetensors"
    if not weights.exists():
        print(f"no adapter at {weights}")
        return 1
    CONFIG.write_text(json.dumps(fine_tuning_config(adapter), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    measured = json.loads((ROOT / "reports" / "evidence" / "eval_report_colab.json").read_text(encoding="utf-8"))["adapter_sha256"]
    adapter_sha = sha256(weights)
    code = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "src", "scripts", "notebooks", "tests", "requirements.txt", "requirements-llm.txt", "README.md"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    manifest = {
        "project": "BIST 100 agent harness (CSE-481 Engineering Economics)",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": {"name": BASE_MODEL, "version": base_revision(), "version_note": "Hugging Face commit of the base weights in the local cache",
                       "download_source": f"https://huggingface.co/{BASE_MODEL}", "license": "Apache-2.0"},
        "adapter": {**attachment(weights, "LoRA adapter weights (safetensors)"), "config": attachment(adapter / "adapter_config.json", "LoRA configuration"),
                    "matches_reported_evaluation": adapter_sha == measured, "reported_evaluation_sha256": measured},
        "fine_tuning": [attachment(DATASET / "records.jsonl", "training data (JSONL)"), attachment(DATASET / "manifest.json", "training data manifest"),
                        attachment(DATASET / "provenance.json", "how records.jsonl was rebuilt (the original was lost)"),
                        attachment(CONFIG, "configuration and hyperparameters (JSON)")],
        "rag": attachment(ROOT / "reports" / "rag_knowledge_base.json", "knowledge sources and source metadata (no RAG is used; see the file)"),
        "execution_traces": attachment(ROOT / "reports" / "execution_traces.json", "question-answer records with full traces for the five scenarios"),
        "source_code": [attachment(ROOT / path, "source") for path in code if not path.endswith(".gitkeep")],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [a["path"] for a in (manifest["adapter"], *manifest["fine_tuning"], manifest["rag"], manifest["execution_traces"]) if not a["present"]]
    print(f"wrote {rel(MANIFEST)} and {rel(CONFIG)}")
    if adapter_sha != measured:
        print(f"WARNING: adapter sha256 {adapter_sha[:12]}... is not the evaluated adapter {measured[:12]}... (reports/evidence/eval_report_colab.json)")
    for path in missing:
        print(f"MISSING: {path}")
    return 1 if missing or adapter_sha != measured else 0


if __name__ == "__main__":
    sys.exit(main())
