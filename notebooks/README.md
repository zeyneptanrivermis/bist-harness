# Notebooks (Task 5: local, group-specific LLM)

Two notebooks, two different jobs. They do not overlap.

| Notebook | Job | Where it runs | GPU |
|---|---|---|---|
| `05_local_llm_finetuning.ipynb` | **Data preparation and guardrail demo.** Generates the A-E seed records, splits them into train/validation/test, validates reviewed data, and demonstrates the guarded tool router and guardrail metrics. It never trains or loads a model. | Your own machine | No |
| `bist_llm_training.ipynb` | **Training, evaluation and use.** Trains the QLoRA adapter, compares base model vs adapter on the held-out split, tries `generate()`, and exports the results. Self-contained: it does not call the other notebook. | Google Colab | T4 |

## Workflow

1. **Generate candidates** (your machine, once; downloads public OHLCV data):
   ```bash
   python -m src.llm.build_corpus --download --dates-per-ticker 3 --output-dir data/llm/candidates_v2 --reviewers Name1 Name2 Name3
   ```
   Creates `candidates.jsonl` (unreviewed drafts with real numbers from the harness), `candidates_review.csv` (for reviewers) and `manifest.json`.
2. **Review** `candidates_review.csv` as a group: set `human_reviewed=true`, fill `reviewer`, edit `reviewed_summary`. Follow the checklist in `roadmap/TASK_5_LOCAL_LLM.md`.
3. **Convert approved rows** into the training file:
   ```bash
   python -m src.llm.prepare_reviewed_records --candidates data/llm/candidates_v2/candidates.jsonl --review-csv data/llm/candidates_v2/candidates_review.csv
   ```
   Creates `data/llm/reviewed_records.jsonl`. Rows that are unreviewed, incomplete, or contain BUY/SELL wording are refused.
4. *(Optional)* Open `05_local_llm_finetuning.ipynb` locally to inspect the seed split and check that `reviewed_records.jsonl` validates.
5. **Train and evaluate** in Colab with `bist_llm_training.ipynb` (below).

## Using `bist_llm_training.ipynb` in Colab

1. Open it in Colab and choose Runtime -> Change runtime type -> **T4 GPU**.
2. Run the cells top to bottom:
   - **clone and install:** gets the `defne` branch and `requirements-llm.txt`. The branch must be pushed, because Colab clones from GitHub.
   - **mode and model cell:** two switches.
     - `BIST_LLM_TRIAL`: `"1"` trains on unreviewed drafts plus seed records. This is a pipeline test only and is **not valid for submission**. `"0"` trains only on `reviewed_records.jsonl`.
     - `BIST_LLM_MODEL`: `Qwen/Qwen3-1.7B` (about 20 min) or `Qwen/Qwen3-4B` (slower).
   - **upload cell:** trial needs `candidates.jsonl`; a real run needs `reviewed_records.jsonl`. Data is not in Git.
   - **training cells:** the adapter is saved to `artifacts/llm/qwen_bist_lora_TRIAL_unreviewed/adapter` (trial) or `artifacts/llm/qwen_bist_lora/adapter` (real).
   - **generate cell:** `generate(question, evidence=...)` returns an English summary; `tool_mode=True` returns a JSON tool request.
   - **evaluation cell:** compares base model and adapter on the held-out split and writes `artifacts/llm/eval_report.json` (model, adapter hash, violation counts, sample answers).
   - **download cell:** zips the adapter and report. Colab runtimes reset, so download before closing.
3. Keep the mode and model settings the same for the evaluation cells as for training. The adapter only fits the model it was trained on.

## Notes

- A trial run proves the pipeline works. For the submitted model use `BIST_LLM_TRIAL="0"` with reviewed data.
- Data, checkpoints and adapters are ignored by Git. Record the model ID, adapter hash and seed in the report (the evaluation report has them).
