# BIST Harness – local fine-tuned LLM (Qwen3 + LoRA)

A question about BIST stocks goes in. The **model** turns it into a JSON tool plan and words the final summary.
**Python code** runs the data-quality gate, the tools and the backtest, and picks the conclusion (WATCH, INVESTIGATE, ANALYSIS_UNSAFE, ...).
A guard checks every summary for numbers not in the tool outputs, advice and dates after the analysis date. A failing summary goes back to the model once with the findings; if the rewrite also fails, a deterministic evidence summary is shown, labelled as harness output (`answer_source`), never as the model's answer.
Everything runs on this machine. **No cloud AI API is called.**

Analysis date defaults to **2026-09-10** (`src/config.py`); nothing after it may be used. Both the CLI and the web page take `--as-of YYYY-MM-DD` (for a live question on newer data, refresh the data first).

---

## What you need (4 things)

| # | Item | Where it comes from |
|---|------|---------------------|
| 1 | This code | `git clone` / the project zip |
| 2 | Price + FX data (`data/processed`, `data/macro`) | **not in Git**; shipped as a zip, or downloaded (step 4) |
| 3 | Base model **Qwen/Qwen3-4B** (or the model named in the report) | Hugging Face, downloaded once (step 5) |
| 4 | LoRA adapter folder | the submitted `adapter.zip` (step 6) |

The adapter only works with the base model it was trained on. The report states which one that is. Use the same one everywhere.

Tested with Python 3.13. Python 3.10+ should work. About 10 GB free disk for the 4B model.

---

## Setup

Run all commands from the repository root. Examples: **bash** (Mac/Linux/Git Bash) and **PowerShell** (Windows).

### 1. Get the code
```bash
git clone https://github.com/zeyneptanrivermis/bist-harness.git
cd bist-harness
git switch zeynep        # skip if the project zip was used
```

### 2. Virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # Mac/Linux/Git Bash
```
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1         # Windows PowerShell
```

### 3. Install packages
```bash
pip install -r requirements.txt
pip install -r requirements-llm.txt
```
- **NVIDIA GPU on Windows/Linux:** if `python -c "import torch; print(torch.cuda.is_available())"` prints `False`, install a CUDA build of torch first, from https://pytorch.org/get-started/locally/, then re-run the two commands above. With CUDA the model loads in 4-bit.
- **No GPU / Mac:** it works, in bf16 on CPU or the Apple GPU. The 4B model is slow on CPU (tens of seconds to minutes per answer) and needs about 10 GB RAM.
- `bitsandbytes` is skipped automatically on a Mac.

### 4. Put the data in place
The price cache is not in Git. Pick **one**:

**A. Use the shipped data (recommended; gives exactly the same numbers as the report).**
Unzip the delivered `bist_data.zip` in the repository root (`unzip bist_data.zip`). Every file in it ends on 2026-09-10, so no price after the analysis date is shipped. These folders must exist:
```
data/processed/    (AKBNK.parquet ... XU100.parquet, 31 tickers)
data/macro/        (USDTRY.csv, EURTRY.csv, and TCMB_POLICY_RATE.csv, TUIK_CPI.csv, FED_FUNDS.csv if provided)
```

**B. Rebuild from the internet.** Yahoo prices may differ slightly from the shipped cache, so numbers can differ.
```bash
python -m src.data_pipeline          # 30 stocks + XU100 into data/processed
python -m src.macro --download       # USD/TRY and EUR/TRY into data/macro
```
Without the TCMB, CPI and Fed files, the macro tool reports those series as unavailable and the model must not state values for them. That is expected. Only macro questions (USD/TRY, rates, inflation) are affected, not training or the other scenarios.

Optional official series go in `data/macro/` with these exact file names:

| File | Content | Source |
|------|---------|--------|
| `TCMB_POLICY_RATE.csv` | TCMB one-week repo (policy) rate | TCMB EVDS |
| `TUIK_CPI.csv` | Turkish CPI (inflation) | TUIK / TCMB EVDS |
| `FED_FUNDS.csv` | Fed funds rate | FRED, series `FEDFUNDS` |

All three use the same columns:
```
date,value,published_date
2024-01-01,5.33,2024-02-01
```
`date` is the period the value belongs to, `value` the number, and `published_date` the day it became public. Add `published_date` whenever possible: CPI for January is published in February, and without the column the loader uses `date`, which leaks future information. Keep only rows from 2024-01-01 whose `published_date` is on or before 2026-09-10. After adding files, re-run `python -m src.llm.check_local` to confirm they are read.

The curated fundamentals file (`data/fundamentals`) is created automatically on first use.

### 5. Download the base model (needs internet, once)
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B')"
```
It is stored in the Hugging Face cache (`~/.cache/huggingface`). To run on a machine without internet, copy that folder to the same place on the target machine.

### 6. Put the adapter in place
Unzip the adapter so that **these files sit directly in this folder** (not in a sub-folder):
```
artifacts/llm/qwen_bist_lora/adapter/adapter_config.json
artifacts/llm/qwen_bist_lora/adapter/adapter_model.safetensors
```
```bash
mkdir -p artifacts/llm/qwen_bist_lora
unzip adapter.zip -d artifacts/llm/qwen_bist_lora/adapter
```
```powershell
Expand-Archive adapter.zip -DestinationPath artifacts\llm\qwen_bist_lora\adapter
```
If the zip contains an extra top-level folder, move its contents up one level.

### 7. Set the model and go offline
The model name must match the one in `adapter_config.json`.
```bash
export BIST_LLM_MODEL=Qwen/Qwen3-4B
export BIST_LLM_OFFLINE=1
```
```powershell
$env:BIST_LLM_MODEL = "Qwen/Qwen3-4B"
$env:BIST_LLM_OFFLINE = "1"
```
`BIST_LLM_OFFLINE=1` makes Hugging Face fail instead of touching the network. Set these again in every new terminal. Optional: `BIST_LLM_DEVICE=cpu` forces CPU.

### 8. Check everything
```bash
python -m src.llm.check_local --adapter artifacts/llm/qwen_bist_lora/adapter
```
You want no `FAIL` lines. `WARN` for missing TCMB, CPI or Fed files is acceptable. The check covers the price cache, torch device, adapter files, adapter and base-model match, and base model on disk.

---

## Run it

Interactive (the model loads once, an empty line quits):
```bash
python -m src.llm.agent --adapter artifacts/llm/qwen_bist_lora/adapter
```
One question:
```bash
python -m src.llm.agent --adapter artifacts/llm/qwen_bist_lora/adapter "Does THYAO show a 'Monday-Tuesday-Wednesday down, then Thursday-Friday up' pattern over 01.01.2024 - 31.12.2025? Does the pattern still hold over 01.01.2026 - 10.09.2026?"
```
Add `--json` to print the full audit trail (tool plan, states, evidence, raw model answer, guard violations). Each answer is also written as a replay log under `data/analysis_logs/`.

The audit trail includes `trace`, the full process record: model, adapter SHA-256 and decoding settings (`temperature` 0, greedy, no seed), every model call with its system and user prompt, every tool call with arguments, date ranges and output, and UTC timestamps. In the web page (`python -m src.llm.web`) it opens from the **Run trace** button under each answer; the window is closed by default. See `reports/06_karar_gunlugu_ve_replay.md`.

Loading the model takes a minute or more. Questions are answered with the temperature at 0, so asking the same question twice gives the same answer.

### What to expect
- Only the 30-stock study universe is accepted. Another ticker returns `ANALYSIS_UNSAFE`.
- A request that needs data after 2026-09-10 returns `ANALYSIS_UNSAFE`.
- Buy/sell or price-target requests get a refusal ("cannot give investment advice"), with the evidence-based status.
- Quarterly fundamentals exist for 5 stocks only (ASELS, BIMAS, THYAO, TUPRS, FROTO). Others return `INVESTIGATE`.
- Every answer starts with `Status: <STATE>` and cites evidence ids (E1, E2, ...).

---

## Reproduce training and evaluation (optional)

**Training** is done in Colab: open `notebooks/bist_llm_training.ipynb`, choose a GPU (a T4 is enough), and run it top to bottom. The notebook also runs on a T4 (fp16). The submitted adapter was trained in bf16 (`training_args.bin`: `bf16=true`, `fp16=false`); the notebook picks bf16 only on compute capability >= 8, so it was trained on a newer Colab GPU (e.g. L4/A100), not on a T4. It builds the data, trains the QLoRA adapter and saves it to `artifacts/llm/qwen_bist_lora/adapter`. See `notebooks/README.md`.

**Dataset only** (needs the data from step 4):
```bash
python -m src.llm.dataset --output-dir data/llm/dataset_v3
```

**Base vs fine-tuned comparison** (12 questions, each asked twice; writes `artifacts/llm/eval_report.json`):
```bash
python -m src.llm.evaluate --adapter artifacts/llm/qwen_bist_lora/adapter
```

**Unit tests** (no model needed):
```bash
python -m unittest discover -s tests
```

---

## Submission package

Run these on the machine that has the **evaluated** adapter (SHA-256 `4d62e052...cf96c4d5c`, see `reports/evidence/eval_report_colab.json`), in the LLM environment (`trl` and `torch` are needed to read `training_args.bin`):

1. Put the training data in place: `data/llm/dataset_v3/records.jsonl`, `manifest.json` and `provenance.json`. The original `records.jsonl` was lost with the Colab runtime; it was rebuilt from commit `cfc6b46` (seed 481) and matches the original manifest except one report status (see `provenance.json`). To rebuild it:
   ```bash
   git worktree add ../bist-v3 cfc6b46 && cp -R data/processed data/raw data/macro data/fundamentals data/news ../bist-v3/data/
   cd ../bist-v3 && BIST_LLM_OFFLINE=1 python -m src.llm.dataset --output-dir data/llm/dataset_v3 --per-family 120 --reports-per-family 90
   ```
2. Record the five scenarios with full traces (model, prompts, tool calls with arguments and outputs, answer source, timestamps, temperature 0, greedy, no seed):
   ```bash
   BIST_LLM_OFFLINE=1 python scripts/run_scenarios.py --adapter artifacts/llm/qwen_bist_lora/adapter
   ```
   -> `reports/execution_traces.json`
3. Rebuild the knowledge-source file: `python scripts/make_rag_file.py` -> `reports/rag_knowledge_base.json`
4. Write the manifest and the fine-tuning configuration:
   ```bash
   python scripts/make_submission.py --adapter artifacts/llm/qwen_bist_lora/adapter
   ```
   -> `submission.json` and `reports/fine_tuning_config.json` (LoRA and training values are read from the adapter's own `adapter_config.json` and `training_args.bin`). The script exits with 1 and prints `MISSING:` / `WARNING:` lines if an attachment is absent or the adapter is not the evaluated one.

## Live question on newer data (demo)

The reports use the frozen data (prices to 2026-09-30, FX to 2026-09-10). A fresh Yahoo download has the same prices but drops a few trading days before 2026-09-10, which would shift report numbers, so never overwrite the frozen data permanently. For a question about October-November 2026:
```bash
tar czf data/backup_frozen.tar.gz data/processed data/raw data/macro      # once
rm -rf data/processed data/raw data/macro && tar xzf data/live_2026-11-26.tar.gz   # or download: python -c "from src.data_pipeline import download_entire_universe as d; d(force_refresh=True)"
python -m src.llm.web --adapter artifacts/llm/qwen_bist_lora/adapter --as-of 2026-11-26
rm -rf data/processed data/raw data/macro && tar xzf data/backup_frozen.tar.gz    # back to the frozen data
```
Policy rate, CPI and Fed funds are manual files; for dates after their last row the macro tool reports the last published value with its date.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FAIL price cache` | Step 4: `data/processed` is missing or incomplete. |
| `FAIL base model on disk` | Step 5 was not done on this machine, or the model name in `BIST_LLM_MODEL` differs from the downloaded one. |
| `FAIL adapter matches base model` | The adapter was trained on another model. Set `BIST_LLM_MODEL` to the value shown as `adapter=...` in the check output. |
| `FAIL adapter files` | The zip was extracted one folder too deep (step 6). |
| `OSError ... local_files_only` / cannot reach huggingface.co | Offline and the model is not in the cache. Copy the cache folder or run step 5 once with internet. |
| `No module named peft/torch/...` | The venv is not active, or step 3 was skipped. |
| Out-of-memory on GPU | Use the 1.7B model and its own adapter, or run on CPU with `BIST_LLM_DEVICE=cpu`. |
| Very slow answers | CPU only. Use a CUDA GPU, or the 1.7B model. |

## Optional: Ollama route
`python -m src.llm.export_local` merges the adapter into the base model, and `python -m src.llm.agent --backend ollama` talks to a local Ollama on localhost. Two routes are described in `notebooks/README.md` step 4: **Route A** (`ollama create -q int8`, tested on macOS, about twice as fast) and **Route B** (GGUF q8_0 through llama.cpp, not tested here). The reported evaluation numbers come from the Transformers route; use it as the reference for the demo.

## Layout
```
src/llm/agent.py        question -> tool plan -> harness -> evidence -> guarded summary (CLI)
src/llm/tool_loop.py    plan validation (tool names, tickers, dates)
src/llm/dataset.py      fine-tuning data generator
src/llm/evaluate.py     base vs adapter evaluation
src/llm/check_local.py  offline readiness check
src/mcp_tools/          the tools the model may call
src/harness/            quality gate, states, replay log
src/scenarios/          sector, weekday, reversal, fundamentals scenarios
roadmap/                project plans and the report draft
reports/01-09            written deliverables (data dictionary, four scenario reports, replay demo, backtests, A-E report, final report)
reports/evidence/       raw tool outputs behind every number; rebuild with scripts/make_report_evidence.py
reports/rag_knowledge_base.json  knowledge sources and their metadata (no RAG is used); rebuild with scripts/make_rag_file.py
scripts/                evidence, strategy-variant and replay-demo scripts
```
