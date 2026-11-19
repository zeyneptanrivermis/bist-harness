# BIST Harness – local fine-tuned LLM (Qwen3 + LoRA)

A question about BIST stocks goes in. The **model** turns it into a JSON tool plan and words the final summary.
**Python code** runs the data-quality gate, the tools and the backtest, and picks the conclusion (WATCH, INVESTIGATE, ANALYSIS_UNSAFE, ...).
A guard replaces any summary that invents numbers, gives advice or uses a date after the analysis date.
Everything runs on this machine. **No cloud AI API is called.**

Analysis date is fixed at **2026-09-10** (`src/config.py`); nothing after it may be used.

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
Unzip the delivered data so these folders exist:
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

Loading the model takes a minute or more. Questions are answered with the temperature at 0, so asking the same question twice gives the same answer.

### What to expect
- Only the 30-stock study universe is accepted. Another ticker returns `ANALYSIS_UNSAFE`.
- A request that needs data after 2026-09-10 returns `ANALYSIS_UNSAFE`.
- Buy/sell or price-target requests get a refusal ("cannot give investment advice"), with the evidence-based status.
- Quarterly fundamentals exist for 5 stocks only (ASELS, BIMAS, THYAO, TUPRS, FROTO). Others return `INVESTIGATE`.
- Every answer starts with `Status: <STATE>` and cites evidence ids (E1, E2, ...).

---

## Reproduce training and evaluation (optional)

**Training** is done on a Colab T4: open `notebooks/bist_llm_training.ipynb`, choose a T4 GPU, and run it top to bottom. It builds the data, trains the QLoRA adapter and saves it to `artifacts/llm/qwen_bist_lora/adapter`. See `notebooks/README.md`.

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

## Optional: Ollama / llama.cpp route
`python -m src.llm.export_local` merges the adapter into the base model, and `python -m src.llm.agent --backend ollama` talks to a local Ollama on localhost. The GGUF conversion steps (in `notebooks/README.md`) were **not tested** in this repo. Use the Transformers route above for the demo.

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
```
