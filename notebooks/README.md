# Task 5: local, group-specific LLM (Qwen3 + QLoRA)

Core principle of the course: the LLM reasons and explains, Python tools calculate, the harness controls, backtests verify, the model runs locally.

## What the model does and what code does

| Step | Who | Where |
|---|---|---|
| Question -> JSON tool plan (1-4 calls, ISO dates) | model | `src/llm/agent.py` |
| Plan validation (tool names, tickers, dates never after the analysis date) | code | `src/llm/tool_loop.py` |
| Quality gate -> tools -> backtest -> evidence, in fixed order | code | `execute_plan` in `agent.py` |
| Final status (WATCH, INVESTIGATE, REJECT_SIGNAL, ...) | code, from the tools' evidence | `mcp_tools/research.py` |
| English summary of the evidence | model | |
| Guard: status, numbers, evidence ids, advice words, future dates | code; a bad summary is replaced by the deterministic draft | `agent.py`, `summaries.py` |

Tools (names from the course table): `get_stock_history`, `calculate_indicators`, `detect_reversal_events`, `rank_sector_relative_strength`, `test_weekday_pattern`, `get_fundamental_metrics`, `calculate_fundamental_change`, `get_macro_context`, `run_backtest`, `check_data_quality`. Each takes `start_date`/`end_date`; the analysis date (`config.ANALYSIS_AS_OF_DATE`, 2026-09-10) is set by the harness, not the model. Forward windows that are not complete by the end date are excluded and reported.

## Train in Colab (T4)

1. Push the `zeynep` branch (the notebook clones it).
2. Open `bist_llm_training.ipynb`, choose a **T4 GPU**, run top to bottom. Cell 4 downloads prices and FX and builds the dataset (`python -m src.llm.dataset`, about 700 records: tool plans, evidence reports, refusals). Targets are deterministic, built from the tool outputs, so no target number can be wrong. Six tickers (`dataset.HOLDOUT_TICKERS`) are never used in training, so the evaluation has unseen stocks.
3. Models: `Qwen/Qwen3-4B` (demo model) or `Qwen/Qwen3-1.7B` (faster fallback). Keep the same model for training, evaluation and the local run: an adapter only fits the model it was trained on.
4. The adapter is copied to Google Drive right after training. Download `results.zip` at the end; Colab runtimes reset.
5. The last cell runs `python -m src.llm.evaluate`: 12 questions in the style of the course sample test, each asked twice, for the base model and the adapter. Rates: invalid tool call, wrong plan (tool/tickers/dates), numeric hallucination, unsupported conclusion, look-ahead, missing evidence, rule violation, not reproducible, guard replaced the answer.

Macro data: FX comes from Yahoo (`python -m src.macro --download`). TCMB policy rate, TUIK CPI and Fed funds must be supplied by the team as `data/macro/<NAME>.csv` (columns `date,value,published_date`); without them the tool reports them as unavailable and the model must not state values.

## Run fully locally (no cloud API, no internet)

1. `pip install -r requirements-llm.txt` (on a Mac `bitsandbytes` is skipped automatically).
2. Download the base model once with internet, then use `export BIST_LLM_OFFLINE=1`.
3. Put the adapter in `artifacts/llm/qwen_bist_lora/adapter` and run:
   ```bash
   python -m src.llm.agent --adapter artifacts/llm/qwen_bist_lora/adapter
   ```
   With no question it opens an interactive session (the model loads once). `--as-of` defaults to 2026-09-10, `--json` prints the audit trail. Without CUDA the model loads in bf16 on the Apple GPU (MPS) or CPU; `BIST_LLM_DEVICE=cpu` forces CPU. 4-bit loading and training are CUDA-only.
4. Optional, faster Ollama/llama.cpp route:
   ```bash
   python -m src.llm.export_local --adapter artifacts/llm/qwen_bist_lora/adapter --out artifacts/llm/merged
   python llama.cpp/convert_hf_to_gguf.py artifacts/llm/merged --outfile artifacts/llm/bist-qwen.gguf --outtype q8_0
   printf 'FROM ./artifacts/llm/bist-qwen.gguf\nPARAMETER temperature 0\n' > Modelfile && ollama create bist-qwen -f Modelfile
   python -m src.llm.agent --backend ollama
   ```
   Ollama only listens on localhost. The GGUF steps need llama.cpp checked out and were not run in this repo.

## Demo-day checklist (the day before, on the demo machine)

1. Price cache in `data/processed` (not in Git): copy it or run `python -m src.data_pipeline`; FX: `python -m src.macro --download`.
2. Base model on disk and the adapter in place.
3. `python -m src.llm.check_local --adapter artifacts/llm/qwen_bist_lora/adapter` prints no FAIL (add `--ollama-model bist-qwen` for the Ollama route).
4. Ask the five sample questions with Wi-Fi off, twice each: the answers must be identical.
5. Keep the 1.7B adapter as a fallback if the 4B is too slow.

## Notes

- The older review-based pipeline (`build_corpus`, `prepare_reviewed_records`, `05_local_llm_finetuning.ipynb`) is still in the repo but is not used by the current training notebook.
- Data, checkpoints and adapters are ignored by Git. Record the model id, adapter hash and seed in the report; the evaluation report has them.
