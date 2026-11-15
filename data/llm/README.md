# Local LLM dataset workflow

`notebooks/05_local_llm_finetuning.ipynb` creates a deterministic **starter**
dataset from the version-controlled A-E experiment catalog. It does not call a
cloud LLM and it must not be treated as a sufficient training corpus by itself.

Before training a project adapter:

1. Generate scenario/evidence examples from the deterministic project tools.
2. Write or carefully edit the English target answer as a group.
3. Ensure every answer is tied only to supplied evidence IDs and contains no
   BUY/SELL instruction, price target, or future data.
4. Mark each approved row with `"human_reviewed": true`.
5. Keep related examples for the same `case_id` in one split; the supplied
   splitter already does this for generated seed data.

The dataset, checkpoints, and adapters are excluded from Git. Commit only
small redacted examples if permitted by the course.
