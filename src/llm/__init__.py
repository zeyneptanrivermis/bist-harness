"""Local-LLM training and guarded tool-use helpers for the project."""

from src.llm.training_data import DEFAULT_SYSTEM_PROMPT, generate_seed_records

__all__ = ["DEFAULT_SYSTEM_PROMPT", "generate_seed_records"]
