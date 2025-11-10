"""
Configuration management for the RAG+GPT pipeline.
- Loads environment from .env (if present)
- Centralizes prompt loading
- Provides OpenAI client, model config, and assets dir
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict
from openai import OpenAI

# ---------------------------------------------------------------------------------
# Resolve project root and .env
# ---------------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent  # repo root expected at: project/
ENV_PATH = PROJECT_ROOT / ".env"

# Optional: python-dotenv
try:
    from dotenv import load_dotenv  # type: ignore
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        print(f"[CONFIG] Loaded environment from: {ENV_PATH}")
    else:
        print(f"[CONFIG] No .env found at {ENV_PATH}; using system env")
except Exception:
    print("[CONFIG] python-dotenv not available; using system env")

# ---------------------------------------------------------------------------------
# Evaluation offsets policy
# ---------------------------------------------------------------------------------
EVAL_OFFSETS = {
    "base": int(os.getenv("EVAL_OFFSET_BASE", "0")),  # 0-based or 1-based
    # Most clinical benchmarks expect end to be EXCLUSIVE. Flip default to false.
    "end_inclusive": os.getenv("EVAL_END_INCLUSIVE", "false").lower() == "true",
}


# (Optional) tiny log helps verify at runtime
print(f"[CONFIG] EVAL_OFFSETS base={EVAL_OFFSETS['base']} end_inclusive={EVAL_OFFSETS['end_inclusive']}")


# ---------------------------------------------------------------------------------
# Prompt utilities
# ---------------------------------------------------------------------------------
def load_prompt(prompt_name: str, prompts_dir: str | Path | None = None) -> Dict:
    """
    Load a prompt JSON by name from src/prompts.

    Args:
        prompt_name: name without .json
        prompts_dir: override directory

    Returns:
        Dict
    """
    if prompts_dir is None:
        prompts_dir = SRC_DIR / "prompts"
    prompt_path = Path(prompts_dir) / f"{prompt_name}.json"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------------
# OpenAI client and model cfg
# ---------------------------------------------------------------------------------
def setup_openai_client() -> OpenAI:
    """
    Initialize OpenAI client using OPENAI_API_KEY (preferred),
    with a fallback to 'api_keys' file containing a line 'chatGPT=<KEY>'.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_file = PROJECT_ROOT / "api_keys"
        try:
            with open(api_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("chatGPT="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        except Exception as e:
            print(f"[CONFIG] Could not read {api_file}: {e}")
    if not api_key:
        print("[CONFIG] WARNING: OPENAI_API_KEY not set; using placeholder.")
        api_key = "YOUR_API_KEY_HERE"
    return OpenAI(api_key=api_key)

def get_model_config() -> Dict:
    """Config for GPT-4o usage."""
    return {
        "model": os.getenv("NER_LLM_MODEL", "gpt-4o"),
        "temperature": float(os.getenv("NER_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("NER_MAX_TOKENS", "4000")),
        "top_p": float(os.getenv("NER_TOP_P", "0.9")),
    }

# ---------------------------------------------------------------------------------
# Assets (FAISS index, ontology artifacts)
# ---------------------------------------------------------------------------------
def get_assets_dir() -> Path:
    """Return absolute path to assets/ontology/"""
    return PROJECT_ROOT / "assets" / "ontology"
