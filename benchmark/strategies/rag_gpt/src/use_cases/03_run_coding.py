#!/usr/bin/env python3
"""
Use-case runner #3: Coding
- Input: JSONL of entities (from 01 or 02)
- Output: JSONL of coded entities (entity_code, anatomy_code, presence_code)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Resolve src for imports when run from repo root
THIS_DIR = Path(__file__).parent.resolve()
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import setup_openai_client, load_prompt, get_model_config, get_assets_dir
from components.coding import SNOMEDCoder
from components.rag import RAGRetriever


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL entities (from 01 or 02)")
    ap.add_argument("--output", required=True, help="Output JSONL coded entities")
    args = ap.parse_args()

    client = setup_openai_client()
    assets_dir = str(get_assets_dir())
    rag = RAGRetriever(assets_dir)

    system_prompt = load_prompt("system")["content"]
    coder = SNOMEDCoder(rag, client, system_prompt=system_prompt)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # collect all entities as list for batch coding
    entities: List[Dict] = list(read_jsonl(in_path))
    coded = coder.code_entities(entities, verbose=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for ent in coded:
            f_out.write(json.dumps(ent, ensure_ascii=False) + "\n")

    print(f"[03_run_coding] Wrote {len(coded)} coded entities to {out_path}")


if __name__ == "__main__":
    main()
