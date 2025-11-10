#!/usr/bin/env python3
"""
Use-case runner #2: RAG candidates retrieval
- Input: JSONL of entities (output of 01_run_ner.py)
- Output: JSONL of entities with RAG candidates for entity and anatomy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Resolve src for imports when run from repo root
THIS_DIR = Path(__file__).parent.resolve()
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import get_assets_dir
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
    ap.add_argument("--input", required=True, help="JSONL: one entity per line (from 01)")
    ap.add_argument("--output", required=True, help="Output JSONL with candidates")
    ap.add_argument("--k", type=int, default=30, help="Top-K candidates to retrieve")
    args = ap.parse_args()

    assets_dir = str(get_assets_dir())
    rag = RAGRetriever(assets_dir)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for obj in read_jsonl(in_path):
            entity = obj.get("span_text") or obj.get("full_span") or ""
            location = obj.get("anatomical_location", "No especificado")

            ent_cands = rag.retrieve(entity, k=args.k)
            anat_cands = rag.retrieve(location, k=min(args.k, 15)) if location and location != "No especificado" else []

            obj["entity_candidates"] = [
                {"concept_id": c, "narrative": nrr, "score": float(s)} for c, nrr, s in ent_cands
            ]
            obj["anatomy_candidates"] = [
                {"concept_id": c, "narrative": nrr, "score": float(s)} for c, nrr, s in anat_cands
            ]

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"[02_run_rag] Wrote {n} lines with candidates to {out_path}")


if __name__ == "__main__":
    main()
