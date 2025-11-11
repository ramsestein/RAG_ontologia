#!/usr/bin/env python3
"""
Use-case runner #1: NER extraction
- Input: CSV with columns [note_id, text]
- Output: JSONL, one entity per line with note_id and offsets from the input text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Resolve src for imports when run from repo root
THIS_DIR = Path(__file__).parent.resolve()
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import setup_openai_client, get_model_config, load_prompt
from components.ner import NERExtractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with columns: note_id,text")
    ap.add_argument("--output", required=True, help="Output JSONL path (entities per line)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if not {"note_id", "text"}.issubset(df.columns):
        raise ValueError("Input CSV must have columns: note_id,text")

    client = setup_openai_client()
    ner_prompt = load_prompt("ner")
    system_prompt = load_prompt("system")["content"]
    model_cfg = get_model_config()

    ner = NERExtractor(client, ner_prompt, model_cfg, system_prompt=system_prompt)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            note_id = row["note_id"]
            text = row["text"]
            entities = ner.extract_entities(str(text))
            for ent in entities:
                ent_out = {
                    "note_id": int(note_id),
                    "span_text": ent["span_text"],
                    "full_span": ent.get("full_span", ent["span_text"]),
                    "anatomical_location": ent.get("anatomical_location", "No especificado"),
                    "presence": ent.get("presence", "presente"),
                    "value": ent.get("value"),
                    "start": ent.get("start"),
                    "end": ent.get("end"),
                }
                f_out.write(json.dumps(ent_out, ensure_ascii=False) + "\n")
                n_total += 1

    print(f"[01_run_ner] Wrote {n_total} entities to {out_path}")


if __name__ == "__main__":
    main()
