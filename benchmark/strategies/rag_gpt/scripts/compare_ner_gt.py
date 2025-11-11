#!/usr/bin/env python3
"""
Compare NER output with ground truth to identify missing entities.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import RAGGPTPipeline

def main():
    # Load data
    notes_path = ROOT.parent.parent / "data" / "mimic-iv_notes_training_set.csv"
    gt_path = ROOT.parent.parent / "data" / "train_annotations.csv"
    
    notes_df = pd.read_csv(notes_path)
    if "note_id" not in notes_df.columns and "id" in notes_df.columns:
        notes_df = notes_df.rename(columns={"id": "note_id"})
    
    gt_df = pd.read_csv(gt_path)
    if "note_id" not in gt_df.columns and "id" in gt_df.columns:
        gt_df = gt_df.rename(columns={"id": "note_id"})
    
    # Initialize pipeline
    print("[INIT] Initializing NER...")
    pipeline = RAGGPTPipeline(verbose=False)
    
    # Process each note
    for _, note_row in notes_df.iterrows():
        note_id = note_row["note_id"]
        text = note_row["text"]
        
        print(f"\n{'='*80}")
        print(f"NOTE {note_id}")
        print(f"{'='*80}")
        
        # Get ground truth for this note
        gt_note = gt_df[gt_df.note_id == note_id]
        print(f"\n[GT] Ground truth has {len(gt_note)} entities:")
        for _, gt_row in gt_note.iterrows():
            start, end = gt_row["start"], gt_row["end"]
            gt_text = text[start:end]
            print(f"  [{start:4d}-{end:4d}] '{gt_text}'")
        
        # Run NER
        print(f"\n[NER] Running entity extraction...")
        ner_entities = pipeline.ner.extract_entities(text)
        print(f"[NER] Found {len(ner_entities)} entities:")
        for ent in ner_entities:
            print(f"  [{ent['start']:4d}-{ent['end']:4d}] '{ent['span_text']}'")
        
        # Find missing entities
        print(f"\n[COMPARE] Looking for missed entities...")
        gt_spans = set((row["start"], row["end"], text[row["start"]:row["end"]]) for _, row in gt_note.iterrows())
        ner_spans = set((ent["start"], ent["end"], ent["span_text"]) for ent in ner_entities)
        
        # Exact matches
        exact_matches = gt_spans & ner_spans
        print(f"[MATCH] Exact span matches: {len(exact_matches)}")
        
        # Missing from NER
        missing = gt_spans - ner_spans
        if missing:
            print(f"\n[MISSING] {len(missing)} entities NOT found by NER:")
            for start, end, text_span in sorted(missing):
                print(f"  [{start:4d}-{end:4d}] '{text_span}'")
        
        # False positives
        false_pos = ner_spans - gt_spans
        if false_pos:
            print(f"\n[EXTRA] {len(false_pos)} entities found by NER but NOT in ground truth:")
            for start, end, text_span in sorted(false_pos)[:10]:
                print(f"  [{start:4d}-{end:4d}] '{text_span}'")
    
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
