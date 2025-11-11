#!/usr/bin/env python3
"""
Analyze which (note_id, concept_id) pairs we're missing compared to ground truth.
This helps identify patterns in missed entities.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    # Load data
    gt_path = ROOT.parent.parent / "data" / "train_annotations.csv"
    
    # Run pipeline to get predictions
    sys.path.insert(0, str(ROOT / "src"))
    from pipeline import RAGGPTPipeline
    
    notes_df = pd.read_csv(ROOT.parent.parent / "data" / "mimic-iv_notes_training_set.csv")
    if "note_id" not in notes_df.columns and "id" in notes_df.columns:
        notes_df = notes_df.rename(columns={"id": "note_id"})
    
    print("[RUNNING] Executing pipeline...")
    pipeline = RAGGPTPipeline(verbose=False)
    preds = pipeline.predict(notes_df)
    
    # Load ground truth
    gt = pd.read_csv(gt_path)
    if "note_id" not in gt.columns and "id" in gt.columns:
        gt = gt.rename(columns={"id": "note_id"})
    
    # Convert to sets of (note_id, concept_id) pairs
    gt_pairs = set(zip(gt["note_id"], gt["concept_id"].astype(str)))
    pred_pairs = set(zip(preds["note_id"], preds["concept_id"].astype(str)))
    
    # Find missing pairs
    missing = gt_pairs - pred_pairs
    extra = pred_pairs - gt_pairs
    matched = gt_pairs & pred_pairs
    
    print("\n" + "="*80)
    print("PAIR-LEVEL ANALYSIS")
    print("="*80)
    print(f"\nGround Truth Pairs: {len(gt_pairs)}")
    print(f"Predicted Pairs:    {len(pred_pairs)}")
    print(f"Matched Pairs:      {len(matched)}")
    print(f"Missing Pairs:      {len(missing)} [MISSING]")
    print(f"Extra Pairs:        {len(extra)} [EXTRA]")
    
    # Group missing by concept
    print("\n" + "="*80)
    print("MISSING PAIRS BY CONCEPT")
    print("="*80)
    missing_by_concept = {}
    for note_id, concept_id in sorted(missing):
        if concept_id not in missing_by_concept:
            missing_by_concept[concept_id] = []
        missing_by_concept[concept_id].append(note_id)
    
    # Show most frequent missing concepts
    print(f"\nTop missing concepts:")
    for concept_id in sorted(missing_by_concept.keys(), key=lambda x: len(missing_by_concept[x]), reverse=True)[:10]:
        notes = missing_by_concept[concept_id]
        # Get concept name from ontology
        concept_examples = gt[gt["concept_id"].astype(str) == concept_id]["span_text"].unique()[:3]
        print(f"  {concept_id}: {len(notes)} pairs missing")
        print(f"    Examples: {', '.join(concept_examples)}")
        print(f"    In notes: {sorted(notes)}")
    
    # Show extra (false positives)
    print("\n" + "="*80)
    print("FALSE POSITIVE PAIRS (Extra)")
    print("="*80)
    extra_by_concept = {}
    for note_id, concept_id in sorted(extra):
        if concept_id not in extra_by_concept:
            extra_by_concept[concept_id] = []
        extra_by_concept[concept_id].append(note_id)
    
    print(f"\nTop extra concepts:")
    for concept_id in sorted(extra_by_concept.keys(), key=lambda x: len(extra_by_concept[x]), reverse=True)[:10]:
        notes = extra_by_concept[concept_id]
        # Get examples from predictions
        concept_examples = preds[preds["concept_id"].astype(str) == concept_id]["span_text"].unique()[:3]
        print(f"  {concept_id}: {len(notes)} extra pairs")
        print(f"    Examples: {', '.join(concept_examples)}")
        print(f"    In notes: {sorted(notes)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
