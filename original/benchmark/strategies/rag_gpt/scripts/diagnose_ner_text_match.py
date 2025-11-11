#!/usr/bin/env python3
"""
Alternative NER diagnostic that matches by text content rather than just spans.
This helps identify if NER is finding the right concepts but at wrong positions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import argparse

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EVAL_DIR = ROOT.parent.parent / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from pipeline import RAGGPTPipeline


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def load_ground_truth_with_text(truth_path: Path, notes_df: pd.DataFrame) -> pd.DataFrame:
    """Load ground truth and extract actual text from notes."""
    df = pd.read_csv(truth_path)
    
    if "note_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "note_id"})
    
    # Create a mapping of note_id to text
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))
    
    # Extract actual text for each annotation
    gt_texts = []
    for _, row in df.iterrows():
        note_id = row["note_id"]
        start = row["start"]
        end = row["end"]
        
        if note_id in note_texts:
            text = note_texts[note_id][start:end]
            gt_texts.append(text)
        else:
            gt_texts.append("")
    
    df["extracted_text"] = gt_texts
    
    return df


def calculate_text_based_metrics(
    ground_truth: pd.DataFrame,
    ner_predictions: pd.DataFrame
) -> Dict:
    """
    Calculate metrics based on text matching rather than span overlap.
    """
    results = {
        "total_gt_entities": len(ground_truth),
        "total_ner_entities": len(ner_predictions),
        "matched_by_text": 0,
        "matched_by_text_and_span": 0,
        "by_note": {},
        "missing_concepts": {},
        "extra_concepts": {}
    }
    
    gt_by_note = ground_truth.groupby("note_id")
    ner_by_note = ner_predictions.groupby("note_id")
    
    all_note_ids = set(ground_truth["note_id"].unique()) | set(ner_predictions["note_id"].unique())
    
    for note_id in sorted(all_note_ids):
        gt_entities = gt_by_note.get_group(note_id) if note_id in gt_by_note.groups else pd.DataFrame()
        ner_entities = ner_by_note.get_group(note_id) if note_id in ner_by_note.groups else pd.DataFrame()
        
        # Normalize texts for comparison
        gt_texts = set(normalize_text(t) for t in gt_entities["extracted_text"])
        ner_texts = set(normalize_text(t) for t in ner_entities["span_text"])
        
        # Count matches
        matched_texts = gt_texts & ner_texts
        results["matched_by_text"] += len(matched_texts)
        
        # Missing concepts (in GT but not in NER)
        missing = gt_texts - ner_texts
        if missing:
            results["missing_concepts"][note_id] = list(missing)
        
        # Extra concepts (in NER but not in GT)
        extra = ner_texts - gt_texts
        if extra:
            results["extra_concepts"][note_id] = list(extra)
        
        # Per-note stats
        gt_count = len(gt_texts)
        ner_count = len(ner_texts)
        matched_count = len(matched_texts)
        
        note_recall = matched_count / gt_count if gt_count > 0 else 0
        note_precision = matched_count / ner_count if ner_count > 0 else 0
        note_f1 = 2 * note_precision * note_recall / (note_precision + note_recall) if (note_precision + note_recall) > 0 else 0
        
        results["by_note"][note_id] = {
            "gt_count": gt_count,
            "ner_count": ner_count,
            "matched": matched_count,
            "recall": note_recall,
            "precision": note_precision,
            "f1": note_f1
        }
    
    # Overall metrics
    total_gt = results["total_gt_entities"]
    total_ner = results["total_ner_entities"]
    total_matched = results["matched_by_text"]
    
    results["text_recall"] = total_matched / total_gt if total_gt > 0 else 0
    results["text_precision"] = total_matched / total_ner if total_ner > 0 else 0
    results["text_f1"] = (
        2 * results["text_precision"] * results["text_recall"] / 
        (results["text_precision"] + results["text_recall"])
        if (results["text_precision"] + results["text_recall"]) > 0 else 0
    )
    
    return results


def print_text_match_report(results: Dict, verbose: bool = False):
    """Print diagnostic report for text-based matching."""
    print("\n" + "=" * 80)
    print("NER TEXT-BASED MATCHING DIAGNOSTIC")
    print("=" * 80)
    
    print("\n[OVERALL METRICS - Text Content Matching]")
    print(f"  Ground Truth Entities: {results['total_gt_entities']}")
    print(f"  NER Detected Entities: {results['total_ner_entities']}")
    print(f"  Matched by Text:       {results['matched_by_text']}")
    print()
    print(f"  Text-Based Recall:     {results['text_recall']:.4f} ({results['text_recall']*100:.2f}%)")
    print(f"  Text-Based Precision:  {results['text_precision']:.4f} ({results['text_precision']*100:.2f}%)")
    print(f"  Text-Based F1:         {results['text_f1']:.4f}")
    
    print("\n[INTERPRETATION]")
    text_recall = results['text_recall']
    
    if text_recall < 0.6:
        print(f"  ❌ NER is MISSING concepts ({(1-text_recall)*100:.1f}% not found)")
        print(f"     → The NER model is not detecting the medical concepts")
        print(f"     → You need to improve the NER prompts or use a better NER model")
    elif text_recall >= 0.6 and text_recall < 0.9:
        print(f"  ⚠️  NER finds MOST concepts but misses some ({(1-text_recall)*100:.1f}% missing)")
        print(f"     → Consider refining NER prompts to catch edge cases")
    else:
        print(f"  ✅ NER finds almost ALL concepts ({text_recall*100:.1f}%)")
        print(f"     → The problem is likely in span boundaries or RAG coding")
    
    # Per-note breakdown
    if verbose:
        print("\n[PER-NOTE BREAKDOWN - Text Matching]")
        print(f"{'Note ID':<10} {'GT':<6} {'NER':<6} {'Match':<6} {'Recall':<8} {'Precision':<10} {'F1':<8}")
        print("-" * 70)
        for note_id in sorted(results["by_note"].keys()):
            note_data = results["by_note"][note_id]
            print(
                f"{note_id:<10} "
                f"{note_data['gt_count']:<6} "
                f"{note_data['ner_count']:<6} "
                f"{note_data['matched']:<6} "
                f"{note_data['recall']:<8.4f} "
                f"{note_data['precision']:<10.4f} "
                f"{note_data['f1']:<8.4f}"
            )
    
    # Show missing concepts
    print("\n[MISSING CONCEPTS - Not Found by NER]")
    missing_total = sum(len(v) for v in results["missing_concepts"].values())
    if missing_total > 0:
        print(f"  Total unique missing concepts: {missing_total}")
        for note_id in sorted(results["missing_concepts"].keys()):
            concepts = results["missing_concepts"][note_id]
            print(f"\n  Note {note_id} ({len(concepts)} missing):")
            for concept in sorted(concepts)[:10]:
                print(f"    - '{concept}'")
            if len(concepts) > 10:
                print(f"    ... and {len(concepts) - 10} more")
    else:
        print("  ✓ No missing concepts!")
    
    if verbose and results["extra_concepts"]:
        print("\n[EXTRA CONCEPTS - Detected but Not in Ground Truth]")
        extra_total = sum(len(v) for v in results["extra_concepts"].values())
        print(f"  Total unique extra concepts: {extra_total}")
        for note_id in sorted(list(results["extra_concepts"].keys())[:3]):
            concepts = results["extra_concepts"][note_id]
            print(f"\n  Note {note_id} ({len(concepts)} extra):")
            for concept in sorted(concepts)[:5]:
                print(f"    - '{concept}'")
    
    print("\n" + "=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Text-based NER diagnostic")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    input_path = args.input or ROOT.parent.parent / "data" / "mimic-iv_notes_training_set.csv"
    truth_path = args.truth or ROOT.parent.parent / "data" / "train_annotations.csv"
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if not truth_path.exists():
        print(f"Error: Ground truth file not found: {truth_path}")
        return 1
    
    print(f"[SETUP] Loading data...")
    
    # Load notes
    notes_df = pd.read_csv(input_path)
    if "note_id" not in notes_df.columns and "id" in notes_df.columns:
        notes_df = notes_df.rename(columns={"id": "note_id"})
    
    # Load ground truth with extracted text
    gt_df = load_ground_truth_with_text(truth_path, notes_df)
    
    print(f"\n[DATA] Loaded {len(notes_df)} notes with {len(gt_df)} ground truth entities")
    
    # Initialize pipeline
    print(f"\n[PIPELINE] Initializing NER...")
    pipeline = RAGGPTPipeline(verbose=False)
    
    # Extract NER entities
    print(f"\n[NER] Running entity extraction...")
    all_entities = []
    
    for _, row in notes_df.iterrows():
        note_id = row["note_id"]
        text = row["text"]
        
        print(f"[NER] Processing note {note_id}...")
        entities = pipeline.ner.extract_entities(text)
        
        for ent in entities:
            all_entities.append({
                "note_id": note_id,
                "start": ent["start"],
                "end": ent["end"],
                "span_text": ent["span_text"]
            })
    
    ner_df = pd.DataFrame(all_entities)
    print(f"[NER] Extracted {len(ner_df)} entities")
    
    # Calculate text-based metrics
    print(f"\n[EVAL] Calculating text-based metrics...")
    results = calculate_text_based_metrics(gt_df, ner_df)
    
    # Print report
    print_text_match_report(results, verbose=args.verbose)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
