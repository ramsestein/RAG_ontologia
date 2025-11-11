#!/usr/bin/env python3
"""
Diagnostic tool to measure NER recall independently from RAG coding performance.

This script helps identify whether low recall is due to:
1. NER failure (not detecting entities)
2. RAG failure (detecting entities but coding them wrong)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import argparse

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Add evaluation to path
EVAL_DIR = ROOT.parent.parent / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from pipeline import RAGGPTPipeline


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def check_span_overlap(
    gt_start: int, 
    gt_end: int, 
    pred_start: int, 
    pred_end: int,
    overlap_threshold: float = 0.5
) -> bool:
    """
    Check if two spans overlap significantly.
    
    Args:
        gt_start: Ground truth start position
        gt_end: Ground truth end position
        pred_start: Predicted start position
        pred_end: Predicted end position
        overlap_threshold: Minimum overlap ratio (IoU-style)
    
    Returns:
        True if spans overlap sufficiently
    """
    # Calculate overlap
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    
    if overlap_start >= overlap_end:
        return False  # No overlap
    
    overlap_length = overlap_end - overlap_start
    gt_length = gt_end - gt_start
    pred_length = pred_end - pred_start
    
    # Calculate IoU (Intersection over Union)
    union_length = gt_length + pred_length - overlap_length
    iou = overlap_length / union_length if union_length > 0 else 0
    
    return iou >= overlap_threshold


def load_ground_truth(truth_path: Path) -> pd.DataFrame:
    """Load and normalize ground truth annotations."""
    df = pd.read_csv(truth_path)
    
    # Standardize column names
    if "note_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "note_id"})
    
    # Ensure we have required columns
    required = ["note_id", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ground truth missing columns: {missing}")
    
    # Add span_text if it exists
    if "span_text" in df.columns:
        df["gt_span_text"] = df["span_text"]
    
    return df


def load_input_notes(input_path: Path) -> pd.DataFrame:
    """Load input notes."""
    df = pd.read_csv(input_path)
    
    if "note_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "note_id"})
    
    if not {"note_id", "text"}.issubset(df.columns):
        raise ValueError(f"Input must have 'note_id' and 'text' columns")
    
    return df


def extract_ner_only(pipeline: RAGGPTPipeline, notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run only the NER step of the pipeline (no coding).
    
    Returns:
        DataFrame with columns: note_id, start, end, span_text
    """
    all_entities = []
    
    for _, row in notes_df.iterrows():
        note_id = row["note_id"]
        text = row["text"]
        
        print(f"[NER] Processing note {note_id}...")
        
        # Run only NER
        entities = pipeline.ner.extract_entities(text)
        
        print(f"[NER] Found {len(entities)} entities in note {note_id}")
        
        for ent in entities:
            all_entities.append({
                "note_id": note_id,
                "start": ent["start"],
                "end": ent["end"],
                "span_text": ent["span_text"]
            })
    
    return pd.DataFrame(all_entities)


def calculate_ner_metrics(
    ground_truth: pd.DataFrame,
    ner_predictions: pd.DataFrame,
    overlap_threshold: float = 0.5
) -> Dict:
    """
    Calculate NER-only metrics (precision, recall, F1).
    
    This measures whether NER detected the entities, regardless of coding.
    """
    results = {
        "total_gt_entities": 0,
        "total_ner_entities": 0,
        "matched_entities": 0,
        "by_note": {},
        "unmatched_gt": [],
        "unmatched_ner": []
    }
    
    # Group by note_id
    gt_by_note = ground_truth.groupby("note_id")
    ner_by_note = ner_predictions.groupby("note_id")
    
    all_note_ids = set(ground_truth["note_id"].unique()) | set(ner_predictions["note_id"].unique())
    
    for note_id in sorted(all_note_ids):
        gt_entities = gt_by_note.get_group(note_id) if note_id in gt_by_note.groups else pd.DataFrame()
        ner_entities = ner_by_note.get_group(note_id) if note_id in ner_by_note.groups else pd.DataFrame()
        
        gt_count = len(gt_entities)
        ner_count = len(ner_entities)
        
        results["total_gt_entities"] += gt_count
        results["total_ner_entities"] += ner_count
        
        # Match entities using span overlap
        matched_gt = set()
        matched_ner = set()
        
        for gt_idx, gt_row in gt_entities.iterrows():
            gt_start = gt_row["start"]
            gt_end = gt_row["end"]
            gt_text = gt_row.get("gt_span_text", "")
            
            best_match = None
            best_iou = 0
            
            for ner_idx, ner_row in ner_entities.iterrows():
                if ner_idx in matched_ner:
                    continue
                
                ner_start = ner_row["start"]
                ner_end = ner_row["end"]
                
                if check_span_overlap(gt_start, gt_end, ner_start, ner_end, overlap_threshold):
                    # Calculate exact IoU for best match
                    overlap_start = max(gt_start, ner_start)
                    overlap_end = min(gt_end, ner_end)
                    overlap_length = overlap_end - overlap_start
                    union_length = (gt_end - gt_start) + (ner_end - ner_start) - overlap_length
                    iou = overlap_length / union_length if union_length > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = ner_idx
            
            if best_match is not None:
                matched_gt.add(gt_idx)
                matched_ner.add(best_match)
            else:
                # Record unmatched ground truth entity
                results["unmatched_gt"].append({
                    "note_id": note_id,
                    "start": gt_start,
                    "end": gt_end,
                    "text": gt_text
                })
        
        # Record unmatched NER predictions (false positives)
        for ner_idx, ner_row in ner_entities.iterrows():
            if ner_idx not in matched_ner:
                results["unmatched_ner"].append({
                    "note_id": note_id,
                    "start": ner_row["start"],
                    "end": ner_row["end"],
                    "text": ner_row["span_text"]
                })
        
        matched_count = len(matched_gt)
        results["matched_entities"] += matched_count
        
        # Store per-note results
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
    total_matched = results["matched_entities"]
    
    results["overall_recall"] = total_matched / total_gt if total_gt > 0 else 0
    results["overall_precision"] = total_matched / total_ner if total_ner > 0 else 0
    results["overall_f1"] = (
        2 * results["overall_precision"] * results["overall_recall"] / 
        (results["overall_precision"] + results["overall_recall"])
        if (results["overall_precision"] + results["overall_recall"]) > 0 else 0
    )
    
    return results


def print_diagnostic_report(results: Dict, verbose: bool = False):
    """Print a detailed diagnostic report."""
    print("\n" + "=" * 80)
    print("NER RECALL DIAGNOSTIC REPORT")
    print("=" * 80)
    
    print("\n[OVERALL METRICS]")
    print(f"  Ground Truth Entities: {results['total_gt_entities']}")
    print(f"  NER Detected Entities: {results['total_ner_entities']}")
    print(f"  Matched Entities:      {results['matched_entities']}")
    print()
    print(f"  NER Recall:     {results['overall_recall']:.4f} ({results['overall_recall']*100:.2f}%)")
    print(f"  NER Precision:  {results['overall_precision']:.4f} ({results['overall_precision']*100:.2f}%)")
    print(f"  NER F1-Score:   {results['overall_f1']:.4f}")
    
    # Diagnosis
    print("\n[DIAGNOSIS]")
    recall = results['overall_recall']
    precision = results['overall_precision']
    
    if recall < 0.7:
        print(f"  ⚠️  LOW RECALL ({recall:.2%})")
        print(f"     → NER is MISSING ~{(1-recall)*100:.1f}% of entities!")
        print(f"     → This is your PRIMARY problem.")
        print(f"     → Ground truth entities not detected: {results['total_gt_entities'] - results['matched_entities']}")
    else:
        print(f"  ✓  GOOD RECALL ({recall:.2%})")
        print(f"     → NER is finding most entities.")
    
    if precision < 0.7:
        print(f"  ⚠️  LOW PRECISION ({precision:.2%})")
        print(f"     → NER is creating too many false positives.")
        print(f"     → Extra entities detected: {results['total_ner_entities'] - results['matched_entities']}")
    else:
        print(f"  ✓  GOOD PRECISION ({precision:.2%})")
    
    # Per-note breakdown
    if verbose:
        print("\n[PER-NOTE BREAKDOWN]")
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
    
    # Show missed entities
    print("\n[MISSED ENTITIES - NER Failed to Detect]")
    if results["unmatched_gt"]:
        print(f"  Total missed: {len(results['unmatched_gt'])}")
        print(f"\n  Sample of missed entities (first 20):")
        for i, entity in enumerate(results["unmatched_gt"][:20], 1):
            print(f"    {i}. Note {entity['note_id']}: '{entity['text']}' [{entity['start']}-{entity['end']}]")
        if len(results["unmatched_gt"]) > 20:
            print(f"    ... and {len(results['unmatched_gt']) - 20} more")
    else:
        print("  ✓ No missed entities!")
    
    # Show false positives
    if verbose and results["unmatched_ner"]:
        print("\n[FALSE POSITIVES - NER Detected but Not in Ground Truth]")
        print(f"  Total false positives: {len(results['unmatched_ner'])}")
        print(f"\n  Sample (first 10):")
        for i, entity in enumerate(results["unmatched_ner"][:10], 1):
            print(f"    {i}. Note {entity['note_id']}: '{entity['text']}' [{entity['start']}-{entity['end']}]")
    
    print("\n" + "=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose NER recall issues")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input notes CSV (defaults to ../../data/mimic-iv_notes_training_set.csv)"
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help="Ground truth CSV (defaults to ../../data/train_annotations.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save detailed results to CSV"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap threshold for matching (default: 0.5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-note breakdown"
    )
    
    args = parser.parse_args()
    
    # Default paths
    input_path = args.input or ROOT.parent.parent / "data" / "mimic-iv_notes_training_set.csv"
    truth_path = args.truth or ROOT.parent.parent / "data" / "train_annotations.csv"
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    if not truth_path.exists():
        print(f"Error: Ground truth file not found: {truth_path}")
        return 1
    
    print(f"[SETUP] Loading data...")
    print(f"  Input:  {input_path}")
    print(f"  Truth:  {truth_path}")
    
    # Load data
    notes_df = load_input_notes(input_path)
    gt_df = load_ground_truth(truth_path)
    
    print(f"\n[DATA] Loaded {len(notes_df)} notes with {len(gt_df)} ground truth entities")
    
    # Initialize pipeline (NER only)
    print(f"\n[PIPELINE] Initializing NER system...")
    pipeline = RAGGPTPipeline(verbose=False)
    
    # Extract entities using NER only
    print(f"\n[NER] Running entity extraction...")
    ner_df = extract_ner_only(pipeline, notes_df)
    
    print(f"[NER] Extracted {len(ner_df)} entities total")
    
    # Calculate metrics
    print(f"\n[EVAL] Calculating NER metrics...")
    results = calculate_ner_metrics(gt_df, ner_df, overlap_threshold=args.overlap)
    
    # Print report
    print_diagnostic_report(results, verbose=args.verbose)
    
    # Save detailed results if requested
    if args.output:
        output_data = {
            "metric": ["recall", "precision", "f1", "total_gt", "total_ner", "matched"],
            "value": [
                results["overall_recall"],
                results["overall_precision"],
                results["overall_f1"],
                results["total_gt_entities"],
                results["total_ner_entities"],
                results["matched_entities"]
            ]
        }
        pd.DataFrame(output_data).to_csv(args.output, index=False)
        print(f"[OUTPUT] Saved results to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
