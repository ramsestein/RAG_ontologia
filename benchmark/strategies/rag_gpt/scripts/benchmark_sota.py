#!/usr/bin/env python3
"""
Benchmark SOTA - Strict Evaluation with Macro-Average F1-Score

This script implements a strict benchmark that evaluates against all 115 annotations
in train_annotations.csv (not just 64 unique pairs).

Matching Criteria:
    - Same note_id
    - Same concept_id
    - Span overlap IoU > 0.5

Metric:
    - Macro-Average F1-Score: Average of F1 scores calculated per-note
    - F1_macro = sum(F1_i for each note) / num_notes
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import RAGGPTPipeline


def calculate_iou(gt_start: int, gt_end: int, pred_start: int, pred_end: int) -> float:
    """
    Calculate Intersection over Union (IoU) for two spans.
    
    Args:
        gt_start: Ground truth start position
        gt_end: Ground truth end position  
        pred_start: Predicted start position
        pred_end: Predicted end position
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Calculate intersection
    overlap_start = max(gt_start, pred_start)
    overlap_end = min(gt_end, pred_end)
    
    if overlap_start >= overlap_end:
        return 0.0  # No overlap
    
    intersection = overlap_end - overlap_start
    
    # Calculate union
    gt_length = gt_end - gt_start
    pred_length = pred_end - pred_start
    union = gt_length + pred_length - intersection
    
    return intersection / union if union > 0 else 0.0


def find_matching_annotations(
    pred_row: pd.Series,
    gt_subset: pd.DataFrame,
    iou_threshold: float = 0.5
) -> Tuple[bool, int]:
    """
    Find if prediction matches any ground truth annotation.
    
    Args:
        pred_row: Single prediction row
        gt_subset: Ground truth annotations for the same note
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        (found_match, gt_index) tuple. gt_index is -1 if no match found.
    """
    pred_concept = str(pred_row['concept_id'])
    pred_start = pred_row['start']
    pred_end = pred_row['end']
    
    best_iou = 0.0
    best_idx = -1
    
    for idx, gt_row in gt_subset.iterrows():
        gt_concept = str(gt_row['concept_id'])
        
        # Must match concept_id
        if pred_concept != gt_concept:
            continue
        
        # Calculate IoU
        iou = calculate_iou(
            gt_row['start'], gt_row['end'],
            pred_start, pred_end
        )
        
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    
    return (best_iou >= iou_threshold, best_idx)


def calculate_f1_for_note(
    pred_subset: pd.DataFrame,
    gt_subset: pd.DataFrame,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate strict F1-score for a single note.
    
    Args:
        pred_subset: Predictions for this note
        gt_subset: Ground truth annotations for this note
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dict with precision, recall, f1, tp, fp, fn counts
    """
    total_pred = len(pred_subset)
    total_gt = len(gt_subset)
    
    if total_pred == 0 and total_gt == 0:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'total_pred': 0,
            'total_gt': 0
        }
    
    if total_pred == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': total_gt,
            'total_pred': 0,
            'total_gt': total_gt
        }
    
    if total_gt == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': total_pred,
            'fn': 0,
            'total_pred': total_pred,
            'total_gt': 0
        }
    
    # Track which GT annotations have been matched (1-to-1 matching)
    matched_gt_indices = set()
    tp = 0
    
    # Iterate through predictions to find matches
    for _, pred_row in pred_subset.iterrows():
        # Create a subset excluding already matched GT annotations
        available_gt = gt_subset[~gt_subset.index.isin(matched_gt_indices)]
        
        found_match, gt_idx = find_matching_annotations(
            pred_row, available_gt, iou_threshold
        )
        
        if found_match:
            tp += 1
            matched_gt_indices.add(gt_idx)
    
    # Calculate metrics
    fp = total_pred - tp
    fn = total_gt - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_pred': total_pred,
        'total_gt': total_gt
    }


def calculate_macro_f1(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    iou_threshold: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Calculate Macro-Average F1-Score across all notes.
    
    Args:
        predictions: DataFrame with predictions
        ground_truth: DataFrame with all 115 ground truth annotations
        iou_threshold: IoU threshold for span matching
        verbose: Print per-note details
    
    Returns:
        Dict with overall metrics and per-note scores
    """
    # Ensure concept_id is string for consistent comparison
    predictions = predictions.copy()
    ground_truth = ground_truth.copy()
    predictions['concept_id'] = predictions['concept_id'].astype(str)
    ground_truth['concept_id'] = ground_truth['concept_id'].astype(str)
    
    # Get unique note IDs
    note_ids = sorted(ground_truth['note_id'].unique())
    
    f1_scores = []
    per_note_results = []
    
    if verbose:
        print("\n" + "="*80)
        print("BENCHMARK SOTA - STRICT EVALUATION (Macro-Average F1)")
        print("="*80)
        print(f"\nIoU Threshold: {iou_threshold}")
        print(f"Total Ground Truth Annotations: {len(ground_truth)}")
        print(f"Total Predictions: {len(predictions)}")
        print(f"Number of Notes: {len(note_ids)}")
        print("\n" + "="*80)
        print("PER-NOTE RESULTS")
        print("="*80)
    
    # Calculate F1 for each note
    for note_id in note_ids:
        pred_subset = predictions[predictions['note_id'] == note_id]
        gt_subset = ground_truth[ground_truth['note_id'] == note_id]
        
        metrics = calculate_f1_for_note(pred_subset, gt_subset, iou_threshold)
        metrics['note_id'] = note_id
        
        f1_scores.append(metrics['f1'])
        per_note_results.append(metrics)
        
        if verbose:
            print(f"\nNote {note_id}:")
            print(f"  Ground Truth Annotations: {metrics['total_gt']}")
            print(f"  Predictions: {metrics['total_pred']}")
            print(f"  True Positives (TP): {metrics['tp']}")
            print(f"  False Positives (FP): {metrics['fp']}")
            print(f"  False Negatives (FN): {metrics['fn']}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # Calculate macro-average
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Calculate micro-average (for comparison)
    total_tp = sum(m['tp'] for m in per_note_results)
    total_fp = sum(m['fp'] for m in per_note_results)
    total_fn = sum(m['fn'] for m in per_note_results)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    if verbose:
        print("\n" + "="*80)
        print("OVERALL METRICS")
        print("="*80)
        print(f"\n[*] MACRO-AVERAGE (Primary Metric):")
        print(f"   F1-Score: {macro_f1:.4f}")
        print(f"   (Average of {len(f1_scores)} per-note F1 scores)")
        
        print(f"\n[*] MICRO-AVERAGE (For Comparison):")
        print(f"   Precision: {micro_precision:.4f}")
        print(f"   Recall:    {micro_recall:.4f}")
        print(f"   F1-Score:  {micro_f1:.4f}")
        
        print(f"\n[*] AGGREGATE COUNTS:")
        print(f"   Total TP: {total_tp}")
        print(f"   Total FP: {total_fp}")
        print(f"   Total FN: {total_fn}")
        print(f"   Total GT Annotations: {len(ground_truth)}")
        print(f"   Total Predictions: {len(predictions)}")
    
    return {
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'per_note_results': per_note_results,
        'f1_scores': f1_scores,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'iou_threshold': iou_threshold
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SOTA - Strict evaluation with Macro-Average F1"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input notes CSV (default: ../../data/mimic-iv_notes_training_set.csv)"
    )
    parser.add_argument(
        "--truth",
        type=str,
        help="Path to ground truth CSV (default: ../../data/train_annotations.csv)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for span matching (default: 0.5)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = ROOT.parent.parent / "data" / "mimic-iv_notes_training_set.csv"
    
    if args.truth:
        truth_path = Path(args.truth)
    else:
        truth_path = ROOT.parent.parent / "data" / "train_annotations.csv"
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    if not truth_path.exists():
        print(f"[ERROR] Ground truth file not found: {truth_path}")
        return 1
    
    verbose = not args.no_verbose
    
    if verbose:
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        print(f"Input notes: {input_path}")
        print(f"Ground truth: {truth_path}")
    
    # Load data
    notes_df = pd.read_csv(input_path)
    ground_truth = pd.read_csv(truth_path)
    
    if verbose:
        print(f"\n[OK] Loaded {len(notes_df)} notes")
        print(f"[OK] Loaded {len(ground_truth)} ground truth annotations")
    
    # Run pipeline
    if verbose:
        print("\n" + "="*80)
        print("RUNNING RAG+GPT PIPELINE")
        print("="*80)
    
    start_time = time.time()
    pipeline = RAGGPTPipeline(verbose=verbose)
    predictions = pipeline.predict(notes_df)
    exec_time = time.time() - start_time
    
    if verbose:
        print(f"\n[OK] Pipeline completed in {exec_time:.2f} seconds")
        print(f"[OK] Generated {len(predictions)} predictions")
    
    # Calculate SOTA metrics
    results = calculate_macro_f1(
        predictions=predictions,
        ground_truth=ground_truth,
        iou_threshold=args.iou_threshold,
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "="*80)
        print(f"[TIME] EXECUTION TIME: {exec_time:.2f} seconds")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
