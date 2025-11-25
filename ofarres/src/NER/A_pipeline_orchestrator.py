#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A_pipeline_orchestrator.py - The Pipeline Orchestrator

RESPONSIBILITY: Execute pipeline steps sequentially and audit the performance
                after each step using specific "RAG-Friendly" metrics.

SEQUENTIAL EXECUTION:
    01_gather_assembly.py -> 02_assign_ranks.py -> 03_safe_deduplication.py -> 04_linguistic_filter.py

RAG-FRIENDLY BENCHMARK LOGIC:
    True Positive Criteria:
    1. There is any physical overlap (IoU > 0.1)
    2. AND Text Containment is satisfied:
       - GT_text in Pred_text (Context Expansion)
       - OR Pred_text in GT_text (Partial Match)
    
    Constraint: 1-to-1 Matching
    - A single Prediction cannot count as a match for two distinct GT entities
    - This prevents "Bad Merges" from inflating Recall

OUTPUT: Dashboard table showing Step Name | Entity Count | Recall | Precision | F1
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports ---
from src.utils.metrics import calculate_iou

# Dynamic import for step modules (names start with numbers)
import importlib.util

def load_step_module(step_file: str):
    """Dynamically load a step module by file path."""
    module_path = PROJECT_ROOT / "src" / "NER" / "postprocessor" / step_file
    spec = importlib.util.spec_from_file_location(step_file.replace('.py', ''), module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- Constants ---
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
MIN_IOU_OVERLAP = 0.1  # Minimum IoU for physical overlap


def text_containment_match(pred_text: str, gt_text: str) -> bool:
    """
    Verifica si hay una relación de contención textual entre predicción y GT.
    
    Criterios (después de normalizar a lowercase y strip):
    - Context Expansion: GT está contenido en Pred (ej: "acute hemorrhage" contiene "hemorrhage")
    - Partial Match: Pred está contenido en GT
    
    Returns: True si hay contención en cualquier dirección.
    """
    pred_norm = pred_text.lower().strip()
    gt_norm = gt_text.lower().strip()
    
    return gt_norm in pred_norm or pred_norm in gt_norm


def load_ground_truth() -> Dict[str, List[Dict]]:
    """Load ground truth data."""
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['note_id']: item['annotations'] for item in data}


def load_notes() -> Dict[str, str]:
    """Load notes text."""
    with open(NOTES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['note_id']: item['text'] for item in data}


def evaluate_rag_friendly(
    predictions: Dict[str, List[Dict]], 
    ground_truth: Dict[str, List[Dict]],
    notes: Dict[str, str]
) -> Dict[str, float]:
    """
    Evaluate predictions using RAG-Friendly metrics.
    
    RAG-Friendly TP Criteria:
    1. IoU > 0.1 (physical overlap)
    2. Text Containment (GT in Pred OR Pred in GT)
    3. 1-to-1 matching (prevent bad merges)
    
    Returns: Dict with precision, recall, f1, tp, fp, fn counts
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for note_id, gt_list in ground_truth.items():
        pred_list = predictions.get(note_id, [])
        note_text = notes.get(note_id, '')
        
        matched_gt_indices: Set[int] = set()
        matched_pred_indices: Set[int] = set()
        
        # Ensure predictions have text
        preds_with_text = []
        for p in pred_list:
            p_copy = dict(p)
            if 'text' not in p_copy:
                p_copy['text'] = note_text[p_copy['start']:p_copy['end']]
            preds_with_text.append(p_copy)
        
        # Match GT to Predictions (1-to-1)
        for gt_idx, gt in enumerate(gt_list):
            gt_text = gt.get('text', note_text[gt['start']:gt['end']])
            best_pred_idx = None
            best_iou = -1.0
            
            for pred_idx, pred in enumerate(preds_with_text):
                # Skip if already matched
                if pred_idx in matched_pred_indices:
                    continue
                
                iou = calculate_iou(pred, gt)
                
                # Condition A: Physical overlap
                if iou <= MIN_IOU_OVERLAP:
                    continue
                
                pred_text = pred.get('text', '')
                
                # Condition B: Text containment
                if not text_containment_match(pred_text, gt_text):
                    continue
                
                # Valid match - track best by IoU
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                matched_gt_indices.add(gt_idx)
                matched_pred_indices.add(best_pred_idx)
        
        # Calculate TP, FP, FN for this note
        tp = len(matched_gt_indices)
        fp = len(preds_with_text) - len(matched_pred_indices)
        fn = len(gt_list) - len(matched_gt_indices)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def format_preds_for_eval(step_data: List[Dict]) -> Dict[str, List[Dict]]:
    """Convert step output to format suitable for evaluation."""
    return {item['note_id']: item['annotations'] for item in step_data}


def print_dashboard_header():
    """Print the dashboard header."""
    print("\n" + "=" * 100)
    print(" PIPELINE PERFORMANCE DASHBOARD (RAG-Friendly Metrics)")
    print("=" * 100)
    print(f"{'Step Name':<35} | {'Entities':<10} | {'Recall':<10} | {'Precision':<10} | {'F1':<10}")
    print("-" * 100)


def print_dashboard_row(step_name: str, entity_count: int, metrics: Dict):
    """Print a row in the dashboard."""
    print(f"{step_name:<35} | {entity_count:<10} | {metrics['recall']:<10.2%} | {metrics['precision']:<10.2%} | {metrics['f1']:<10.4f}")


def print_dashboard_footer():
    """Print the dashboard footer."""
    print("=" * 100)


def run_pipeline():
    """
    Main pipeline orchestrator.
    Runs all steps sequentially and audits performance after each step.
    """
    print("\n" + "=" * 100)
    print(" NER PIPELINE ORCHESTRATOR")
    print(" RAG-Friendly High-Recall Entity Extraction")
    print("=" * 100)
    
    # Load ground truth and notes
    print("\n[Orchestrator] Loading ground truth and notes...")
    ground_truth = load_ground_truth()
    notes = load_notes()
    
    total_gt = sum(len(anns) for anns in ground_truth.values())
    print(f"[Orchestrator] Ground Truth: {total_gt} entities across {len(ground_truth)} notes")
    
    # Results storage
    results = []
    
    # === STEP 01: Harvester ===
    print("\n")
    step_01_module = load_step_module("01_gather_assembly.py")
    step_01_data = step_01_module.run_harvester(verbose=True)
    step_01_preds = format_preds_for_eval(step_01_data)
    step_01_count = sum(len(anns) for anns in step_01_preds.values())
    step_01_metrics = evaluate_rag_friendly(step_01_preds, ground_truth, notes)
    results.append(("01_gather_assembly", step_01_count, step_01_metrics))
    
    # === STEP 02: Classifier ===
    print("\n")
    step_02_module = load_step_module("02_assign_ranks.py")
    step_02_data = step_02_module.run_classifier(verbose=True)
    step_02_preds = format_preds_for_eval(step_02_data)
    step_02_count = sum(len(anns) for anns in step_02_preds.values())
    step_02_metrics = evaluate_rag_friendly(step_02_preds, ground_truth, notes)
    results.append(("02_assign_ranks", step_02_count, step_02_metrics))
    
    # === STEP 03: Safe Deduplication ===
    print("\n")
    step_03_module = load_step_module("03_safe_deduplication.py")
    step_03_data = step_03_module.run_safe_deduplication(verbose=True)
    step_03_preds = format_preds_for_eval(step_03_data)
    step_03_count = sum(len(anns) for anns in step_03_preds.values())
    step_03_metrics = evaluate_rag_friendly(step_03_preds, ground_truth, notes)
    results.append(("03_safe_deduplication", step_03_count, step_03_metrics))
    
    # === STEP 04: Linguistic Filter ===
    print("\n")
    step_04_module = load_step_module("04_linguistic_filter.py")
    step_04_data = step_04_module.run_linguistic_filter(verbose=True)
    step_04_preds = format_preds_for_eval(step_04_data)
    step_04_count = sum(len(anns) for anns in step_04_preds.values())
    step_04_metrics = evaluate_rag_friendly(step_04_preds, ground_truth, notes)
    results.append(("04_linguistic_filter", step_04_count, step_04_metrics))
    
    # === DASHBOARD ===
    print_dashboard_header()
    for step_name, count, metrics in results:
        print_dashboard_row(step_name, count, metrics)
    print_dashboard_footer()
    
    # === DETAILED ANALYSIS ===
    print("\n" + "=" * 80)
    print(" DETAILED ANALYSIS")
    print("=" * 80)
    
    # Final metrics
    final_metrics = results[-1][2]
    print(f"\n  Final Step Results:")
    print(f"    True Positives:  {final_metrics['tp']}")
    print(f"    False Positives: {final_metrics['fp']}")
    print(f"    False Negatives: {final_metrics['fn']}")
    
    # Recall check
    if final_metrics['recall'] >= 0.99:
        print(f"\n  [SUCCESS] Recall is {final_metrics['recall']:.1%} - Target achieved!")
    else:
        print(f"\n  [WARNING] Recall is {final_metrics['recall']:.1%} - Below 100% target")
        print(f"            {final_metrics['fn']} GT entities were not matched")
    
    print("\n" + "=" * 80)
    
    return results


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
