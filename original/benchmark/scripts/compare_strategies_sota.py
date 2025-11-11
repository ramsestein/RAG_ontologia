#!/usr/bin/env python3
"""
Script para comparar RAG-GPT vs KIRIS usando el Benchmark SOTA
Ejecuta ambas estrategias y genera una comparación lado a lado
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

# Setup paths
BENCHMARK_DIR = Path(__file__).resolve().parent.parent
STRATEGIES_DIR = BENCHMARK_DIR / "strategies"
DATA_DIR = BENCHMARK_DIR / "data"
RAG_GPT_DIR = STRATEGIES_DIR / "rag_gpt"

# Add paths to sys.path
sys.path.insert(0, str(STRATEGIES_DIR))
sys.path.insert(0, str(RAG_GPT_DIR / "src"))
sys.path.insert(0, str(RAG_GPT_DIR / "scripts"))

# Import benchmark function
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark_sota", RAG_GPT_DIR / "scripts" / "benchmark_sota.py")
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)
calculate_macro_f1 = benchmark_module.calculate_macro_f1

print("\n" + "="*100)
print("BENCHMARK SOTA COMPARISON: RAG-GPT vs KIRIS")
print("="*100)

# Load data
input_path = DATA_DIR / "mimic-iv_notes_training_set.csv"
truth_path = DATA_DIR / "train_annotations.csv"

print(f"\nLoading data...")
print(f"  Input: {input_path}")
print(f"  Truth: {truth_path}")

notes_df = pd.read_csv(input_path)
ground_truth = pd.read_csv(truth_path)

print(f"\n[OK] Loaded {len(notes_df)} notes")
print(f"[OK] Loaded {len(ground_truth)} ground truth annotations")

# Dictionary to store results
all_results = {}

# ============================================================================
# 1. RAG-GPT Strategy
# ============================================================================
print("\n" + "="*100)
print("[1/2] EVALUATING RAG-GPT STRATEGY")
print("="*100)

try:
    rag_gpt_path = STRATEGIES_DIR / "rag_gpt"
    sys.path.insert(0, str(rag_gpt_path / "src"))
    
    from pipeline import RAGGPTPipeline
    
    start_time = time.time()
    pipeline = RAGGPTPipeline(verbose=True)
    rag_predictions = pipeline.predict(notes_df)
    rag_exec_time = time.time() - start_time
    
    print(f"\n[OK] RAG-GPT completed in {rag_exec_time:.2f} seconds")
    print(f"[OK] Generated {len(rag_predictions)} predictions")
    
    # Calculate SOTA metrics
    rag_results = calculate_macro_f1(
        predictions=rag_predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        verbose=True
    )
    rag_results['execution_time'] = rag_exec_time
    rag_results['strategy_name'] = 'RAG-GPT'
    all_results['RAG-GPT'] = rag_results
    
except Exception as e:
    print(f"\n[ERROR] Failed to run RAG-GPT: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. KIRIS Strategy
# ============================================================================
print("\n\n" + "="*100)
print("[2/2] EVALUATING KIRIS STRATEGY")
print("="*100)

try:
    # Import KIRIS
    sys.path.insert(0, str(STRATEGIES_DIR))
    
    # Import from 01_kiris.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("kiris_module", STRATEGIES_DIR / "01_kiris.py")
    kiris_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kiris_module)
    
    start_time = time.time()
    kiris = kiris_module.RealKIRIsStrategy()
    kiris_predictions = kiris.predict(notes_df)
    kiris_exec_time = time.time() - start_time
    
    print(f"\n[OK] KIRIS completed in {kiris_exec_time:.2f} seconds")
    print(f"[OK] Generated {len(kiris_predictions)} predictions")
    
    # Calculate SOTA metrics
    kiris_results = calculate_macro_f1(
        predictions=kiris_predictions,
        ground_truth=ground_truth,
        iou_threshold=0.5,
        verbose=True
    )
    kiris_results['execution_time'] = kiris_exec_time
    kiris_results['strategy_name'] = 'KIRIS'
    all_results['KIRIS'] = kiris_results
    
except Exception as e:
    print(f"\n[ERROR] Failed to run KIRIS: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Comparison Table
# ============================================================================
if len(all_results) >= 2:
    print("\n\n" + "="*100)
    print("COMPARATIVE RESULTS - BENCHMARK SOTA (Strict Evaluation)")
    print("="*100)
    
    # Main metrics table
    print(f"\n{'Metric':<30} {'RAG-GPT':<20} {'KIRIS':<20} {'Difference':<20}")
    print("-" * 90)
    
    rag = all_results.get('RAG-GPT', {})
    kiris = all_results.get('KIRIS', {})
    
    # Macro F1
    rag_macro = rag.get('macro_f1', 0)
    kiris_macro = kiris.get('macro_f1', 0)
    diff_macro = kiris_macro - rag_macro
    print(f"{'Macro-Average F1':<30} {rag_macro:<20.4f} {kiris_macro:<20.4f} {diff_macro:+.4f}")
    
    # Micro Precision
    rag_prec = rag.get('micro_precision', 0)
    kiris_prec = kiris.get('micro_precision', 0)
    diff_prec = kiris_prec - rag_prec
    print(f"{'Micro Precision':<30} {rag_prec:<20.4f} {kiris_prec:<20.4f} {diff_prec:+.4f}")
    
    # Micro Recall
    rag_rec = rag.get('micro_recall', 0)
    kiris_rec = kiris.get('micro_recall', 0)
    diff_rec = kiris_rec - rag_rec
    print(f"{'Micro Recall':<30} {rag_rec:<20.4f} {kiris_rec:<20.4f} {diff_rec:+.4f}")
    
    # Micro F1
    rag_micro = rag.get('micro_f1', 0)
    kiris_micro = kiris.get('micro_f1', 0)
    diff_micro = kiris_micro - rag_micro
    print(f"{'Micro F1':<30} {rag_micro:<20.4f} {kiris_micro:<20.4f} {diff_micro:+.4f}")
    
    print()
    print("-" * 90)
    
    # Counts
    rag_tp = rag.get('total_tp', 0)
    kiris_tp = kiris.get('total_tp', 0)
    print(f"{'True Positives (TP)':<30} {rag_tp:<20} {kiris_tp:<20} {kiris_tp - rag_tp:+d}")
    
    rag_fp = rag.get('total_fp', 0)
    kiris_fp = kiris.get('total_fp', 0)
    print(f"{'False Positives (FP)':<30} {rag_fp:<20} {kiris_fp:<20} {kiris_fp - rag_fp:+d}")
    
    rag_fn = rag.get('total_fn', 0)
    kiris_fn = kiris.get('total_fn', 0)
    print(f"{'False Negatives (FN)':<30} {rag_fn:<20} {kiris_fn:<20} {kiris_fn - rag_fn:+d}")
    
    print()
    print("-" * 90)
    
    # Execution time
    rag_time = rag.get('execution_time', 0)
    kiris_time = kiris.get('execution_time', 0)
    print(f"{'Execution Time (s)':<30} {rag_time:<20.2f} {kiris_time:<20.2f} {kiris_time - rag_time:+.2f}")
    
    # Per-note comparison
    print("\n" + "="*100)
    print("PER-NOTE F1 COMPARISON")
    print("="*100)
    
    rag_per_note = rag.get('per_note_results', [])
    kiris_per_note = kiris.get('per_note_results', [])
    
    print(f"\n{'Note':<10} {'RAG-GPT F1':<15} {'KIRIS F1':<15} {'Difference':<15} {'Winner':<10}")
    print("-" * 65)
    
    for i in range(5):
        note_id = i + 1
        rag_note = next((n for n in rag_per_note if n['note_id'] == note_id), None)
        kiris_note = next((n for n in kiris_per_note if n['note_id'] == note_id), None)
        
        if rag_note and kiris_note:
            rag_f1 = rag_note['f1']
            kiris_f1 = kiris_note['f1']
            diff = kiris_f1 - rag_f1
            winner = 'KIRIS' if kiris_f1 > rag_f1 else ('RAG-GPT' if rag_f1 > kiris_f1 else 'TIE')
            print(f"{'Note ' + str(note_id):<10} {rag_f1:<15.4f} {kiris_f1:<15.4f} {diff:+.4f}          {winner}")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    if kiris_macro > rag_macro:
        improvement = ((kiris_macro - rag_macro) / rag_macro) * 100
        print(f"\n[*] KIRIS outperforms RAG-GPT by {improvement:.1f}% in Macro-Average F1")
        print(f"    KIRIS: {kiris_macro:.4f} vs RAG-GPT: {rag_macro:.4f}")
    elif rag_macro > kiris_macro:
        improvement = ((rag_macro - kiris_macro) / kiris_macro) * 100
        print(f"\n[*] RAG-GPT outperforms KIRIS by {improvement:.1f}% in Macro-Average F1")
        print(f"    RAG-GPT: {rag_macro:.4f} vs KIRIS: {kiris_macro:.4f}")
    else:
        print(f"\n[*] Both strategies achieved the same Macro-Average F1: {rag_macro:.4f}")
    
    print(f"\n[*] Total annotations matched:")
    print(f"    RAG-GPT: {rag_tp}/115 ({rag_tp/115*100:.1f}%)")
    print(f"    KIRIS: {kiris_tp}/115 ({kiris_tp/115*100:.1f}%)")
    
    print("\n" + "="*100)

else:
    print("\n[WARNING] Could not compare - one or both strategies failed to run")

print("\n[DONE] Benchmark comparison completed\n")
