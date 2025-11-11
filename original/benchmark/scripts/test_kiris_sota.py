#!/usr/bin/env python3
"""
Quick test: Run KIRIS only with SOTA benchmark
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

# Import benchmark function
sys.path.insert(0, str(RAG_GPT_DIR / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("benchmark_sota", RAG_GPT_DIR / "scripts" / "benchmark_sota.py")
benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)
calculate_macro_f1 = benchmark_module.calculate_macro_f1

print("\n" + "="*80)
print("KIRIS STRATEGY - BENCHMARK SOTA")
print("="*80)

# Load data
input_path = DATA_DIR / "mimic-iv_notes_training_set.csv"
truth_path = DATA_DIR / "train_annotations.csv"

notes_df = pd.read_csv(input_path)
ground_truth = pd.read_csv(truth_path)

print(f"\n[OK] Loaded {len(notes_df)} notes")
print(f"[OK] Loaded {len(ground_truth)} ground truth annotations")

# Import KIRIS
sys.path.insert(0, str(STRATEGIES_DIR))
spec = importlib.util.spec_from_file_location("kiris_module", STRATEGIES_DIR / "01_kiris.py")
kiris_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kiris_module)

print("\n" + "="*80)
print("RUNNING KIRIS STRATEGY")
print("="*80)

start_time = time.time()
kiris = kiris_module.RealKIRIsStrategy()
kiris_predictions = kiris.predict(notes_df)
kiris_exec_time = time.time() - start_time

print(f"\n[OK] KIRIS completed in {kiris_exec_time:.2f} seconds")
print(f"[OK] Generated {len(kiris_predictions)} predictions")

# Calculate SOTA metrics
print("\n" + "="*80)
print("CALCULATING SOTA METRICS")
print("="*80)

kiris_results = calculate_macro_f1(
    predictions=kiris_predictions,
    ground_truth=ground_truth,
    iou_threshold=0.5,
    verbose=True
)

print("\n" + "="*80)
print(f"[TIME] Total execution time: {kiris_exec_time:.2f} seconds")
print("="*80)
