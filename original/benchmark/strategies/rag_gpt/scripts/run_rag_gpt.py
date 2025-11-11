#!/usr/bin/env python3
"""
CLI: end-to-end pipeline (NER -> RAG -> Coding)

Defaults:
- Input auto-detected from ../../data (repo layout: .../benchmark/strategies/rag_gpt)
- Prints results to terminal by default
- If --results is provided, also writes a CSV there
- At the end, computes F1 metrics using benchmark/evaluation/metric_calculator.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # .../strategies/rag_gpt
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# evaluation (…/benchmark/evaluation/metric_calculator.py)
EVAL_DIR = ROOT.parent.parent / "evaluation"
if EVAL_DIR.exists() and str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Lazy import (will fail gracefully if the file isn't there)
try:
    # The project's module is named `metrics_calculator.py` (plural).
    # Earlier code attempted to import `metric_calculator` (singular),
    # which causes an ImportError even when the file exists under
    # benchmark/evaluation. Import the correct module name here.
    from metrics_calculator import MetricsCalculator  # type: ignore
except Exception:
    MetricsCalculator = None  # type: ignore

from pipeline import RAGGPTPipeline  # noqa: E402


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_default_input() -> Path:
    """
    Find a sensible default CSV under ../../data (i.e., .../benchmark/data).
    Order of preference:
      1) notes.csv
      2) notes_dev.csv
      3) first *.csv in the folder
    You can override the folder with BENCHMARK_DATA_DIR.
    """
    default_dir = Path(os.getenv("BENCHMARK_DATA_DIR", str(ROOT.parent.parent / "data")))
    if not default_dir.exists():
        raise FileNotFoundError(f"Default data directory not found: {default_dir}")

    preferred = ["notes.csv", "notes_dev.csv"]
    for name in preferred:
        p = default_dir / name
        if p.exists():
            return p

    csvs = sorted(default_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {default_dir}")
    return csvs[0]


def _resolve_default_truth() -> Path | None:
    """
    Try common ground-truth filenames in ../../data.
    """
    data_dir = Path(os.getenv("BENCHMARK_DATA_DIR", str(ROOT.parent.parent / "data")))
    if not data_dir.exists():
        return None
    # First, try a set of exact common filenames
    for name in ["ground_truth.csv", "gold.csv", "annotations.csv", "labels.csv"]:
        p = data_dir / name
        if p.exists():
            return p

    # If none of the exact names match, try more flexible substring matches
    # to accept names like 'train_annotations.csv' or 'mimic_gold.csv'.
    keywords = ["ground", "gold", "annotation", "annotations", "label"]
    for f in sorted(data_dir.glob("*.csv")):
        fname = f.name.lower()
        if any(k in fname for k in keywords):
            return f
    return None


def _read_input_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "note_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "note_id"})
    if not {"note_id", "text"}.issubset(df.columns):
        raise ValueError(
            f"Input CSV must contain columns: note_id,text (got: {list(df.columns)})"
        )
    return df


def _read_truth_csv(path: Path) -> pd.DataFrame:
    """
    Read ground-truth and standardize to have at least: note_id, concept_id
    Accept a few common aliases for concept_id.
    """
    df = pd.read_csv(path)
    if "note_id" not in df.columns:
        raise ValueError(f"Ground truth must include 'note_id' (got: {list(df.columns)})")

    # Map possible aliases to 'concept_id'
    alias_map = {
        "concept": "concept_id",
        "code": "concept_id",
        "snomed": "concept_id",
        "snomed_code": "concept_id",
        "gold_concept_id": "concept_id",
        "label": "concept_id",
        "target": "concept_id",
    }
    cols_lower = {c.lower(): c for c in df.columns}
    if "concept_id" not in df.columns:
        for alias_lower, target in alias_map.items():
            if alias_lower in cols_lower:
                df = df.rename(columns={cols_lower[alias_lower]: target})
                break

    if "concept_id" not in df.columns:
        raise ValueError(
            f"Ground truth must include 'concept_id' (got: {list(df.columns)}). "
            f"Consider renaming your SNOMED column to 'concept_id'."
        )

    # Coerce types that the provided MetricsCalculator expects
    df["note_id"] = df["note_id"].astype(int, errors="ignore")
    # leave concept_id as-is; the calculator stringifies when comparing
    return df


def _print_predictions(preds: pd.DataFrame, max_rows: int = 50) -> None:
    cols = [
        "note_id", "start", "end", "concept_id",
        "span_text", "confidence", "anatomy_code", "presence_code"
    ]
    cols = [c for c in cols if c in preds.columns]
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_colwidth", 120,
        "display.width", 0
    ):
        print()
        print("=== RAG+GPT Predictions ===")
        print(preds[cols].to_string(index=False))
        print()
        print(f"[run_rag_gpt] Total predictions: {len(preds)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # Accept a compatibility alias: `-results` -> `--results`
    argv = ["--results" if a == "-results" else a for a in sys.argv[1:]]

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Input CSV with columns: note_id,text (defaults to ../../data/*)")
    ap.add_argument("--results", help="If provided, write predictions CSV to this path")
    ap.add_argument("--truth", help="Ground-truth CSV path (defaults to ../../data/<common-names>)")
    ap.add_argument("--no-metrics", action="store_true", help="Skip F1 evaluation")
    ap.add_argument("--limit", type=int, default=100, help="Max rows to print to terminal (default: 100)")
    ap.add_argument("--no-verbose", action="store_true", help="Silence pipeline logs")
    args = ap.parse_args(argv)

    # Input
    input_path = Path(args.input) if args.input else _resolve_default_input()
    df = _read_input_csv(input_path)

    # Run pipeline
    start_time = time.time()
    pipeline = RAGGPTPipeline(verbose=not args.no_verbose)
    preds = pipeline.predict(df)
    exec_time = time.time() - start_time

    # Write CSV only if --results was passed; always print to terminal
    if args.results:
        out_path = Path(args.results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[run_rag_gpt] Wrote predictions to {out_path}")

    _print_predictions(preds, max_rows=args.limit)

    # -----------------------------------------------------------------
    # Metrics (F1) at the end
    # -----------------------------------------------------------------
    if args.no_metrics:
        return

    if MetricsCalculator is None:
        print("[EVAL] metric_calculator.py no disponible en ../../evaluation; omitiendo métricas.")
        return

    truth_path = Path(args.truth) if args.truth else _resolve_default_truth()
    if not truth_path or not truth_path.exists():
        print("[EVAL] Ground truth no encontrado (usa --truth <path> o coloca ground_truth.csv/gold.csv/annotations.csv/labels.csv en ../../data).")
        return

    try:
        gt = _read_truth_csv(truth_path)
    except Exception as e:
        print(f"[EVAL] Error leyendo ground truth: {e}")
        return

    # La clase suministrada compara por (note_id, concept_id).
    calc = MetricsCalculator()
    metrics = calc.calculate_metrics(predictions=preds, ground_truth=gt, strategy_name="RAG+GPT")
    report = calc.format_single_report(metrics, execution_time=exec_time, strategy_name="RAG+GPT")
    print(report)


if __name__ == "__main__":
    main()
