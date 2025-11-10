import pandas as pd
from typing import Dict, List, Tuple


class MetricsCalculator:
    def __init__(self):
        """Initialize the MetricsCalculator."""
        pass
    
    def calculate_metrics(self,
                        predictions: pd.DataFrame,
                        ground_truth: pd.DataFrame,
                        strategy_name: str) -> Dict:
        # Empty guard
        if predictions is None or len(predictions) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "predictions": 0,
                "matches": 0,
                "partial_matches": 0,
                "ground_truth": int(len(ground_truth) if ground_truth is not None else 0),
                "coverage": 0.0
            }

        # Build sets of unique (note_id, concept_id)
        preds_df = predictions.copy()
        preds_df["concept_id"] = preds_df["concept_id"].astype(str)
        pred_pairs = set(zip(preds_df["note_id"], preds_df["concept_id"]))

        gt_df = ground_truth.copy()
        gt_df["concept_id"] = gt_df["concept_id"].astype(str)
        gt_pairs = set(zip(gt_df["note_id"], gt_df["concept_id"]))

        # True Positives / False Positives / False Negatives
        tp = len(pred_pairs & gt_pairs)
        fp = len(pred_pairs - gt_pairs)
        fn = len(gt_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Partial matches = prediction has correct note_id but wrong/missing concept in GT
        gt_notes = set(gt_df["note_id"].tolist())
        partial_matches = sum(1 for (n, c) in pred_pairs if (n in gt_notes) and ((n, c) not in gt_pairs))

        # Coverage = fraction of GT notes for which we produced at least one prediction
        pred_notes = set(preds_df["note_id"].tolist())
        truth_notes = set(gt_df["note_id"].tolist())
        coverage = len(pred_notes & truth_notes) / len(truth_notes) if len(truth_notes) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": len(pred_pairs),   # count unique predictions
            "matches": tp,
            "partial_matches": partial_matches,
            "ground_truth": len(gt_pairs),    # count unique gold pairs
            "coverage": coverage
        }
        
    def format_single_report(self, 
                            metrics: Dict, 
                            execution_time: float, 
                            strategy_name: str) -> str:
        
        report = []
        report.append("\n" + "="*80)
        report.append(f"RESULTS - {strategy_name}")
        report.append("="*80)
        
        report.append("\n[METRICS] METRICS:")
        report.append(f"   Precision:  {metrics['precision']:.4f}")
        report.append(f"   Recall:     {metrics['recall']:.4f}")
        report.append(f"   F1-Score:   {metrics['f1']:.4f}")
        report.append(f"   Coverage:   {metrics['coverage']:.4f}")
        
        report.append("\n[CHART] COUNTS:")
        report.append(f"   Predictions:     {metrics['predictions']}")
        report.append(f"   Exact Matches:   {metrics['matches']}")
        report.append(f"   Partial Matches: {metrics['partial_matches']}")
        report.append(f"   Ground Truth:    {metrics['ground_truth']}")
        
        report.append(f"\n[TIME]  EXECUTION TIME: {execution_time:.2f} seconds")
        report.append("="*80)
        
        return "\n".join(report)
    
    def format_comparison_report(self, results_dict: Dict[str, Dict]) -> str:
        
        if not results_dict:
            return "\nNo results to compare."
        
        report = []
        report.append("\n" + "="*120)
        report.append("COMPARISON REPORT - ALL STRATEGIES")
        report.append("="*120)
        
        # Table header
        header = f"\n{'Strategy':<30} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Pred':<6} {'Match':<6} {'Time':<10}"
        report.append(header)
        report.append("-" * 120)
        
        # Sort by F1-Score
        sorted_results = sorted(
            results_dict.items(),
            key=lambda x: x[1]['metrics'].get('f1', 0),
            reverse=True
        )
        
        # Table rows
        for name, data in sorted_results:
            metrics = data['metrics']
            exec_time = data['execution_time']
            
            row = (f"{name:<30} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['predictions']:<6} "
                  f"{metrics['matches']:<6} "
                  f"{exec_time:<10.3f}s")
            report.append(row)
        
        # Ranking
        report.append("\n" + "="*80)
        report.append("RANKING BY F1-SCORE:")
        report.append("="*80)
        
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd", "   4th"]
        for i, (name, data) in enumerate(sorted_results):
            medal = medals[i] if i < len(medals) else f"   {i+1}th"
            f1_score = data['metrics']['f1']
            report.append(f"{medal} place: {name:<25} (F1 = {f1_score:.4f})")
        
        report.append("\n" + "="*120)
        
        return "\n".join(report)
