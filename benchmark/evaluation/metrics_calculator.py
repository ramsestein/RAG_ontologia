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
        if len(predictions) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "predictions": 0,
                "matches": 0,
                "partial_matches": 0,
                "ground_truth": len(ground_truth),
                "coverage": 0.0
            }
        
        # Count exact matches
        exact_matches = 0
        partial_matches = 0
        
        for _, pred in predictions.iterrows():
            pred_concept = str(pred['concept_id'])
            pred_note = pred['note_id']
            
            # Look for exact matches (same note_id and concept_id)
            exact_match_found = False
            for _, true in ground_truth.iterrows():
                if (pred_note == true['note_id'] and 
                    pred_concept == str(true['concept_id'])):
                    exact_matches += 1
                    exact_match_found = True
                    break
            
            # If no exact match, look for partial (same note_id only)
            if not exact_match_found:
                for _, true in ground_truth.iterrows():
                    if pred_note == true['note_id']:
                        partial_matches += 1
                        break
        
        # Calculate metrics
        precision = exact_matches / len(predictions) if len(predictions) > 0 else 0
        recall = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate coverage (how many notes have at least one prediction)
        pred_notes = len(predictions.groupby('note_id'))
        truth_notes = len(ground_truth.groupby('note_id'))
        coverage = pred_notes / truth_notes if truth_notes > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": len(predictions),
            "matches": exact_matches,
            "partial_matches": partial_matches,
            "ground_truth": len(ground_truth),
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
        
        report.append("\n📊 METRICS:")
        report.append(f"   Precision:  {metrics['precision']:.4f}")
        report.append(f"   Recall:     {metrics['recall']:.4f}")
        report.append(f"   F1-Score:   {metrics['f1']:.4f}")
        report.append(f"   Coverage:   {metrics['coverage']:.4f}")
        
        report.append("\n📈 COUNTS:")
        report.append(f"   Predictions:     {metrics['predictions']}")
        report.append(f"   Exact Matches:   {metrics['matches']}")
        report.append(f"   Partial Matches: {metrics['partial_matches']}")
        report.append(f"   Ground Truth:    {metrics['ground_truth']}")
        
        report.append(f"\n⏱  EXECUTION TIME: {execution_time:.2f} seconds")
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
                  f"{exec_time:<10.1f}s")
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
