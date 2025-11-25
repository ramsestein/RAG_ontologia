import json
import time
import sys
import itertools
import importlib
from pathlib import Path
from typing import List, Dict

# --- Paths Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- FIX: Import from sibling file directly ---
# Since both files are in 'benchmarks/', we import directly from 'diagnose_NER'
# instead of 'benchmarks.diagnose_NER'
try:
    from diagnose_NER import (
        load_data, 
        load_ner_worker, 
        get_detailed_matches, 
        text_containment_match,
        deduplicate_predictions,
        CONFIG_PATH, NOTES_PATH, GT_PATH
    )
except ModuleNotFoundError:
    # Fallback in case it's run as a module from root
    from benchmarks.diagnose_NER import (
        load_data, 
        load_ner_worker, 
        get_detailed_matches, 
        text_containment_match,
        deduplicate_predictions,
        CONFIG_PATH, NOTES_PATH, GT_PATH
    )

from src.utils.metrics import calculate_ner_micro_f1

def evaluate_permutation(worker_order: tuple, loaded_workers: Dict, notes: Dict, gt_data: Dict, iou_th: float):
    """
    Runs the assembly pipeline for a specific order of workers.
    """
    all_preds = {}
    
    # 1. Process Notes
    for nid, text in notes.items():
        raw_preds = []
        
        # Execute workers IN ORDER
        for wid in worker_order:
            worker = loaded_workers[wid]
            w_preds = worker.extract_entities(text)
            raw_preds.extend(w_preds)

        # Final Deduplication (Union)
        preds = deduplicate_predictions(raw_preds)
        all_preds[nid] = preds

    # 2. Calculate Global Metrics
    global_m = calculate_ner_micro_f1(all_preds, gt_data, iou_th)
    
    return {
        "Order": " + ".join(worker_order),
        "F1": global_m['f1'],
        "Rec": global_m['recall'],
        "Prec": global_m['precision'],
        "TP": global_m['tp']
    }

def run_efficiency_test(worker_order: tuple, worker_cache: Dict, notes: Dict, gt_data: Dict, iou_th: float):
    """
    Calculates the 'Efficiency Score': Sum of cumulative recall at each step.
    Higher score = The heavy lifters are at the start of the chain.
    
    Uses Text Containment + IoU matching logic for RAG-Ready Recall.
    """
    MIN_IOU_OVERLAP = 0.1  # Minimum IoU for physical overlap
    
    # Generate unique IDs for all GT items to track coverage
    total_gt_ids = set()
    for nid, anns in gt_data.items():
        for ann in anns:
            total_gt_ids.add(f"{nid}_{ann['start']}_{ann['end']}")
    
    total_gt = len(total_gt_ids)
    found_so_far = set()
    cumulative_scores = []
    
    for wid in worker_order:
        # Get pre-calculated predictions from cache
        
        current_step_found = set()
        
        for nid, text in notes.items():
            preds = worker_cache[wid][nid]
            gt_list = gt_data.get(nid, [])
            
            # Check matches using Text Containment + IoU logic
            for gt_item in gt_list:
                gid = f"{nid}_{gt_item['start']}_{gt_item['end']}"
                
                # Skip if already found by previous worker
                if gid in found_so_far:
                    continue
                
                gt_text = gt_item.get('text', text[gt_item['start']:gt_item['end']])
                
                # Check if any prediction matches this GT
                for p in preds:
                    from src.utils.metrics import calculate_iou
                    iou = calculate_iou(p, gt_item)
                    
                    # Condition A: Physical overlap
                    if iou <= MIN_IOU_OVERLAP:
                        continue
                    
                    # Extract prediction text
                    pred_text = text[p['start']:p['end']]
                    
                    # Condition B: Text containment
                    if text_containment_match(pred_text, gt_text):
                        current_step_found.add(gid)
                        break
        
        # Update global found
        found_so_far.update(current_step_found)
        recall_at_step = len(found_so_far) / total_gt if total_gt > 0 else 0
        cumulative_scores.append(recall_at_step)
        
    # Metric: Average Recall over steps (Efficiency)
    # e.g. [0.9, 0.95, 0.99] is better than [0.2, 0.8, 0.99]
    efficiency_score = sum(cumulative_scores) / len(cumulative_scores)
    
    return efficiency_score, cumulative_scores

def main():
    iou = 0.25
    print(f"🚀 Starting Cross-Validation (Permutation Test) [IoU > {iou}]")
    
    # 1. Load Data & Config
    try:
        with open(CONFIG_PATH) as f: reg = json.load(f)
        notes, gt = load_data(NOTES_PATH, GT_PATH)
    except Exception as e: return print(f"[ERROR] {e}")

    worker_ids = list(reg.keys())
    print(f"📋 Models to permute ({len(worker_ids)}): {worker_ids}")
    
    # 2. Pre-load Workers (Once)
    loaded_workers = {}
    print("⏳ Pre-loading models into RAM...")
    for wid in worker_ids:
        w = load_ner_worker(wid, reg[wid])
        if w: loaded_workers[wid] = w
    
    # 3. Pre-calculate Predictions (Optimization)
    # This prevents running NLP inference 120 times. We run it once per worker.
    print("⚡ Caching predictions to speed up permutations...")
    prediction_cache = {} # {wid: {nid: [preds]}}
    for wid, worker in loaded_workers.items():
        prediction_cache[wid] = {}
        for nid, text in notes.items():
            prediction_cache[wid][nid] = worker.extract_entities(text)

    # 4. Permutation Loop
    perms = list(itertools.permutations(loaded_workers.keys()))
    print(f"🔄 Testing all {len(perms)} permutations...")
    
    results = []
    
    for i, p in enumerate(perms):
        # Run efficiency test on cached data
        score, curves = run_efficiency_test(p, prediction_cache, notes, gt, iou)
        
        results.append({
            "Order": p,
            "Score": score,
            "Curve": curves,
            "FinalRecall": curves[-1]
        })

    # 5. Sort by "Efficiency Score" (Area Under Curve)
    results.sort(key=lambda x: x['Score'], reverse=True)
    
    # 6. Report
    print("\n" + "="*100)
    print(f" 🏆 BEST MODEL ORDER (Most Efficient First)")
    print("   Criteria: Maximizes cumulative recall at earliest steps.")
    print("="*100)
    print(f"{'Rank':<4} | {'Efficiency':<10} | {'Model Sequence (1 -> 5)'}")
    print("-" * 100)
    
    for i in range(min(10, len(results))):
        r = results[i]
        order_str = " -> ".join(r['Order'])
        print(f"#{i+1:<3} | {r['Score']:<10.4f} | {order_str}")
        
    print("-" * 100)
    
    best = results[0]
    print(f"\n📈 Cumulative Recall Curve for Best Order:")
    print(f"{'Step':<5} | {'Model':<20} | {'Total Recall':<15} | {'Gain'}")
    print("-" * 60)
    
    for i, mod in enumerate(best['Order']):
        rec = best['Curve'][i]
        prev = best['Curve'][i-1] if i > 0 else 0
        gain = rec - prev
        print(f" {i+1:<4} | {mod:<20} | {rec:<15.2%} | +{gain:.2%}")
    
    print("-" * 60)
    print("\n💡 Insight: Models with +0.00% gain at the end are redundant for this dataset.")

if __name__ == "__main__":
    main()