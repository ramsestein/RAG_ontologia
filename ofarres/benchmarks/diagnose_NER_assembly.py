import json
import time
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports ---
from src.utils.metrics import calculate_iou

# --- Constants ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
IOU_THRESHOLD = 0.25  # The threshold for loose matching

def load_data():
    print(f"[Loader] Notes: {NOTES_PATH}")
    print(f"[Loader] GT:    {GT_PATH}")
    if not NOTES_PATH.exists() or not GT_PATH.exists():
        raise FileNotFoundError("Missing JSON files in /data")
    with open(NOTES_PATH, 'r', encoding='utf-8') as f: n = json.load(f)
    with open(GT_PATH, 'r', encoding='utf-8') as f: g = json.load(f)
    return {i['note_id']: i['text'] for i in n}, {i['note_id']: i['annotations'] for i in g}

def load_all_workers(registry):
    workers = []
    print(f"\n[Assembly] Loading {len(registry)} workers...")
    for ner_id, config in registry.items():
        try:
            mod = importlib.import_module(config['module'])
            cls = getattr(mod, config['class'])
            kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
            instance = cls(**kwargs)
            workers.append(instance)
            print(f"  ✅ {ner_id} Loaded")
        except Exception as e:
            print(f"  ❌ Error loading {ner_id}: {e}")
    return workers

def deduplicate_predictions(preds: List[Dict]) -> List[Dict]:
    """Simple Union: Removes exact duplicate spans."""
    seen = set()
    unique = []
    for p in preds:
        k = (p['start'], p['end'])
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique

def classify_predictions(preds, gt, text):
    """
    Classifies each prediction as TP or FP based on Coverage Logic.
    Returns a list of dicts ready for printing.
    """
    rows = []
    
    # 1. Determine Status for every prediction
    for p in preds:
        # Find ALL GTs covered by this prediction
        matches = []
        for g in gt:
            if calculate_iou(p, g) > IOU_THRESHOLD:
                matches.append(g)
        
        p_text = text[p['start']:p['end']].replace('\n', ' ')
        
        if matches:
            # It is a True Positive
            # It might match multiple nested GTs (e.g. SAH + Hemorrhage)
            match_texts = [text[m['start']:m['end']] for m in matches]
            match_str = ", ".join(list(set(match_texts))) # Dedupe strings
            
            rows.append({
                "start": p['start'],
                "end": p['end'],
                "text": p_text,
                "status": "✅ TP",
                "match_info": f"Matches: {match_str}"
            })
        else:
            # It is a False Positive
            rows.append({
                "start": p['start'],
                "end": p['end'],
                "text": p_text,
                "status": "❌ FP",
                "match_info": "-"
            })
            
    # 2. Check for False Negatives (GTs that were NOT covered by ANY prediction)
    fn_rows = []
    for g in gt:
        is_covered = False
        for p in preds:
            if calculate_iou(p, g) > IOU_THRESHOLD:
                is_covered = True
                break
        
        if not is_covered:
            g_text = text[g['start']:g['end']].replace('\n', ' ')
            fn_rows.append({
                "start": g['start'],
                "end": g['end'],
                "text": g_text,
                "status": "⚠️ FN",
                "match_info": "MISSED completely"
            })

    # Combine and Sort by Offset
    all_rows = rows + fn_rows
    all_rows.sort(key=lambda x: x['start'])
    
    return all_rows, len(rows), len(fn_rows)

def calculate_metrics(all_rows):
    tp = sum(1 for r in all_rows if r['status'] == "✅ TP")
    fp = sum(1 for r in all_rows if r['status'] == "❌ FP")
    fn = sum(1 for r in all_rows if r['status'] == "⚠️ FN")
    
    # Note: In Coverage logic, one TP pred can cover 2 GT items.
    # Strictly speaking for Recall, we count FOUND GTs.
    # But for this visual table summary, we count Prediction Hits.
    
    # Precision = Useful Preds / Total Preds
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall calculation here is simplified based on row counts.
    # Ideally, use the rigorous calculate_ner_micro_f1 for final stats.
    # Here just for the note summary:
    # Found GTs / Total GTs
    # (Approx: TPs usually align with Found GTs unless one pred covers many)
    
    return tp, fp, fn, prec

def main():
    print(f"🚀 DIAGNOSE NER ASSEMBLY [IoU > {IOU_THRESHOLD}]")
    
    # 1. Load
    try:
        with open(CONFIG_PATH) as f: reg = json.load(f)
        notes, gt_data = load_data()
    except Exception as e: return print(f"[ERROR] {e}")

    # 2. Workers
    workers = load_all_workers(reg)
    if not workers: return

    total_tp, total_fp, total_fn = 0, 0, 0
    start_time = time.time()

    # 3. Process
    for nid, text in notes.items():
        print(f"\n" + "="*100)
        print(f" 📝 NOTE {nid}")
        print("="*100)
        
        # A. Extract (Assembly)
        raw_preds = []
        for w in workers:
            raw_preds.extend(w.extract_entities(text))
        
        # B. Deduplicate (Union)
        preds = deduplicate_predictions(raw_preds)
        
        # C. Classify & Print
        rows, n_preds, n_missed = classify_predictions(preds, gt_data.get(nid, []), text)
        
        # Print Table header
        print(f"{'STATUS':<6} | {'SPAN':<9} | {'PREDICTION (TEXT)':<35} | {'GROUND TRUTH MATCH'}")
        print("-" * 100)
        
        local_tp = 0
        local_fp = 0
        local_fn = 0
        
        for r in rows:
            txt = (r['text'][:33] + '..') if len(r['text']) > 33 else r['text']
            info = (r['match_info'][:45] + '..') if len(r['match_info']) > 45 else r['match_info']
            print(f"{r['status']:<6} | {r['start']}-{r['end']:<5} | {txt:<35} | {info}")
            
            if r['status'] == "✅ TP": local_tp += 1
            if r['status'] == "❌ FP": local_fp += 1
            if r['status'] == "⚠️ FN": local_fn += 1

        total_tp += local_tp
        total_fp += local_fp
        total_fn += local_fn
        
        print("-" * 100)
        print(f"   Summary Note {nid}: TP={local_tp} | FP={local_fp} | FN={local_fn}")

    # 4. Global Stats
    total_time = time.time() - start_time
    
    # Recalculate strict metrics based on GT coverage
    # (A TP prediction might cover 2 GTs, so real Recall is higher than pred count)
    total_gt_items = sum(len(g) for g in gt_data.values())
    # Assuming FN count is accurate (items completely missed)
    found_gt_count = total_gt_items - total_fn
    
    recall = found_gt_count / total_gt_items if total_gt_items > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*60)
    print(f" 🏆 FINAL ASSEMBLY RESULTS")
    print("="*60)
    print(f" Time Taken:   {total_time:.2f}s")
    print("-" * 60)
    print(f" Total Preds:  {total_tp + total_fp}")
    print(f" Total GT:     {total_gt_items}")
    print("-" * 60)
    print(f" TP (Useful):  {total_tp}")
    print(f" FP (Noise):   {total_fp}")
    print(f" FN (Missed):  {total_fn}")
    print("-" * 60)
    print(f" Precision:    {precision:.4f}  (How much output is garbage?)")
    print(f" Recall:       {recall:.4f}  (Did we catch everything?)")
    print(f" F1 Score:     {f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()