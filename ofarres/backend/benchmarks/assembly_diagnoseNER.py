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
MIN_IOU_OVERLAP = 0.1  # Minimum IoU for physical overlap (Text Containment logic)


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
    
    # Context Expansion: GT['text'] is substring of Pred['text']
    # O Partial Match: Pred['text'] is substring of GT['text']
    return gt_norm in pred_norm or pred_norm in gt_norm

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
    Classifies each prediction as TP or FP based on Text Containment + IoU Logic.
    
    NEW MATCHING CRITERIA (RAG-Ready Recall):
    1. Condition A: IoU > 0.1 (physical overlap)
    2. Condition B: Text containment (GT in Pred OR Pred in GT)
    3. Constraint: 1-to-1 matching (protects against "Bad Merge")
    
    Returns a list of dicts ready for printing.
    """
    rows = []
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    # Ensure predictions have text field
    preds_with_text = []
    for p in preds:
        p_copy = dict(p)
        if 'text' not in p_copy:
            p_copy['text'] = text[p_copy['start']:p_copy['end']]
        preds_with_text.append(p_copy)
    
    # 1. Match predictions to GT using 1-to-1 Text Containment + IoU logic
    for g_idx, g in enumerate(gt):
        gt_text = g.get('text', text[g['start']:g['end']])
        best_pred_idx = None
        best_iou = -1.0
        
        for p_idx, p in enumerate(preds_with_text):
            if p_idx in matched_pred_indices:
                continue
            
            iou = calculate_iou(p, g)
            
            # Condition A: Physical overlap
            if iou <= MIN_IOU_OVERLAP:
                continue
            
            pred_text = p.get('text', '')
            
            # Condition B: Text containment
            if not text_containment_match(pred_text, gt_text):
                continue
            
            # Valid match - track best by IoU
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = p_idx
        
        if best_pred_idx is not None:
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(best_pred_idx)
    
    # 2. Build rows for display
    for p_idx, p in enumerate(preds_with_text):
        p_text = p.get('text', '').replace('\n', ' ')
        
        if p_idx in matched_pred_indices:
            # Find which GT it matched
            for g_idx, g in enumerate(gt):
                if g_idx in matched_gt_indices:
                    gt_text = g.get('text', text[g['start']:g['end']])
                    iou = calculate_iou(p, g)
                    if iou > MIN_IOU_OVERLAP and text_containment_match(p_text, gt_text):
                        rows.append({
                            "start": p['start'],
                            "end": p['end'],
                            "text": p_text,
                            "status": "✅ TP",
                            "match_info": f"Matches: {gt_text}"
                        })
                        break
        else:
            # False Positive
            rows.append({
                "start": p['start'],
                "end": p['end'],
                "text": p_text,
                "status": "❌ FP",
                "match_info": "-"
            })
    
    # 3. Check for False Negatives (GTs not matched)
    fn_rows = []
    for g_idx, g in enumerate(gt):
        if g_idx not in matched_gt_indices:
            g_text = g.get('text', text[g['start']:g['end']]).replace('\n', ' ')
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
    
    n_tp = len(matched_pred_indices)
    n_fp = len(preds_with_text) - n_tp
    n_fn = len(gt) - len(matched_gt_indices)
    
    return all_rows, n_tp, n_fn

def calculate_metrics(all_rows):
    tp = sum(1 for r in all_rows if r['status'] == "✅ TP")
    fp = sum(1 for r in all_rows if r['status'] == "❌ FP")
    fn = sum(1 for r in all_rows if r['status'] == "⚠️ FN")
    
    # Precision = Useful Preds / Total Preds
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall = Found GTs / Total GTs (using Text Containment + IoU logic)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return tp, fp, fn, prec, recall

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