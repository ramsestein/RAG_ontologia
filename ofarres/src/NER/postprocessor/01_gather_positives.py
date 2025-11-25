"""
01_gather_positives.py

PURPOSE: Step 1 of RAG Pipeline.
1. Runs NER models in efficient order: Ontology -> SBert -> Acronyms.
2. Creates a UNION of candidates (Recall focused).
3. Classifies them as TP/FP against Ground Truth.
4. Saves output in COMPACT format (one line per entity).
"""

import json
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports ---
from src.utils.metrics import calculate_iou

# --- Constants ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "01_all_positives.json"

IOU_THRESHOLD = 0.25  
WORKER_ORDER = ['OntologyExact', 'SBert', 'Acronyms'] # Efficiency Order

def load_data():
    print(f"[Loader] Notes: {NOTES_PATH}")
    print(f"[Loader] GT:    {GT_PATH}")
    
    if not NOTES_PATH.exists() or not GT_PATH.exists():
        raise FileNotFoundError("Missing JSON files in /data")
    
    with open(NOTES_PATH, 'r', encoding='utf-8') as f:
        notes_data = json.load(f)
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    notes = {item['note_id']: item['text'] for item in notes_data}
    ground_truth = {item['note_id']: item['annotations'] for item in gt_data}
    
    return notes, ground_truth

def load_ordered_workers(registry):
    """Load NER workers in the specific WORKER_ORDER."""
    workers = [] # List of (id, instance) tuples to preserve order
    print(f"\n[Assembly] Loading workers in order: {WORKER_ORDER}...")
    
    for ner_id in WORKER_ORDER:
        if ner_id not in registry:
            print(f"  [!] Warning: {ner_id} not found in registry. Skipping.")
            continue
            
        config = registry[ner_id]
        try:
            mod = importlib.import_module(config['module'])
            cls = getattr(mod, config['class'])
            kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
            instance = cls(**kwargs)
            workers.append((ner_id, instance))
            print(f"  [OK] {ner_id} Loaded")
        except Exception as e:
            print(f"  [ERROR] Error loading {ner_id}: {e}")
    
    return workers

def deduplicate_predictions(preds: List[Dict]) -> List[Dict]:
    """
    Union with Source Tracking: Aggregates all sources that found each span.
    When multiple models find the same span, their sources are combined into a list.
    """
    span_dict = {}  # Key: (start, end), Value: dict with prediction info
    
    for p in preds:
        key = (p['start'], p['end'])
        if key not in span_dict:
            # First occurrence: initialize with source as a list
            span_dict[key] = {
                'start': p['start'],
                'end': p['end'],
                'source': [p.get('source', 'unknown')]
            }
        else:
            # Duplicate span: append the source to the list
            source = p.get('source', 'unknown')
            if source not in span_dict[key]['source']:
                span_dict[key]['source'].append(source)
    
    # Convert dict back to list, preserving order of first occurrence
    unique = list(span_dict.values())
    return unique

def check_coverage(unique_preds: List[Dict], gt_list: List[Dict]) -> int:
    """Count found GTs (Recall)."""
    found_count = 0
    for gt_item in gt_list:
        is_found = False
        for pred in unique_preds:
            if calculate_iou(pred, gt_item) > IOU_THRESHOLD:
                is_found = True
                break
        if is_found:
            found_count += 1
    return found_count

def classify_predictions(unique_preds: List[Dict], gt_list: List[Dict], text: str) -> List[Dict]:
    """Tag predictions as TP/FP."""
    formatted_preds = []
    
    for pred in unique_preds:
        # Check against ALL GTs (Coverage logic)
        is_tp = False
        for g in gt_list:
            if calculate_iou(pred, g) > IOU_THRESHOLD:
                is_tp = True
                break
        
        status = 'TP' if is_tp else 'FP'
        
        # Keep only essential fields, no matched_gt_concept
        entry = {
            "start": pred['start'],
            "end": pred['end'],
            "text": text[pred['start']:pred['end']],
            "source": pred.get('source', ['unknown']),  # Now source is a list
            "status": status
        }
        formatted_preds.append(entry)
        
    formatted_preds.sort(key=lambda x: x['start'])
    return formatted_preds

def save_compact_json(data: List[Dict], filepath: Path):
    """Writes JSON with one annotation per line."""
    print("Saving compact JSON...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("[\n")
        
        for i, note_entry in enumerate(data):
            f.write("  {\n")
            f.write(f'    "note_id": "{note_entry["note_id"]}",\n')
            f.write('    "annotations": [\n')
            
            anns = note_entry['annotations']
            for j, ann in enumerate(anns):
                # Serialize dict to string
                json_str = json.dumps(ann, ensure_ascii=False)
                # Add spaces for readability { "key": ... }
                json_str = json_str.replace('{"', '{ "').replace('"}', '" }').replace('":', '": ').replace(',"', ', "')
                
                comma = "," if j < len(anns) - 1 else ""
                f.write(f"      {json_str}{comma}\n")
                
            f.write("    ]\n")
            
            comma = "," if i < len(data) - 1 else ""
            f.write(f"  }}{comma}\n")
            
        f.write("]\n")

def main():
    print("=" * 80)
    print("[STEP 01] GATHER POSITIVES (OPTIMIZED ORDER)")
    print("=" * 80)
    
    # 1. Load
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        notes, ground_truth = load_data()
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return
    
    # 2. Load Workers (Ordered)
    ordered_workers = load_ordered_workers(registry)
    if not ordered_workers: return
    
    # 3. Process
    final_output = []
    stats = {"total_gt": 0, "found_gt": 0, "tp": 0, "fp": 0}
    
    print("\n" + "=" * 80)
    print("Processing...")
    print("=" * 80)
    
    for note_id, text in notes.items():
        print(f"\n[Note {note_id}]")
        gt_list = ground_truth.get(note_id, [])
        stats["total_gt"] += len(gt_list)
        
        # A. Sequential Extraction
        raw_preds = []
        for worker_id, worker in ordered_workers:
            w_preds = worker.extract_entities(text)
            for p in w_preds: p['source'] = worker_id
            raw_preds.extend(w_preds)
            
        # B. Union with source tracking
        unique_preds = deduplicate_predictions(raw_preds)
        
        # C. Stats
        found = check_coverage(unique_preds, gt_list)
        stats["found_gt"] += found
        
        missing = len(gt_list) - found
        if missing > 0:
            print(f"   [!] Recall: {found}/{len(gt_list)} (Missed {missing})")
        else:
            print(f"   [OK] Recall: {found}/{len(gt_list)} (100%)")

        # D. Classify
        formatted_anns = classify_predictions(unique_preds, gt_list, text)
        
        # Count TP/FP based on prediction classification
        note_tps = sum(1 for p in formatted_anns if p['status'] == 'TP')
        note_fps = sum(1 for p in formatted_anns if p['status'] == 'FP')
        stats["tp"] += note_tps
        stats["fp"] += note_fps
        
        print(f"   Candidates: {len(formatted_anns)} (TP: {note_tps} | FP: {note_fps})")

        final_output.append({
            "note_id": note_id,
            "annotations": formatted_anns
        })

    # 4. Save
    print("\n" + "=" * 80)
    save_compact_json(final_output, OUTPUT_PATH)
    print(f"[OK] Saved to: {OUTPUT_PATH}")
    
    # 5. Summary
    recall = (stats["found_gt"] / stats["total_gt"]) * 100 if stats["total_gt"] else 0
    total_cands = stats["tp"] + stats["fp"]
    prec = (stats["tp"] / total_cands) * 100 if total_cands else 0
    
    print("\n" + "=" * 80)
    print("[FINAL STATS]")
    print("=" * 80)
    print(f"GT Entities:   {stats['total_gt']}")
    print(f"Recall:        {stats['found_gt']} ({recall:.2f}%)")
    print("-" * 80)
    print(f"Total Cands:   {total_cands}")
    print(f"TP (Valid):    {stats['tp']}")
    print(f"FP (Noise):    {stats['fp']}")
    print(f"Precision:     {prec:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()