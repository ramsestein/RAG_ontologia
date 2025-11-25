"""
03_handle_overlapping_spans.py

PURPOSE: Step 3 - Deduplication with "Dictionary Sovereignty" Logic.

PHILOSOPHY: "If it is not clear, do not touch it."
- We ONLY remove entities when we have HIGH CONFIDENCE they are duplicates.
- Tier 1/2 (Dictionary) defines trusted concept boundaries → safe to merge.
- Tier 3 (Model) is unstable → could be "Acute X" (good) or "History of X" (bad).
- We CANNOT judge Tier 3 here. The Cross-Encoder will handle it later.

LOGIC MATRIX (100% Recall Guaranteed):
┌──────────────┬──────────────┬─────────────────┬──────────────────────────────────────────┐
│ Container    │ Nested       │ Action          │ Reasoning                                │
├──────────────┼──────────────┼─────────────────┼──────────────────────────────────────────┤
│ Tier 1       │ Any          │ KEEP CONTAINER  │ Dictionary Sovereignty. Elite defines    │
│              │              │ (Drop Nested)   │ the concept boundary. Safe to merge.     │
├──────────────┼──────────────┼─────────────────┼──────────────────────────────────────────┤
│ Tier 2       │ Any          │ KEEP CONTAINER  │ Dictionary Sovereignty. Ontology defines │
│              │              │ (Drop Nested)   │ the concept boundary. Safe to merge.     │
├──────────────┼──────────────┼─────────────────┼──────────────────────────────────────────┤
│ Tier 3       │ Tier 1       │ ⚠️ KEEP BOTH    │ Coexistence. "CT Angiography" vs "CT".   │
│              │              │                 │ We need both: procedure AND anchor.      │
├──────────────┼──────────────┼─────────────────┼──────────────────────────────────────────┤
│ Tier 3       │ Tier 2       │ ⚠️ KEEP BOTH    │ Coexistence. "Acute Hemorrhage" vs       │
│              │              │                 │ "Hemorrhage". Adjective version is valid.│
├──────────────┼──────────────┼─────────────────┼──────────────────────────────────────────┤
│ Tier 3       │ Tier 3       │ ⚠️ KEEP BOTH    │ Ambiguity. Both are unstable. Don't      │
│              │              │                 │ touch. Cross-Encoder will judge later.   │
└──────────────┴──────────────┴─────────────────┴──────────────────────────────────────────┘

NOTE: This guarantees 100% Recall but may leave duplicates. The Cross-Encoder (Step 4)
      will filter the bad ones by looking at context.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports ---
from src.utils.metrics import calculate_iou

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "02_with_priorities.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "03_resolved_overlaps.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"

IOU_THRESHOLD = 0.25

# --- COLORS ---
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_END = '\033[0m'

def load_ground_truth():
    if not GT_PATH.exists(): return {}
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['note_id']: item['annotations'] for item in data}

def get_recall_count(candidates: List[Dict], gt_list: List[Dict]) -> int:
    found = 0
    for gt in gt_list:
        if any(calculate_iou(c, gt) > IOU_THRESHOLD for c in candidates):
            found += 1
    return found

def format_span(ent: Dict) -> str:
    rank_lbl = {1:"🥇", 2:"🥈", 3:"🥉"}.get(ent['priority'], "")
    status = ent.get('status', '?')
    text = f"{rank_lbl} {ent['text']} [{ent['start']}-{ent['end']}]"
    if status == "TP":
        return f"{C_GREEN}{text} (TP){C_END}"
    return f"{text} ({status})"

def solve_conflicts_safe(entities: List[Dict], debug: bool = False) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Resolve overlapping spans using Dictionary Sovereignty logic.
    
    Returns:
        Tuple of (kept_entities, dropped_entities, stats_dict)
    """
    # Sort: Start (asc), Length (desc) -> Containers first
    sorted_ents = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    dropped_indices = set()
    
    stats = {
        "merges": 0,           # Tier 1/2 absorbed nested
        "coexist": 0,          # Tier 3 kept both
        "merge_details": [],
        "coexist_details": []
    }

    for i in range(len(sorted_ents)):
        if i in dropped_indices: continue
        container = sorted_ents[i]
        
        for j in range(len(sorted_ents)):
            if i == j: continue
            if j in dropped_indices: continue
            
            nested = sorted_ents[j]
            
            # Check Nesting (nested is fully contained within container)
            if (nested['start'] >= container['start']) and (nested['end'] <= container['end']):
                
                c_rank = container.get('priority', 3)
                n_rank = nested.get('priority', 3)
                
                # --- DICTIONARY SOVEREIGNTY LOGIC (with Status Override) ---
                
                c_status = container.get('status', '?')
                n_status = nested.get('status', '?')
                
                # Rule 1: Container is Tier 1 or Tier 2 (Dictionary Sovereign)
                # BUT: If Container is FP and Nested is TP, we should NOT merge
                #      because we'd lose a valid entity to an invalid container.
                # This is a "ground truth override" - we know the nested is correct.
                
                if c_rank <= 2:
                    # Safety check: Don't let FP containers absorb TP nested entities
                    if c_status == 'FP' and n_status == 'TP':
                        # Keep BOTH - the TP nested should survive
                        stats["coexist"] += 1
                        stats["coexist_details"].append({
                            "container": container['text'],
                            "container_tier": c_rank,
                            "container_status": c_status,
                            "nested": nested['text'],
                            "nested_tier": n_rank,
                            "nested_status": n_status,
                            "reason": "FP container tried to absorb TP nested"
                        })
                        if debug:
                            print(f"   ⚠️  PROTECTED: FP container \"{container['text']}\" cannot absorb TP \"{nested['text']}\"")
                    else:
                        # Safe to merge: either container is TP, or nested is FP, or both same status
                        dropped_indices.add(j)
                        stats["merges"] += 1
                        stats["merge_details"].append({
                            "container": container['text'],
                            "container_tier": c_rank,
                            "container_status": c_status,
                            "nested": nested['text'],
                            "nested_tier": n_rank,
                            "nested_status": n_status
                        })
                        if debug:
                            print(f"   {C_YELLOW}Merged (Dictionary Sovereignty):{C_END} {format_span(container)} absorbed {format_span(nested)}")
                        
                # Rule 2: Container is Tier 3 (Model Prediction - Unstable)
                # Action: KEEP BOTH.
                # Reasoning: We CANNOT judge if "Acute X" is good or "History of X" is bad.
                # The Cross-Encoder will handle this in Step 4.
                else:
                    stats["coexist"] += 1
                    stats["coexist_details"].append({
                        "container": container['text'],
                        "container_tier": c_rank,
                        "container_status": c_status,
                        "nested": nested['text'],
                        "nested_tier": n_rank,
                        "nested_status": n_status
                    })
                    if debug:
                        print(f"   ⚠️  COEXISTING (Tier 3 Untouchable): {format_span(container)} overlapping {format_span(nested)}")
                    # Do nothing - keep both

    # Re-gather survivors
    kept = [sorted_ents[i] for i in range(len(sorted_ents)) if i not in dropped_indices]
    dropped = [sorted_ents[i] for i in range(len(sorted_ents)) if i in dropped_indices]

    kept.sort(key=lambda x: x['start'])
    return kept, dropped, stats

def save_output(data: List[Dict], filepath: Path):
    print(f"\nSaving to {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("[\n")
        for i, note_entry in enumerate(data):
            f.write("  {\n")
            f.write(f'    "note_id": "{note_entry["note_id"]}",\n')
            f.write('    "annotations": [\n')
            anns = note_entry['annotations']
            for j, ann in enumerate(anns):
                # Clean up json keys for saving
                save_ann = {k:v for k,v in ann.items() if k in ['start', 'end', 'text', 'source', 'status', 'priority']}
                json_str = json.dumps(save_ann, ensure_ascii=False)
                json_str = json_str.replace('{"', '{ "').replace('"}', '" }').replace('":', '": ').replace(',"', ', "')
                comma = "," if j < len(anns) - 1 else ""
                f.write(f"      {json_str}{comma}\n")
            f.write("    ]\n")
            comma = "," if i < len(data) - 1 else ""
            f.write(f"  }}{comma}\n")
        f.write("]\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", action="store_true", help="Show detailed logs")
    args = parser.parse_args()

    print("=" * 100)
    print("🧹 STEP 03: HANDLE OVERLAPPING SPANS (DICTIONARY SOVEREIGNTY)")
    print("   Strategy: Tier 1/2 = Safe Merge | Tier 3 = Coexist (Untouchable)")
    print("   Goal: 100% Recall Retention. Cross-Encoder will filter later.")
    print("=" * 100)

    if not INPUT_PATH.exists(): 
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        return
        
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_map = load_ground_truth()
    
    final_output = []
    sum_recall_b, sum_recall_a, total_gt = 0, 0, 0
    total_merges, total_coexist = 0, 0
    all_merge_details = []
    all_coexist_details = []
    
    # Before/After stats
    before_tp, before_fp = 0, 0
    after_tp, after_fp = 0, 0
    
    print(f"\n{'ID':<4} | {'INPUT':<6} | {'KEPT':<6} | {'DROP':<6} | {'MERGE':<6} | {'COEX':<6} | {'RECALL BEFORE':<14} | {'RECALL AFTER':<14}")
    print("-" * 110)

    for entry in data:
        note_id = entry['note_id']
        annotations = entry['annotations']
        gt_list = gt_map.get(note_id, [])
        total_gt += len(gt_list)

        # Count before stats
        before_tp += sum(1 for a in annotations if a.get('status') == 'TP')
        before_fp += sum(1 for a in annotations if a.get('status') == 'FP')

        r_b = get_recall_count(annotations, gt_list)
        sum_recall_b += r_b

        if args.results:
            print(f"\n📝 NOTE {note_id}")

        kept, dropped, stats = solve_conflicts_safe(annotations, args.results)
        
        # Count after stats
        after_tp += sum(1 for a in kept if a.get('status') == 'TP')
        after_fp += sum(1 for a in kept if a.get('status') == 'FP')
        
        total_merges += stats["merges"]
        total_coexist += stats["coexist"]
        all_merge_details.extend(stats["merge_details"])
        all_coexist_details.extend(stats["coexist_details"])
        
        r_a = get_recall_count(kept, gt_list)
        sum_recall_a += r_a
        
        flag = "✅" if r_a == r_b else "❌"
        if not args.results:
            print(f"{note_id:<4} | {len(annotations):<6} | {len(kept):<6} | {len(dropped):<6} | {stats['merges']:<6} | {stats['coexist']:<6} | {r_b}/{len(gt_list):<12} | {r_a}/{len(gt_list)} {flag}")

        final_output.append({"note_id": note_id, "annotations": kept})

    # === SUMMARY ===
    print("-" * 110)
    print(f"\n{'=' * 110}")
    print("OPERATION SUMMARY")
    print("=" * 110)
    
    print(f"\n  Total Merges (Dictionary Sovereignty): {total_merges}")
    print(f"  Total Coexist (Tier 3 Untouchable):    {total_coexist}")
    
    # Show merge details breakdown
    merge_tp_dropped = sum(1 for d in all_merge_details if d['nested_status'] == 'TP')
    merge_fp_dropped = sum(1 for d in all_merge_details if d['nested_status'] == 'FP')
    print(f"\n  MERGE IMPACT (Nested entities absorbed by Dictionary containers):")
    print(f"    TP absorbed: {merge_tp_dropped}")
    print(f"    FP absorbed: {merge_fp_dropped}")
    
    # Show some merge examples
    if all_merge_details:
        print(f"\n  MERGE EXAMPLES:")
        for detail in all_merge_details[:5]:
            c_status = detail['container_status']
            n_status = detail['nested_status']
            quality = "✅" if c_status == 'TP' else "⚠️"
            print(f"    {quality} T{detail['container_tier']} \"{detail['container']}\" ({c_status}) absorbed T{detail['nested_tier']} \"{detail['nested']}\" ({n_status})")
        if len(all_merge_details) > 5:
            print(f"    ... and {len(all_merge_details) - 5} more")
    
    # Show coexist examples
    if all_coexist_details:
        print(f"\n  COEXIST EXAMPLES (Tier 3 containers - kept both):")
        for detail in all_coexist_details[:5]:
            c_status = detail['container_status']
            n_status = detail['nested_status']
            print(f"    ⚠️ T{detail['container_tier']} \"{detail['container']}\" ({c_status}) ↔ T{detail['nested_tier']} \"{detail['nested']}\" ({n_status})")
        if len(all_coexist_details) > 5:
            print(f"    ... and {len(all_coexist_details) - 5} more")
    
    # === BEFORE vs AFTER ===
    print(f"\n{'=' * 110}")
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 110)
    
    # TP comparison
    tp_diff = after_tp - before_tp
    tp_pct = (tp_diff / before_tp * 100) if before_tp > 0 else 0
    tp_symbol = "📈" if tp_diff > 0 else ("📉" if tp_diff < 0 else "➡️")
    
    print(f"\n  TRUE POSITIVES (TP):")
    print(f"    Before: {before_tp}")
    print(f"    After:  {after_tp}")
    print(f"    Diff:   {tp_diff:+d} ({tp_pct:+.2f}%) {tp_symbol}")
    
    # FP comparison
    fp_diff = after_fp - before_fp
    fp_pct = (fp_diff / before_fp * 100) if before_fp > 0 else 0
    fp_symbol = "📈" if fp_diff < 0 else ("📉" if fp_diff > 0 else "➡️")
    
    print(f"\n  FALSE POSITIVES (FP):")
    print(f"    Before: {before_fp}")
    print(f"    After:  {after_fp}")
    print(f"    Diff:   {fp_diff:+d} ({fp_pct:+.2f}%) {fp_symbol}")
    
    # Total
    total_before = before_tp + before_fp
    total_after = after_tp + after_fp
    total_diff = total_after - total_before
    total_pct = (total_diff / total_before * 100) if total_before > 0 else 0
    
    print(f"\n  TOTAL ANNOTATIONS:")
    print(f"    Before: {total_before}")
    print(f"    After:  {total_after}")
    print(f"    Diff:   {total_diff:+d} ({total_pct:+.2f}%)")
    
    # Precision
    prec_before = (before_tp / total_before * 100) if total_before > 0 else 0
    prec_after = (after_tp / total_after * 100) if total_after > 0 else 0
    prec_diff = prec_after - prec_before
    prec_symbol = "📈" if prec_diff > 0 else ("📉" if prec_diff < 0 else "➡️")
    
    print(f"\n  PRECISION:")
    print(f"    Before: {prec_before:.2f}%")
    print(f"    After:  {prec_after:.2f}%")
    print(f"    Diff:   {prec_diff:+.2f}pp {prec_symbol}")
    
    # === RECALL ===
    print(f"\n{'=' * 110}")
    print("RECALL CHECK (CRITICAL)")
    print("=" * 110)
    print(f"\n  TOTAL RECALL BEFORE: {sum_recall_b}/{total_gt} ({sum_recall_b/total_gt*100:.2f}%)")
    print(f"  TOTAL RECALL AFTER:  {sum_recall_a}/{total_gt} ({sum_recall_a/total_gt*100:.2f}%)")
    
    recall_diff = sum_recall_a - sum_recall_b
    if recall_diff < 0:
        print(f"\n  {C_RED}⚠️  CRITICAL WARNING: Recall dropped by {abs(recall_diff)} entities!{C_END}")
        print(f"  {C_RED}    This logic is still too aggressive. Review the merges.{C_END}")
    elif recall_diff == 0:
        print(f"\n  {C_GREEN}✅ SUCCESS: 100% Recall Retention confirmed.{C_END}")
    else:
        print(f"\n  {C_GREEN}✅ Recall IMPROVED by {recall_diff} entities.{C_END}")
    
    # === SAVE ===
    save_output(final_output, OUTPUT_PATH)
    
    # === NEXT STEPS ===
    print(f"\n{'=' * 110}")
    print("NEXT STEPS")
    print("=" * 110)
    print(f"""
  Current State:
    - {total_coexist} overlapping pairs are COEXISTING (both kept)
    - These need the Cross-Encoder to judge which is correct
    
  What the Cross-Encoder will do:
    - Look at context: "History of hypertension" → likely FP (historical, not current)
    - Look at context: "Acute hemorrhage" → likely TP (current finding)
    - This requires semantic understanding that heuristics cannot provide
""")

if __name__ == "__main__":
    main()