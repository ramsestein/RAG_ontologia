"""
02_assign_priorities.py

PURPOSE: Step 2 of RAG Pipeline - Assign Priority Tiers.

3-TIER HIERARCHY based on empirical precision data:
┌──────────┬─────────────────────────────────────────────────────────┬───────────────┐
│ Tier     │ Composition                                             │ Est. Precision│
├──────────┼─────────────────────────────────────────────────────────┼───────────────┤
│ 1 (Plat) │ Acronyms (any combo) OR (OntologyExact + SBert)         │ 80% - 95%     │
│ 2 (Gold) │ OntologyExact (single source)                           │ ~40% - 52%    │
│ 3 (Bronz)│ SBert (single source)                                   │ ~18% - 30%    │
└──────────┴─────────────────────────────────────────────────────────┴───────────────┘

RATIONALE:
- Acronyms: ~89% precision when participating. Elite detector.
- OntologyExact + SBert intersection: 62.79% precision (validated by both).
- OntologyExact alone: 40.74% precision. Precise but limited.
- SBert alone: 18.06% precision. "Spam cannon" - high recall, low precision.

INPUT:  data/ner/01_all_positives.json
OUTPUT: data/ner/02_with_priorities.json (same structure + "priority" field)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Set

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "01_all_positives.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "02_with_priorities.json"

# Strategy names (must match source values from 01_gather_positives.py)
ACRONYMS = "Acronyms"
ONTOLOGY = "OntologyExact"
SBERT = "SBert"


def get_priority(sources: List[str]) -> int:
    """
    Assign priority tier based on source combination.
    
    Priority 1 (Platinum) - Highest confidence:
        - Any combination involving Acronyms (solo, with Ontology, with SBert, all three)
        - Intersection of OntologyExact + SBert (validated by both dictionary and semantic)
        
    Priority 2 (Gold) - Medium confidence:
        - OntologyExact alone (precise dictionary match)
        
    Priority 3 (Bronze) - Low confidence:
        - SBert alone (semantic similarity, high recall but noisy)
    
    Args:
        sources: List of strategy names that detected this entity
        
    Returns:
        int: Priority tier (1, 2, or 3). Lower is better.
    """
    source_set: Set[str] = set(sources)
    
    # TIER 1 (Platinum): Acronyms participates OR (Ontology + SBert intersection)
    if ACRONYMS in source_set:
        # Any combo with Acronyms is Platinum
        return 1
    
    if ONTOLOGY in source_set and SBERT in source_set:
        # Both Ontology and SBert agree -> validated, Platinum
        return 1
    
    # TIER 2 (Gold): OntologyExact alone
    if ONTOLOGY in source_set and SBERT not in source_set:
        return 2
    
    # TIER 3 (Bronze): SBert alone (or unknown)
    return 3


def assign_priorities(data: List[Dict]) -> List[Dict]:
    """
    Add priority field to each annotation.
    
    Args:
        data: List of note entries with annotations
        
    Returns:
        List of note entries with priority field added to each annotation
    """
    stats = {1: {"TP": 0, "FP": 0}, 2: {"TP": 0, "FP": 0}, 3: {"TP": 0, "FP": 0}}
    
    for note_entry in data:
        for ann in note_entry["annotations"]:
            sources = ann.get("source", [])
            priority = get_priority(sources)
            ann["priority"] = priority
            
            # Track stats
            status = ann.get("status", "FP")
            stats[priority][status] += 1
    
    return data, stats


def save_compact_json(data: List[Dict], filepath: Path):
    """Writes JSON with one annotation per line (same format as 01_gather_positives)."""
    print(f"[Save] Writing to: {filepath}")
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
                # Add spaces for readability
                json_str = json_str.replace('{"', '{ "').replace('"}', '" }').replace('":', '": ').replace(',"', ', "')
                
                comma = "," if j < len(anns) - 1 else ""
                f.write(f"      {json_str}{comma}\n")
                
            f.write("    ]\n")
            
            comma = "," if i < len(data) - 1 else ""
            f.write(f"  }}{comma}\n")
            
        f.write("]\n")


def print_tier_summary(stats: Dict):
    """Print summary of priority tier distribution."""
    print("\n" + "=" * 70)
    print("PRIORITY TIER DISTRIBUTION")
    print("=" * 70)
    
    tier_names = {1: "Platinum (Acronyms / Onto+SBert)", 
                  2: "Gold (OntologyExact only)", 
                  3: "Bronze (SBert only)"}
    
    total_tp = 0
    total_fp = 0
    
    for tier in [1, 2, 3]:
        tp = stats[tier]["TP"]
        fp = stats[tier]["FP"]
        total = tp + fp
        total_tp += tp
        total_fp += fp
        
        precision = (tp / total * 100) if total > 0 else 0
        
        print(f"\n  Tier {tier} - {tier_names[tier]}:")
        print(f"    Total: {total}")
        print(f"    TP: {tp}, FP: {fp}")
        print(f"    Precision: {precision:.2f}%")
    
    print("\n" + "-" * 70)
    total = total_tp + total_fp
    overall_precision = (total_tp / total * 100) if total > 0 else 0
    print(f"  TOTAL: {total} annotations")
    print(f"  Overall TP: {total_tp}, FP: {total_fp}")
    print(f"  Overall Precision: {overall_precision:.2f}%")
    print("=" * 70)


def main():
    print("=" * 70)
    print("[STEP 02] ASSIGN PRIORITIES (3-TIER SYSTEM)")
    print("=" * 70)
    
    # 1. Load input
    print(f"\n[Load] Reading from: {INPUT_PATH}")
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        print("        Run 01_gather_positives.py first.")
        return
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_notes = len(data)
    total_annotations = sum(len(note["annotations"]) for note in data)
    print(f"[OK] Loaded {total_notes} notes with {total_annotations} annotations")
    
    # 2. Assign priorities
    print("\n[Process] Assigning priority tiers...")
    data_with_priorities, stats = assign_priorities(data)
    
    # 3. Save output
    save_compact_json(data_with_priorities, OUTPUT_PATH)
    print(f"[OK] Saved to: {OUTPUT_PATH}")
    
    # 4. Print summary
    print_tier_summary(stats)
    
    # 5. Show tier logic reminder
    print("\n" + "=" * 70)
    print("TIER LOGIC REFERENCE")
    print("=" * 70)
    print("""
  Tier 1 (Platinum) - KING - Never deleted by lower tier:
    • Acronyms (any combination)
    • OntologyExact + SBert intersection
    
  Tier 2 (Gold) - CITIZEN - Respects Platinum, dominates Bronze:
    • OntologyExact (single source)
    
  Tier 3 (Bronze) - PEASANT - Useful for recall, loses conflicts:
    • SBert (single source)
    
  Use in 03_handle_overlapping_spans.py:
    - If container_tier > nested_tier: EXPLODE (keep nested)
    - If container_tier <= nested_tier: Keep container
""")


if __name__ == "__main__":
    main()
