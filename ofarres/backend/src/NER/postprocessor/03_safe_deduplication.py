#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_safe_deduplication.py - Safe Deduplication

RESPONSIBILITY: Resolve overlapping entities ("Russian Doll" problem) without
                sacrificing recall using the "Dictionary Sovereign + Coexistence" strategy.

LOGIC:
1. Sort entities by Start Position (asc), then Length (desc) to handle "Containers" first
2. Apply Conflict Resolution Matrix:

   If Container is Tier 1 or 2 (Dictionary/Consensus):
     - Action: ABSORB - Keep the Container, Drop the Nested entity
     - Reasoning: Dictionary definition allows us to trust the boundary
                  (e.g., "Middle Cerebral Artery" absorbs "Artery")
   
   If Container is Tier 3 (SBert/Model):
     - Action: COEXIST - Keep BOTH the Container and the Nested entity
     - Reasoning: Cannot algorithmically distinguish valid context ("Acute Hemorrhage")
                  from noise ("History of Hemorrhage"). Preserve both for Cross-Encoder.

INPUT: data/ner/02_ranked.json
OUTPUT: data/ner/03_deduplicated.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "02_ranked.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "03_deduplicated.json"


def is_nested(container: Dict, nested: Dict) -> bool:
    """
    Check if 'nested' is fully contained within 'container'.
    Returns True if nested is inside container (not equal spans).
    """
    # Nested must be strictly inside container
    return (nested['start'] >= container['start'] and 
            nested['end'] <= container['end'] and
            not (nested['start'] == container['start'] and nested['end'] == container['end']))


def resolve_conflicts(entities: List[Dict], verbose: bool = False) -> Tuple[List[Dict], Dict]:
    """
    Resolve overlapping entities using Dictionary Sovereign + Coexistence strategy.
    
    Returns:
        Tuple of (kept_entities, stats_dict)
    """
    if not entities:
        return [], {"absorbed": 0, "coexist": 0}
    
    # Sort: Start (asc), Length (desc) -> Containers first
    sorted_ents = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    dropped_indices: Set[int] = set()
    stats = {
        "absorbed": 0,
        "coexist": 0,
        "absorb_details": [],
        "coexist_details": []
    }
    
    for i in range(len(sorted_ents)):
        if i in dropped_indices:
            continue
        
        container = sorted_ents[i]
        container_tier = container.get('priority', 3)
        
        for j in range(len(sorted_ents)):
            if i == j or j in dropped_indices:
                continue
            
            nested = sorted_ents[j]
            
            # Check if nested is fully contained within container
            if not is_nested(container, nested):
                continue
            
            nested_tier = nested.get('priority', 3)
            
            # === CONFLICT RESOLUTION MATRIX ===
            
# === CONFLICT RESOLUTION MATRIX ===
            
            if container_tier <= 2:
                # Container is Tier 1 or 2 (Dictionary/Consensus)
                
                # SUB-RULE: PROTECTION
                # If Nested is Tier 1 (Elite) and Container is Tier 2 (Standard),
                # we do NOT absorb. We Explode (keep both).
                # Example: "Alberta...Score" (T2) vs "Stroke" (T1-Acronym).
                if nested_tier < container_tier:
                     stats["coexist"] += 1 # Keep both
                     if verbose:
                        stats["coexist_details"].append({
                            "container": container.get('text', ''),
                            "nested": nested.get('text', ''),
                            "reason": "Rank Protection (T2 cannot eat T1)"
                        })
                else:
                     # Standard Absorption (T1 eats T1/T2/T3, T2 eats T2/T3)
                     # The Dictionary boundary is trusted.
                     dropped_indices.add(j)
                     stats["absorbed"] += 1
                     
                     if verbose:
                        stats["absorb_details"].append({
                            "container": container.get('text', ''),
                            "nested": nested.get('text', '')
                        })

            else:
                # Container is Tier 3 (SBert/Model)
                # Action: COEXIST - Keep both
                stats["coexist"] += 1
                
                if verbose:
                    stats["coexist_details"].append({
                        "container": container.get('text', ''),
                        "nested": nested.get('text', ''),
                        "reason": "Tier 3 instability"
                    })

    
    # Gather survivors
    kept = [sorted_ents[i] for i in range(len(sorted_ents)) if i not in dropped_indices]
    kept.sort(key=lambda x: x['start'])
    
    return kept, stats


def run_safe_deduplication(verbose: bool = True) -> List[Dict]:
    """
    Main safe deduplication function.
    Returns the deduplicated assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 03: SAFE DEDUPLICATION (Dictionary Sovereign + Coexistence)")
        print("   Responsibility: Resolve overlaps without sacrificing recall")
        print("=" * 80)
    
    # Load input
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        print("[INFO] Run 02_assign_ranks.py first.")
        return []
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"\n[Deduplication] Loaded {len(data)} notes from {INPUT_PATH}")
    
    # Process each note
    output_data = []
    total_before = 0
    total_after = 0
    total_absorbed = 0
    total_coexist = 0
    
    for note_entry in data:
        note_id = note_entry['note_id']
        annotations = note_entry['annotations']
        
        total_before += len(annotations)
        
        # Resolve conflicts
        deduplicated, stats = resolve_conflicts(annotations, verbose=verbose)
        
        total_after += len(deduplicated)
        total_absorbed += stats["absorbed"]
        total_coexist += stats["coexist"]
        
        if verbose:
            print(f"    Note {note_id}: {len(annotations)} -> {len(deduplicated)} "
                  f"(absorbed: {stats['absorbed']}, coexist: {stats['coexist']})")
        
        output_data.append({
            "note_id": note_id,
            "annotations": deduplicated
        })
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        reduction = total_before - total_after
        reduction_pct = (reduction / total_before * 100) if total_before > 0 else 0
        
        print(f"\n[Deduplication] Summary:")
        print(f"    Entities before: {total_before}")
        print(f"    Entities after:  {total_after}")
        print(f"    Reduction:       {reduction} ({reduction_pct:.1f}%)")
        print(f"    Absorbed:        {total_absorbed} (Tier 1/2 containers ate nested)")
        print(f"    Coexisting:      {total_coexist} (Tier 3 containers kept both)")
        print(f"    Output saved to: {OUTPUT_PATH}")
    
    return output_data


def main():
    run_safe_deduplication(verbose=True)


if __name__ == "__main__":
    main()
