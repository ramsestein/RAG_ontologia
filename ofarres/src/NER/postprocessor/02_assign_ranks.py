#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_assign_ranks.py - The Classifier

RESPONSIBILITY: Analyze the source list from Step 01 and assign a Confidence Tier
                (priority) to each entity.

LOGIC:
- Tier 1 (Platinum/Elite):
    * Any entity found by Acronyms (regardless of overlaps)
    * OR Any entity found by Consensus (Source list contains BOTH "OntologyExact" AND "SBert")
    
- Tier 2 (Gold/Standard):
    * Entities found by OntologyExact only
    
- Tier 3 (Bronze/Weak):
    * Entities found by SBert only

OUTPUT: data/ner/02_ranked.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "01_raw_assembly.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "02_ranked.json"


def get_sources_as_list(source) -> List[str]:
    """Convert source field to list format."""
    if isinstance(source, list):
        return source
    elif isinstance(source, str):
        return [source]
    else:
        return []


def assign_priority(entity: Dict) -> int:
    """
    Assign a confidence tier (priority) based on the source(s).
    
    Tier 1 (Platinum/Elite): Acronyms OR Consensus (OntologyExact + SBert)
    Tier 2 (Gold/Standard): OntologyExact only
    Tier 3 (Bronze/Weak): SBert only
    """
    sources = get_sources_as_list(entity.get('source', []))
    sources_set = set(sources)
    
    # Tier 1: Acronyms (highest priority - medical abbreviations are critical)
    if 'Acronyms' in sources_set:
        return 1
    
    # Tier 1: Consensus (both OntologyExact and SBert agree)
    if 'OntologyExact' in sources_set and 'SBert' in sources_set:
        return 1
    
    # Tier 2: OntologyExact only (dictionary-based, high confidence)
    if 'OntologyExact' in sources_set:
        return 2
    
    # Tier 3: SBert only (model prediction, lower confidence)
    if 'SBert' in sources_set:
        return 3
    
    # Default fallback (unknown source)
    return 3


def get_tier_label(priority: int) -> str:
    """Get human-readable tier label."""
    labels = {
        1: "Platinum/Elite",
        2: "Gold/Standard", 
        3: "Bronze/Weak"
    }
    return labels.get(priority, "Unknown")


def run_classifier(verbose: bool = True) -> List[Dict]:
    """
    Main classifier function.
    Returns the ranked assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 02: THE CLASSIFIER (Assign Ranks)")
        print("   Responsibility: Assign Confidence Tiers based on source consensus")
        print("=" * 80)
    
    # Load input
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        print("[INFO] Run 01_gather_assembly.py first.")
        return []
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"\n[Classifier] Loaded {len(data)} notes from {INPUT_PATH}")
    
    # Process each note
    output_data = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    consensus_count = 0
    acronym_count = 0
    
    for note_entry in data:
        note_id = note_entry['note_id']
        annotations = note_entry['annotations']
        
        ranked_annotations = []
        
        for ann in annotations:
            # Assign priority
            priority = assign_priority(ann)
            
            # Create ranked annotation
            ranked_ann = dict(ann)
            ranked_ann['priority'] = priority
            ranked_annotations.append(ranked_ann)
            
            # Track stats
            tier_counts[priority] = tier_counts.get(priority, 0) + 1
            
            sources = get_sources_as_list(ann.get('source', []))
            if 'Acronyms' in sources:
                acronym_count += 1
            if 'OntologyExact' in sources and 'SBert' in sources:
                consensus_count += 1
        
        output_data.append({
            "note_id": note_id,
            "annotations": ranked_annotations
        })
        
        if verbose:
            t1 = sum(1 for a in ranked_annotations if a['priority'] == 1)
            t2 = sum(1 for a in ranked_annotations if a['priority'] == 2)
            t3 = sum(1 for a in ranked_annotations if a['priority'] == 3)
            print(f"    Note {note_id}: T1={t1} | T2={t2} | T3={t3}")
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        total = sum(tier_counts.values())
        print(f"\n[Classifier] Summary:")
        print(f"    Total entities:   {total}")
        print(f"    Tier 1 (Elite):   {tier_counts[1]} ({tier_counts[1]/total*100:.1f}%)")
        print(f"      - Acronyms:     {acronym_count}")
        print(f"      - Consensus:    {consensus_count}")
        print(f"    Tier 2 (Gold):    {tier_counts[2]} ({tier_counts[2]/total*100:.1f}%)")
        print(f"    Tier 3 (Bronze):  {tier_counts[3]} ({tier_counts[3]/total*100:.1f}%)")
        print(f"    Output saved to:  {OUTPUT_PATH}")
    
    return output_data


def main():
    run_classifier(verbose=True)


if __name__ == "__main__":
    main()
