#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_gather_assembly.py - The Harvester

RESPONSIBILITY: Run all NER workers, collect entities, and merge exact duplicates
                to determine consensus.

LOGIC:
1. Load workers defined in config/ner_registry.json
2. Extract entities from data/notes.json
3. Consensus Merge: If multiple workers find the exact same span (same start/end):
   - Merge them into a single JSON object
   - Transform the source field into a list (e.g., source: ["OntologyExact", "SBert"])

OUTPUT: data/ner/01_raw_assembly.json
"""

import json
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "01_raw_assembly.json"


def load_notes() -> Dict[str, str]:
    """Load notes from notes.json."""
    with open(NOTES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item['note_id']: item['text'] for item in data}


def load_ner_worker(ner_id: str, config: Dict) -> Any:
    """Load a single NER worker from the registry."""
    try:
        mod = importlib.import_module(config['module'])
        cls = getattr(mod, config['class'])
        kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
        return cls(**kwargs)
    except Exception as e:
        print(f"  [ERROR] Failed to load {ner_id}: {e}")
        return None


def merge_duplicates(entities: List[Dict]) -> List[Dict]:
    """
    Merge entities with the exact same span (start, end) into single entries
    with a source list indicating which workers found them.
    """
    # Group by (start, end)
    span_groups = defaultdict(list)
    
    for ent in entities:
        key = (ent['start'], ent['end'])
        span_groups[key].append(ent)
    
    merged = []
    for (start, end), group in span_groups.items():
        # Collect all sources
        sources = []
        for ent in group:
            src = ent.get('source', 'Unknown')
            if isinstance(src, list):
                sources.extend(src)
            else:
                sources.append(src)
        
        # Deduplicate sources while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)
        
        # Use first entity as base, update source
        merged_ent = dict(group[0])
        merged_ent['source'] = unique_sources if len(unique_sources) > 1 else unique_sources[0]
        merged.append(merged_ent)
    
    # Sort by start position
    merged.sort(key=lambda x: x['start'])
    return merged


def run_harvester(verbose: bool = True) -> List[Dict]:
    """
    Main harvester function.
    Returns the merged assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 01: THE HARVESTER (Gather Assembly)")
        print("   Responsibility: Run all NER workers and merge consensus")
        print("=" * 80)
    
    # Load config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    # Load notes
    notes = load_notes()
    if verbose:
        print(f"\n[Harvester] Loaded {len(notes)} notes from {NOTES_PATH}")
    
    # Load all workers
    workers = {}
    if verbose:
        print(f"\n[Harvester] Loading {len(registry)} NER workers...")
    
    for ner_id, config in registry.items():
        worker = load_ner_worker(ner_id, config)
        if worker:
            workers[ner_id] = worker
            if verbose:
                print(f"  [OK] {ner_id} loaded")
    
    if not workers:
        print("[ERROR] No workers loaded. Aborting.")
        return []
    
    # Process each note
    output_data = []
    total_raw = 0
    total_merged = 0
    
    for note_id, text in notes.items():
        if verbose:
            print(f"\n[Harvester] Processing Note {note_id}...")
        
        raw_entities = []
        
        # Extract entities from each worker
        for worker_id, worker in workers.items():
            entities = worker.extract_entities(text)
            
            # Tag each entity with its source
            for ent in entities:
                ent['source'] = worker_id
                # Ensure text field exists
                if 'text' not in ent:
                    ent['text'] = text[ent['start']:ent['end']]
            
            raw_entities.extend(entities)
            if verbose:
                print(f"    {worker_id}: {len(entities)} entities")
        
        total_raw += len(raw_entities)
        
        # Merge duplicates
        merged_entities = merge_duplicates(raw_entities)
        total_merged += len(merged_entities)
        
        if verbose:
            consensus_count = sum(1 for e in merged_entities if isinstance(e.get('source'), list))
            print(f"    -> Merged: {len(raw_entities)} raw -> {len(merged_entities)} unique ({consensus_count} consensus)")
        
        output_data.append({
            "note_id": note_id,
            "annotations": merged_entities
        })
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n[Harvester] Summary:")
        print(f"    Raw entities:    {total_raw}")
        print(f"    Merged entities: {total_merged}")
        print(f"    Output saved to: {OUTPUT_PATH}")
    
    return output_data


def main():
    run_harvester(verbose=True)


if __name__ == "__main__":
    main()
