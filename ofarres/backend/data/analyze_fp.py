#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_fp.py - Analyze False Positives from NER Pipeline

This script compares the pipeline output against ground truth to identify:
1. True Positives (correctly identified)
2. False Positives (extra entities not in GT)
3. False Negatives (missed GT entities)

And categorizes FPs by type:
- Lexical FP: Different text span but same concept (e.g., "diabetes mellitus" vs "diabetes")
- Semantic FP: Valid medical terms not annotated in GT (GT coverage gap)
- Noise FP: Non-medical terms that should have been filtered

Usage: python analyze_fp.py
"""

import json
from pathlib import Path
from collections import defaultdict

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GT_PATH = DATA_DIR / "ground_truth.json"
PRED_PATH = DATA_DIR / "ner" / "05_semantically_clean.json"


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def load_data():
    """Load ground truth and predictions."""
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(PRED_PATH, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    return gt_data, pred_data


def extract_entities(annotations):
    """Extract set of normalized entity texts from annotations."""
    return {normalize_text(ann.get('text', '')) for ann in annotations}


def analyze_note(note_id: str, gt_annotations: list, pred_annotations: list):
    """
    Analyze a single note comparing GT vs Predictions.
    
    Returns:
        dict with TP, FP, FN and details
    """
    gt_texts = extract_entities(gt_annotations)
    pred_texts = extract_entities(pred_annotations)
    
    # Calculate overlaps
    tp_texts = gt_texts & pred_texts
    fn_texts = gt_texts - pred_texts
    fp_texts = pred_texts - gt_texts
    
    # Get detailed info for FPs
    fp_details = []
    for ann in pred_annotations:
        text_norm = normalize_text(ann.get('text', ''))
        if text_norm in fp_texts:
            fp_details.append({
                'text': ann.get('text', ''),
                'text_norm': text_norm,
                'tier': ann.get('priority', 3),
                'source': ann.get('source', 'unknown'),
                'label': ann.get('label', 'unknown'),
                'concept_id': ann.get('concept_id', None),
            })
    
    # Get detailed info for TPs
    tp_details = []
    for ann in pred_annotations:
        text_norm = normalize_text(ann.get('text', ''))
        if text_norm in tp_texts:
            tp_details.append({
                'text': ann.get('text', ''),
                'tier': ann.get('priority', 3),
                'source': ann.get('source', 'unknown'),
                'concept_id': ann.get('concept_id', None),
            })
    
    # Get detailed info for FNs
    fn_details = []
    for ann in gt_annotations:
        text_norm = normalize_text(ann.get('text', ''))
        if text_norm in fn_texts:
            fn_details.append({
                'text': ann.get('text', ''),
                'concept_id': ann.get('concept_id', None),
            })
    
    return {
        'note_id': note_id,
        'gt_count': len(gt_texts),
        'pred_count': len(pred_texts),
        'tp_count': len(tp_texts),
        'fp_count': len(fp_texts),
        'fn_count': len(fn_texts),
        'tp_texts': sorted(tp_texts),
        'fp_texts': sorted(fp_texts),
        'fn_texts': sorted(fn_texts),
        'fp_details': fp_details,
        'tp_details': tp_details,
        'fn_details': fn_details,
    }


def categorize_fp(fp_details: list, gt_texts: set) -> dict:
    """
    Categorize false positives into types.
    
    Categories:
    - has_concept_id: FP has a SNOMED concept ID (likely valid medical term)
    - ontology_match: From ontology lookup (Tier 1 or 2)
    - spacy_only: From SpaCy NER (Tier 3)
    - tier_breakdown: By tier
    """
    categories = {
        'has_concept_id': [],
        'no_concept_id': [],
        'tier_1': [],
        'tier_2': [],
        'tier_3': [],
        'from_ontology': [],
        'from_spacy': [],
    }
    
    for fp in fp_details:
        text = fp['text']
        tier = fp['tier']
        source = fp['source']
        concept_id = fp['concept_id']
        
        # By concept ID
        if concept_id:
            categories['has_concept_id'].append(fp)
        else:
            categories['no_concept_id'].append(fp)
        
        # By tier
        if tier == 1:
            categories['tier_1'].append(fp)
        elif tier == 2:
            categories['tier_2'].append(fp)
        else:
            categories['tier_3'].append(fp)
        
        # By source
        if 'Ontology' in str(source) or 'ONTOLOGY' in str(fp.get('label', '')):
            categories['from_ontology'].append(fp)
        if 'SBert' in str(source) or 'spacy' in str(source).lower():
            categories['from_spacy'].append(fp)
    
    return categories


def main():
    print("=" * 100)
    print(" FALSE POSITIVE ANALYSIS - Pipeline vs Ground Truth")
    print("=" * 100)
    
    gt_data, pred_data = load_data()
    
    # Build lookup by note_id
    gt_by_note = {str(note['note_id']): note['annotations'] for note in gt_data}
    pred_by_note = {str(note['note_id']): note['annotations'] for note in pred_data}
    
    # Aggregate stats
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_fp_details = []
    all_fn_details = []
    all_tp_texts = set()
    all_gt_texts = set()
    
    print(f"\nLoaded {len(gt_data)} notes from Ground Truth")
    print(f"Loaded {len(pred_data)} notes from Predictions")
    
    # Analyze each note
    print("\n" + "-" * 100)
    print(" PER-NOTE ANALYSIS")
    print("-" * 100)
    
    for note_id in sorted(gt_by_note.keys(), key=lambda x: int(x)):
        gt_ann = gt_by_note.get(note_id, [])
        pred_ann = pred_by_note.get(note_id, [])
        
        result = analyze_note(note_id, gt_ann, pred_ann)
        
        total_tp += result['tp_count']
        total_fp += result['fp_count']
        total_fn += result['fn_count']
        all_fp_details.extend(result['fp_details'])
        all_fn_details.extend(result['fn_details'])
        all_tp_texts.update(result['tp_texts'])
        all_gt_texts.update(extract_entities(gt_ann))
        
        print(f"\n📋 Note {note_id}:")
        print(f"   GT: {result['gt_count']} | Pred: {result['pred_count']} | TP: {result['tp_count']} | FP: {result['fp_count']} | FN: {result['fn_count']}")
        
        if result['tp_texts']:
            print(f"   ✅ TRUE POSITIVES ({len(result['tp_texts'])}):")
            for t in result['tp_texts']:
                print(f"      • {t}")
        
        if result['fn_texts']:
            print(f"   ❌ FALSE NEGATIVES (missed) ({len(result['fn_texts'])}):")
            for fn in result['fn_details']:
                print(f"      • \"{fn['text']}\" (concept: {fn['concept_id']})")
        
        if result['fp_texts']:
            print(f"   ⚠️  FALSE POSITIVES (extra) ({len(result['fp_texts'])}):")
            for fp in result['fp_details']:
                tier_label = f"T{fp['tier']}"
                concept = f"concept:{fp['concept_id']}" if fp['concept_id'] else "no-concept"
                source = fp['source']
                print(f"      • \"{fp['text']}\" [{tier_label}] ({concept}) src={source}")
    
    # Summary
    print("\n" + "=" * 100)
    print(" AGGREGATE SUMMARY")
    print("=" * 100)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 Overall Metrics:")
    print(f"   True Positives:  {total_tp}")
    print(f"   False Positives: {total_fp}")
    print(f"   False Negatives: {total_fn}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.4f}")
    
    # Categorize FPs
    print("\n" + "-" * 100)
    print(" FALSE POSITIVE BREAKDOWN")
    print("-" * 100)
    
    categories = categorize_fp(all_fp_details, all_gt_texts)
    
    print(f"\n📁 By Tier:")
    print(f"   Tier 1 (Elite):    {len(categories['tier_1'])} FPs")
    print(f"   Tier 2 (Ontology): {len(categories['tier_2'])} FPs")
    print(f"   Tier 3 (SpaCy):    {len(categories['tier_3'])} FPs")
    
    print(f"\n📁 By Concept ID:")
    print(f"   Has SNOMED Concept ID: {len(categories['has_concept_id'])} (likely valid medical terms)")
    print(f"   No Concept ID:         {len(categories['no_concept_id'])} (need manual review)")
    
    # List FPs with concept IDs (these are likely GT coverage gaps)
    print("\n" + "-" * 100)
    print(" FPs WITH CONCEPT IDs (Likely valid medical terms not in GT)")
    print("-" * 100)
    
    # Deduplicate by normalized text
    seen_texts = set()
    unique_fp_with_concept = []
    for fp in categories['has_concept_id']:
        if fp['text_norm'] not in seen_texts:
            seen_texts.add(fp['text_norm'])
            unique_fp_with_concept.append(fp)
    
    unique_fp_with_concept.sort(key=lambda x: (x['tier'], x['text_norm']))
    
    for fp in unique_fp_with_concept:
        print(f"   T{fp['tier']} | {fp['text']:40} | concept:{fp['concept_id']}")
    
    # List FPs without concept IDs (these need review)
    print("\n" + "-" * 100)
    print(" FPs WITHOUT CONCEPT IDs (Need manual review)")
    print("-" * 100)
    
    seen_texts = set()
    unique_fp_no_concept = []
    for fp in categories['no_concept_id']:
        if fp['text_norm'] not in seen_texts:
            seen_texts.add(fp['text_norm'])
            unique_fp_no_concept.append(fp)
    
    unique_fp_no_concept.sort(key=lambda x: (x['tier'], x['text_norm']))
    
    for fp in unique_fp_no_concept:
        print(f"   T{fp['tier']} | {fp['text']:40} | src={fp['source']}")
    
    # Unique FPs list (deduplicated)
    print("\n" + "-" * 100)
    print(" UNIQUE FALSE POSITIVE TERMS (Deduplicated)")
    print("-" * 100)
    
    unique_fp_texts = set(fp['text_norm'] for fp in all_fp_details)
    print(f"\n   Total unique FP terms: {len(unique_fp_texts)}")
    print(f"   Sorted alphabetically:")
    for i, text in enumerate(sorted(unique_fp_texts), 1):
        print(f"   {i:3}. {text}")
    
    # Ground Truth terms (for reference)
    print("\n" + "-" * 100)
    print(" GROUND TRUTH TERMS (All unique)")
    print("-" * 100)
    print(f"\n   Total unique GT terms: {len(all_gt_texts)}")
    for i, text in enumerate(sorted(all_gt_texts), 1):
        print(f"   {i:3}. {text}")
    
    print("\n" + "=" * 100)
    print(" ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
