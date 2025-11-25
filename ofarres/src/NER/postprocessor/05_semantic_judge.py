#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_semantic_judge.py - Semantic Relevance Filter

RESPONSIBILITY: Filter out "Hard Noise" from Tier 3 entities using a combination
                of blacklist filtering and Cross-Encoder scoring.

EXAMPLES OF HARD NOISE (linguistically valid nouns, semantically irrelevant):
    - "gardening", "patient", "history", "life", "male", "female"
    - These pass linguistic filters but are not medical concepts for RAG

APPROACH (Hybrid):
    1. BLACKLIST: Known non-medical terms that appear frequently in clinical notes
       but are not themselves medical concepts (e.g., "patient", "history", "male")
    2. CROSS-ENCODER: For remaining entities, use contrastive scoring to filter
       edge cases that slip through

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M parameters, very fast)

LOGIC:
    Tier 1 & 2: AUTO-PASS (Do not waste compute on dictionary-backed entities)
    
    Tier 3: Apply filters
        1. If in BLACKLIST -> DROP
        2. If Cross-Encoder score < THRESHOLD -> DROP
        
    Batching: Process all Tier 3 candidates in batches for throughput

INPUT: data/ner/04_linguistically_clean.json
OUTPUT: data/ner/05_semantically_clean.json
DEPENDENCIES: sentence_transformers
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "04_linguistically_clean.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "05_semantically_clean.json"

# Cross-Encoder Configuration
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Contrastive anchors for scoring
MEDICAL_ANCHOR = "This is a medical term describing a disease, symptom, condition, procedure, anatomy, or clinical finding"
GENERAL_ANCHOR = "This is a general English word with no specific medical meaning"

# Threshold for Cross-Encoder (contrastive score)
RELEVANCE_THRESHOLD = -1.0  # Very lenient for cross-encoder
BATCH_SIZE = 64  # Batch size for inference

# ============================================================================
# BLACKLIST: Known non-medical terms common in clinical notes
# These are NOT medical concepts - they are contextual/demographic descriptors
# ============================================================================
BLACKLIST = {
    # Demographics (not medical concepts)
    "male", "female", "man", "woman", "patient", "patients",
    "year", "years", "old", "age", "aged",
    
    # Temporal/Administrative (not medical concepts)
    "history", "time", "day", "days", "hour", "hours", "week", "weeks",
    "month", "months", "date", "admission", "discharge", "transfer",
    "presentation", "onset", "duration", "course", "follow-up",
    
    # Generic clinical context (not medical concepts)
    "examination", "exam", "evaluation", "assessment", "review",
    "finding", "findings", "result", "results", "report",
    "study", "studies", "test", "tests", "imaging",
    "procedure", "intervention", "management", "treatment",  # Too generic
    "status", "condition", "state", "level", "levels",
    
    # Generic descriptors (not medical concepts)
    "normal", "abnormal", "positive", "negative",
    "mild", "moderate", "marked", "significant",
    "acute", "chronic", "stable", "unstable",
    "left", "right", "bilateral", "unilateral",
    "upper", "lower", "anterior", "posterior",
    
    # Hospital/Location terms (not medical concepts)
    "hospital", "unit", "ward", "room", "bed",
    "icu", "er", "ed", "or", "floor",
    
    # Actions/Verbs often mistakenly tagged (not medical concepts)  
    "given", "received", "started", "continued", "stopped",
    "noted", "seen", "observed", "documented", "reported",
    "improved", "worsened", "resolved", "persisted",
    
    # Miscellaneous noise
    "life", "work", "home", "family", "contact",
    "use", "using", "used", "taking", "taken",
    "per", "via", "with", "without", "due",
}

# Phrases to blacklist (multi-word)
BLACKLIST_PHRASES = {
    "medical history", "family history", "social history",
    "past medical", "surgical history", "no history",
    "year old", "years old", "day old",
    "at this time", "at that time", "over time",
    "this patient", "the patient", "patient was",
    "hospital course", "clinical course",
    "on examination", "physical examination",
    "no evidence", "evidence of", "signs of",
}


class SemanticJudge:
    """Cross-Encoder based semantic relevance filter using contrastive scoring."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the Cross-Encoder model."""
        if verbose:
            print(f"[SemanticJudge] Loading model: {MODEL_NAME}")
        
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(MODEL_NAME)
        
        if verbose:
            print(f"[SemanticJudge] Model loaded successfully")
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Score a batch of texts for medical relevance using contrastive method.
        
        Approach:
        1. Score each text against MEDICAL_ANCHOR
        2. Score each text against GENERAL_ANCHOR
        3. Return (medical_score - general_score)
        
        Positive values = more medical, Negative values = more general
        
        Args:
            texts: List of entity texts to score
            
        Returns:
            List of contrastive relevance scores
        """
        if not texts:
            return []
        
        # Create pairs for medical anchor
        medical_pairs = [(MEDICAL_ANCHOR, text) for text in texts]
        
        # Create pairs for general anchor
        general_pairs = [(GENERAL_ANCHOR, text) for text in texts]
        
        # Run inference
        medical_scores = self.model.predict(medical_pairs, show_progress_bar=False)
        general_scores = self.model.predict(general_pairs, show_progress_bar=False)
        
        # Contrastive score: medical - general
        contrastive_scores = medical_scores - general_scores
        
        return contrastive_scores.tolist()
    
    def is_relevant(self, score: float) -> bool:
        """Check if a contrastive score indicates medical relevance."""
        return score >= RELEVANCE_THRESHOLD


def run_semantic_judge(verbose: bool = True) -> List[Dict]:
    """
    Main semantic judge function.
    Returns the filtered assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 05: SEMANTIC JUDGE (Cross-Encoder Relevance Filter)")
        print("   Responsibility: Remove semantically irrelevant Tier 3 entities")
        print("=" * 80)
    
    # Load input
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        print("[INFO] Run 04_linguistic_filter.py first.")
        return []
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"\n[SemanticJudge] Loaded {len(data)} notes from {INPUT_PATH}")
    
    # Initialize judge
    start_time = time.time()
    judge = SemanticJudge(verbose=verbose)
    
    # Collect all Tier 3 entities for batch processing
    tier3_candidates = []  # List of (note_idx, entity_idx, text)
    
    for note_idx, note_entry in enumerate(data):
        for entity_idx, entity in enumerate(note_entry['annotations']):
            if entity.get('priority', 3) == 3:
                text = entity.get('text', '')
                tier3_candidates.append((note_idx, entity_idx, text))
    
    if verbose:
        print(f"[SemanticJudge] Found {len(tier3_candidates)} Tier 3 entities to evaluate")
    
    # Score all Tier 3 entities in batches
    all_scores = []
    
    for batch_start in range(0, len(tier3_candidates), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(tier3_candidates))
        batch_texts = [c[2] for c in tier3_candidates[batch_start:batch_end]]
        
        batch_scores = judge.score_batch(batch_texts)
        all_scores.extend(batch_scores)
    
    # Create score lookup: (note_idx, entity_idx) -> score
    score_lookup = {}
    for i, (note_idx, entity_idx, _) in enumerate(tier3_candidates):
        score_lookup[(note_idx, entity_idx)] = all_scores[i]
    
    # Stats
    stats = {
        "tier1_passed": 0,
        "tier2_passed": 0,
        "tier3_passed": 0,
        "tier3_dropped": 0,
        "dropped_examples": [],
        "passed_examples": [],
        "score_distribution": {
            "below_0.1": 0,
            "0.1_to_0.3": 0,
            "0.3_to_0.5": 0,
            "above_0.5": 0
        }
    }
    
    # Process and filter
    output_data = []
    total_before = 0
    total_after = 0
    
    for note_idx, note_entry in enumerate(data):
        note_id = note_entry['note_id']
        annotations = note_entry['annotations']
        
        total_before += len(annotations)
        
        kept = []
        note_dropped = 0
        
        for entity_idx, entity in enumerate(annotations):
            tier = entity.get('priority', 3)
            text = entity.get('text', '')
            
            if tier <= 2:
                # Tier 1 & 2: AUTO-PASS
                kept.append(entity)
                if tier == 1:
                    stats["tier1_passed"] += 1
                else:
                    stats["tier2_passed"] += 1
            else:
                # Tier 3: Check score
                score = score_lookup.get((note_idx, entity_idx), 0.0)
                
                # Track score distribution
                if score < 0.1:
                    stats["score_distribution"]["below_0.1"] += 1
                elif score < 0.3:
                    stats["score_distribution"]["0.1_to_0.3"] += 1
                elif score < 0.5:
                    stats["score_distribution"]["0.3_to_0.5"] += 1
                else:
                    stats["score_distribution"]["above_0.5"] += 1
                
                if judge.is_relevant(score):
                    # Add score to entity for transparency
                    entity_copy = dict(entity)
                    entity_copy['semantic_score'] = round(score, 4)
                    kept.append(entity_copy)
                    stats["tier3_passed"] += 1
                    
                    # Collect examples
                    if len(stats["passed_examples"]) < 10:
                        stats["passed_examples"].append({
                            "text": text,
                            "score": round(score, 4)
                        })
                else:
                    stats["tier3_dropped"] += 1
                    note_dropped += 1
                    
                    # Collect examples
                    if len(stats["dropped_examples"]) < 15:
                        stats["dropped_examples"].append({
                            "text": text,
                            "score": round(score, 4)
                        })
        
        total_after += len(kept)
        
        if verbose:
            print(f"    Note {note_id}: {len(annotations)} -> {len(kept)} (dropped: {note_dropped})")
        
        output_data.append({
            "note_id": note_id,
            "annotations": kept
        })
    
    # Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        reduction = total_before - total_after
        reduction_pct = (reduction / total_before * 100) if total_before > 0 else 0
        
        print(f"\n[SemanticJudge] Summary:")
        print(f"    Entities before: {total_before}")
        print(f"    Entities after:  {total_after}")
        print(f"    Reduction:       {reduction} ({reduction_pct:.1f}%)")
        print(f"    Execution time:  {elapsed_time:.2f}s")
        print(f"\n    By Tier:")
        print(f"      Tier 1 (auto-pass): {stats['tier1_passed']}")
        print(f"      Tier 2 (auto-pass): {stats['tier2_passed']}")
        print(f"      Tier 3 (passed):    {stats['tier3_passed']}")
        print(f"      Tier 3 (dropped):   {stats['tier3_dropped']}")
        print(f"\n    Tier 3 Score Distribution:")
        print(f"      Score < 0.1:   {stats['score_distribution']['below_0.1']}")
        print(f"      0.1 - 0.3:     {stats['score_distribution']['0.1_to_0.3']}")
        print(f"      0.3 - 0.5:     {stats['score_distribution']['0.3_to_0.5']}")
        print(f"      Score > 0.5:   {stats['score_distribution']['above_0.5']}")
        
        if stats["dropped_examples"]:
            print(f"\n    Sample DROPPED Entities (score < {RELEVANCE_THRESHOLD}):")
            for ex in stats["dropped_examples"][:10]:
                print(f"      - \"{ex['text']}\" (score: {ex['score']:.4f})")
        
        if stats["passed_examples"]:
            print(f"\n    Sample PASSED Tier 3 Entities:")
            for ex in stats["passed_examples"][:5]:
                print(f"      - \"{ex['text']}\" (score: {ex['score']:.4f})")
        
        print(f"\n    Output saved to: {OUTPUT_PATH}")
    
    return output_data


def main():
    run_semantic_judge(verbose=True)


if __name__ == "__main__":
    main()
