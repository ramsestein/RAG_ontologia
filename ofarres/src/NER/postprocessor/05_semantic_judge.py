#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_semantic_judge.py - Multi-Anchor Semantic Competition (K-Anchor Zero-Shot Logic)

RESPONSIBILITY: Filter out "Hard Noise" from Tier 2 & 3 entities using Multi-Anchor
                Semantic Competition. NO BLACKLISTS. Pure semantic inference.

EXAMPLES OF HARD NOISE (linguistically valid nouns, semantically irrelevant):
    - "gardening", "patient", "history", "life", "male", "female"
    - These pass linguistic filters but are not medical concepts for RAG

APPROACH (Semantic Race):
    Instead of asking "Is this medical?" (Binary), compare the candidate against
    K competing semantic definitions. The definition with the highest Cross-Encoder
    score wins.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M parameters, very fast)

THE ANCHORS (4 Competitors that cleave the embedding space):
    TARGET (Keep): "A clinical medical term: disease, symptom, anatomical structure,
                   procedure, diagnostic test, drug, or pathological finding."
    
    NOISE_DEMO (Drop): "A patient demographic: male, female, elderly, young,
                       adult, child, age, patient, race, ethnicity, person."
    
    NOISE_ADMIN (Drop): "A clinical documentation term: history, presentation,
                        examination, admission, discharge, hospital, course."
    
    NOISE_GENERIC (Drop): "A common non-clinical English word: improved, completed,
                          noted, observed, consistent, found, life, gardening."

LOGIC:
    Tier 1: AUTO-PASS (Elite - highest confidence, do not touch)
    
    Tier 2 & 3: Run the "Semantic Race"
        1. Score candidate against all 4 Anchors
        2. If TARGET has highest score -> KEEP
        3. If NOISE wins BUT margin < NOISE_WIN_MARGIN -> KEEP (low confidence)
        4. If NOISE wins with sufficient margin -> DROP
        
    Batching: Process all Tier 2 & 3 candidates in batches for throughput

MARGIN-BASED DECISION:
    To preserve recall, we use a margin-based decision:
    - If NOISE wins by less than NOISE_WIN_MARGIN, default to KEEP
    - This prevents dropping valid medical terms when the model has low confidence

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

# ============================================================================
# DECISION THRESHOLDS
# ============================================================================

# Minimum margin a NOISE anchor must win by to trigger a DROP.
# If all scores are within this margin, we default to KEEP (favoring recall).
# This prevents dropping valid medical terms when the model has low confidence.
# Lower value = more aggressive filtering (drops more)
# Higher value = more conservative (keeps more)
NOISE_WIN_MARGIN = 0.2

# ============================================================================
# SEMANTIC ANCHORS - The K Competitors
# These definitions are designed to be mutually exclusive and cleave the 
# embedding space effectively. NO BLACKLISTS - pure semantic categories.
#
# Key refinements:
# - TARGET is broad to catch all medical terms (anatomical, pathological, etc.)
# - NOISE anchors are specific to catch clear non-medical categories
# - When in doubt (low margin), default to KEEP for recall preservation
# ============================================================================

ANCHOR_TARGET = (
    "A clinical medical term: disease name, symptom, anatomical structure, "
    "medical procedure, diagnostic test, drug, or pathological finding."
)

ANCHOR_NOISE_DEMO = (
    "A patient demographic: male, female, man, woman, elderly, young, adult, "
    "child, age, patient, race, ethnicity, person."
)

ANCHOR_NOISE_ADMIN = (
    "A clinical documentation term: history, presentation, examination, "
    "admission, discharge, hospital, department, arrival, course, evaluation."
)

ANCHOR_NOISE_GENERIC = (
    "A common non-clinical English word: improved, completed, noted, observed, "
    "consistent, showed, found, appeared, underwent, life, gardening, activity."
)

# Anchor registry for iteration
ANCHORS = {
    "TARGET": ANCHOR_TARGET,
    "NOISE_DEMO": ANCHOR_NOISE_DEMO,
    "NOISE_ADMIN": ANCHOR_NOISE_ADMIN,
    "NOISE_GENERIC": ANCHOR_NOISE_GENERIC,
}


class SemanticJudge:
    """
    Multi-Anchor Semantic Competition using Cross-Encoder.
    
    The "Semantic Race" approach: Compare each candidate against K competing
    semantic definitions. The definition with the highest score wins.
    """
    
    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize Cross-Encoder model."""
        from sentence_transformers import CrossEncoder
        
        print(f"[SemanticJudge] Loading Cross-Encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        self.anchors = ANCHORS
        print(f"[SemanticJudge] Loaded. Using {len(self.anchors)} competing anchors.")
    
    def score_candidate(self, candidate_text: str) -> Dict[str, float]:
        """
        Score a single candidate against all anchors.
        
        Returns: Dict mapping anchor name to score
        """
        pairs = [(candidate_text, anchor_def) for anchor_def in self.anchors.values()]
        scores = self.model.predict(pairs)
        
        return {name: float(score) for name, score in zip(self.anchors.keys(), scores)}
    
    def score_batch(self, candidates: List[str]) -> List[Dict[str, float]]:
        """
        Score a batch of candidates against all anchors efficiently.
        
        Constructs all (candidate, anchor) pairs and scores in single batch,
        then reshapes results.
        
        Returns: List of dicts mapping anchor name to score for each candidate
        """
        if not candidates:
            return []
        
        # Build all pairs: each candidate against each anchor
        all_pairs = []
        for candidate in candidates:
            for anchor_def in self.anchors.values():
                all_pairs.append((candidate, anchor_def))
        
        # Score all pairs in single batch
        all_scores = self.model.predict(all_pairs)
        
        # Reshape: group scores by candidate
        num_anchors = len(self.anchors)
        results = []
        for i, candidate in enumerate(candidates):
            start_idx = i * num_anchors
            candidate_scores = all_scores[start_idx:start_idx + num_anchors]
            score_dict = {
                name: float(score) 
                for name, score in zip(self.anchors.keys(), candidate_scores)
            }
            results.append(score_dict)
        
        return results
    
    def judge(self, scores: Dict[str, float]) -> Tuple[str, bool, str]:
        """
        Determine winner of the semantic race with margin-based decision.
        
        Decision Logic:
        1. Find the anchor with the highest score (winner)
        2. If TARGET wins -> KEEP
        3. If NOISE wins but margin over TARGET < NOISE_WIN_MARGIN -> KEEP (low confidence)
        4. If NOISE wins with sufficient margin -> DROP
        
        Args:
            scores: Dict mapping anchor name to score
            
        Returns:
            Tuple of (winning_anchor, should_keep, reason)
            - winning_anchor: Name of the anchor with highest score
            - should_keep: True if entity should be kept
            - reason: Explanation for the decision
        """
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]
        target_score = scores["TARGET"]
        
        if winner == "TARGET":
            return winner, True, "TARGET_WINS"
        
        # NOISE anchor won - check margin
        margin = winner_score - target_score
        
        if margin < NOISE_WIN_MARGIN:
            # Low confidence - default to KEEP for recall preservation
            return winner, True, "LOW_MARGIN_KEEP"
        else:
            # Clear NOISE win - DROP
            return winner, False, "NOISE_CLEAR_WIN"


def run_semantic_judge(verbose: bool = True) -> List[Dict]:
    """
    Main semantic judge function using Multi-Anchor Semantic Competition.
    Returns the filtered assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 05: SEMANTIC JUDGE (Multi-Anchor Semantic Competition)")
        print("   Responsibility: Filter Tier 3 via K-Anchor Zero-Shot Classification")
        print("   Method: NO BLACKLISTS - Pure semantic inference with 4 anchors")
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
    judge = SemanticJudge()
    
    # Collect Tier 2 AND Tier 3 entities for batch processing
    # Tier 1 (Elite) auto-passes, Tier 2 & 3 go through semantic filtering
    tier_entities = []
    tier_positions = []  # Track (note_idx, entity_idx, tier) for later
    
    for note_idx, note_entry in enumerate(data):
        for entity_idx, entity in enumerate(note_entry['annotations']):
            tier = entity.get('priority', 3)
            if tier >= 2:  # Tier 2 and Tier 3
                tier_entities.append(entity.get('text', ''))
                tier_positions.append((note_idx, entity_idx, tier))
    
    if verbose:
        tier2_count = sum(1 for _, _, t in tier_positions if t == 2)
        tier3_count = sum(1 for _, _, t in tier_positions if t == 3)
        print(f"[SemanticJudge] Found {len(tier_entities)} entities to evaluate (Tier 2: {tier2_count}, Tier 3: {tier3_count})")
        print("[SemanticJudge] Running Semantic Race (batch scoring)...")
    
    # Batch score all Tier 2 & Tier 3 entities
    start_time = time.time()
    all_scores = judge.score_batch(tier_entities)
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"[SemanticJudge] Batch scoring completed in {elapsed:.2f}s")
        print(f"[SemanticJudge] Throughput: {len(tier_entities) / elapsed:.1f} entities/sec")
    
    # Map (note_idx, entity_idx) -> (winner, should_keep, scores, reason)
    decisions = {}
    for (note_idx, entity_idx, tier), scores in zip(tier_positions, all_scores):
        winner, should_keep, reason = judge.judge(scores)
        decisions[(note_idx, entity_idx)] = (winner, should_keep, scores, reason, tier)
    
    # Stats
    stats = {
        "tier1_passed": 0,
        "tier2_passed": 0,
        "tier2_dropped": 0,
        "tier3_passed": 0,
        "tier3_dropped": 0,
        "anchor_wins": {name: 0 for name in ANCHORS.keys()},
        "decision_reasons": {
            "TARGET_WINS": 0,
            "LOW_MARGIN_KEEP": 0,
            "NOISE_CLEAR_WIN": 0,
        },
        "dropped_examples": [],
        "kept_examples": [],
    }
    
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
            
            if tier == 1:
                # Tier 1: AUTO-PASS (Elite - highest confidence)
                kept.append(entity)
                stats["tier1_passed"] += 1
            else:
                # Tier 2 & 3: Check decision from batch
                pos = (note_idx, entity_idx)
                winner, should_keep, scores, reason, _ = decisions[pos]
                stats["anchor_wins"][winner] += 1
                stats["decision_reasons"][reason] += 1
                
                if should_keep:
                    kept.append(entity)
                    if tier == 2:
                        stats["tier2_passed"] += 1
                    else:
                        stats["tier3_passed"] += 1
                    
                    # Collect examples
                    if len(stats["kept_examples"]) < 10:
                        stats["kept_examples"].append({
                            "text": entity.get('text', ''),
                            "tier": tier,
                            "winner": winner,
                            "reason": reason,
                            "scores": {k: round(v, 3) for k, v in scores.items()},
                            "note_id": note_id
                        })
                else:
                    if tier == 2:
                        stats["tier2_dropped"] += 1
                    else:
                        stats["tier3_dropped"] += 1
                    note_dropped += 1
                    
                    # Collect examples
                    if len(stats["dropped_examples"]) < 20:
                        stats["dropped_examples"].append({
                            "text": entity.get('text', ''),
                            "tier": tier,
                            "winner": winner,
                            "reason": reason,
                            "scores": {k: round(v, 3) for k, v in scores.items()},
                            "note_id": note_id
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
    
    if verbose:
        reduction = total_before - total_after
        reduction_pct = (reduction / total_before * 100) if total_before > 0 else 0
        
        print(f"\n[SemanticJudge] Summary:")
        print(f"    Entities before: {total_before}")
        print(f"    Entities after:  {total_after}")
        print(f"    Reduction:       {reduction} ({reduction_pct:.1f}%)")
        print(f"\n    By Tier:")
        print(f"      Tier 1 (passed): {stats['tier1_passed']}")
        print(f"      Tier 2 (passed): {stats['tier2_passed']}")
        print(f"      Tier 2 (dropped): {stats['tier2_dropped']}")
        print(f"      Tier 3 (passed): {stats['tier3_passed']}")
        print(f"      Tier 3 (dropped): {stats['tier3_dropped']}")
        print(f"\n    Anchor Wins (Semantic Race Results):")
        for anchor, count in stats["anchor_wins"].items():
            marker = "✓ KEEP" if anchor == "TARGET" else "? NOISE"
            print(f"      {anchor}: {count} ({marker})")
        
        print(f"\n    Decision Breakdown:")
        print(f"      TARGET_WINS (keep):      {stats['decision_reasons']['TARGET_WINS']}")
        print(f"      LOW_MARGIN_KEEP (keep):  {stats['decision_reasons']['LOW_MARGIN_KEEP']}")
        print(f"      NOISE_CLEAR_WIN (drop):  {stats['decision_reasons']['NOISE_CLEAR_WIN']}")
        
        if stats["dropped_examples"]:
            print(f"\n    Sample DROPPED Entities (NOISE clear win):")
            for ex in stats["dropped_examples"][:10]:
                print(f"      - T{ex['tier']} \"{ex['text']}\" -> {ex['winner']} ({ex['reason']})")
                print(f"        Scores: {ex['scores']}")
        
        if stats["kept_examples"]:
            print(f"\n    Sample KEPT Entities:")
            for ex in stats["kept_examples"][:5]:
                print(f"      + T{ex['tier']} \"{ex['text']}\" -> {ex['winner']} ({ex['reason']})")
                print(f"        Scores: {ex['scores']}")
        
        print(f"\n    Output saved to: {OUTPUT_PATH}")
    
    return output_data


def main():
    run_semantic_judge(verbose=True)


if __name__ == "__main__":
    main()
