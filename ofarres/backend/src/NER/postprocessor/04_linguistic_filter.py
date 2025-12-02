#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_linguistic_filter.py - Linguistic Filter

RESPONSIBILITY: Apply fast heuristic filters to remove obvious garbage from Tier 3
                entities BEFORE sending to expensive Cross-Encoder.

LOGIC:
    Tier 1 & 2: AUTO-PASS (Do not touch - these are dictionary-backed)
    
    Tier 3: Apply exclusion rules. If ANY are true, DROP the entity:
    
    1. Is Header: Text is ALL CAPS and length > 3
       - Examples to drop: "IMAGING", "HISTORY", "HOSPITAL COURSE"
       - Note: Valid acronyms (CT, MRI) are Tier 1, so won't be touched
    
    2. Is Ghost: Span contains ONLY stopwords, punctuation, or numbers
       - Examples to drop: "and", "the", ".", "123"
    
    3. Is Lonely Modifier: Single token AND POS tag is ADJ, ADV, DET, PRON, or CCONJ
       - Examples to drop: "Severe", "Left", "The", "And"
       - Examples to keep: "Severe Stroke" (multi-token), "Vomiting" (Noun)

INPUT: data/ner/03_deduplicated.json
OUTPUT: data/ner/04_linguistically_clean.json
DEPENDENCIES: spacy (en_core_web_sm for speed)
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

# --- Setup Project Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Constants ---
INPUT_PATH = PROJECT_ROOT / "data" / "ner" / "03_deduplicated.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ner" / "04_linguistically_clean.json"

# POS tags that indicate "lonely modifiers" (garbage when alone)
LONELY_MODIFIER_POS = {'ADJ', 'ADV', 'DET', 'PRON', 'CCONJ', 'SCONJ', 'ADP', 'PART'}

# Additional stopwords beyond spaCy's defaults (medical context)
EXTRA_STOPWORDS = {
    'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for',
    'with', 'by', 'from', 'as', 'is', 'was', 'were', 'are', 'been', 'be',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
    'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
    'his', 'her', 'their', 'our', 'your', 'my', 'who', 'which', 'what',
    'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
    'then', 'here', 'there', 'but', 'if', 'because', 'until', 'while',
    'although', 'though', 'after', 'before', 'since', 'during', 'about',
    'into', 'through', 'over', 'under', 'again', 'further', 'once',
}


class LinguisticFilter:
    """Fast linguistic filter using spaCy for POS tagging."""
    
    def __init__(self):
        """Initialize with en_core_web_sm for speed."""
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        except OSError:
            print("[LinguisticFilter] Downloading en_core_web_sm...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        
        # Merge spaCy stopwords with our extras
        self.stopwords = set(self.nlp.Defaults.stop_words) | EXTRA_STOPWORDS
    
    def is_header(self, text: str) -> bool:
        """
        Check if text is a header (ALL CAPS, length > 3).
        Valid acronyms like CT, MRI are Tier 1, so won't reach here.
        """
        stripped = text.strip()
        return len(stripped) > 3 and stripped.isupper()
    
    def is_ghost(self, text: str) -> bool:
        """
        Check if text contains ONLY stopwords, punctuation, or numbers.
        """
        # Tokenize
        doc = self.nlp(text.lower().strip())
        
        for token in doc:
            # Skip whitespace
            if token.is_space:
                continue
            # If token is not a stopword, punct, or number -> not a ghost
            if not (token.is_stop or token.is_punct or token.like_num or 
                    token.text.lower() in self.stopwords):
                return False
        
        # All tokens were ghosts
        return True
    
    def is_lonely_modifier(self, text: str) -> bool:
        """
        Check if text is a single token with modifier POS tag.
        """
        doc = self.nlp(text.strip())
        
        # Filter out whitespace/punct tokens
        content_tokens = [t for t in doc if not t.is_space and not t.is_punct]
        
        # Must be exactly 1 content token
        if len(content_tokens) != 1:
            return False
        
        token = content_tokens[0]
        return token.pos_ in LONELY_MODIFIER_POS
    
    def should_drop(self, entity: Dict) -> Tuple[bool, str]:
        """
        Determine if a Tier 3 entity should be dropped.
        
        Returns:
            Tuple of (should_drop: bool, reason: str)
        """
        text = entity.get('text', '')
        
        # Rule 1: Is Header
        if self.is_header(text):
            return True, "header"
        
        # Rule 2: Is Ghost
        if self.is_ghost(text):
            return True, "ghost"
        
        # Rule 3: Is Lonely Modifier
        if self.is_lonely_modifier(text):
            return True, "lonely_modifier"
        
        return False, ""


def run_linguistic_filter(verbose: bool = True) -> List[Dict]:
    """
    Main linguistic filter function.
    Returns the filtered assembly data.
    """
    if verbose:
        print("=" * 80)
        print(" STEP 04: LINGUISTIC FILTER (Tier 3 Garbage Removal)")
        print("   Responsibility: Remove headers, ghosts, and lonely modifiers")
        print("=" * 80)
    
    # Load input
    if not INPUT_PATH.exists():
        print(f"[ERROR] Input file not found: {INPUT_PATH}")
        print("[INFO] Run 03_safe_deduplication.py first.")
        return []
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if verbose:
        print(f"\n[LinguisticFilter] Loaded {len(data)} notes from {INPUT_PATH}")
        print("[LinguisticFilter] Initializing spaCy (en_core_web_sm)...")
    
    # Initialize filter
    lf = LinguisticFilter()
    
    if verbose:
        print("[LinguisticFilter] Ready. Processing entities...")
    
    # Stats
    total_before = 0
    total_after = 0
    stats = {
        "tier1_passed": 0,
        "tier2_passed": 0,
        "tier3_passed": 0,
        "tier3_dropped": 0,
        "drop_reasons": {
            "header": 0,
            "ghost": 0,
            "lonely_modifier": 0
        },
        "dropped_examples": []
    }
    
    output_data = []
    
    for note_entry in data:
        note_id = note_entry['note_id']
        annotations = note_entry['annotations']
        
        total_before += len(annotations)
        
        kept = []
        note_dropped = 0
        
        for entity in annotations:
            tier = entity.get('priority', 3)
            
            if tier <= 2:
                # Tier 1 & 2: AUTO-PASS
                kept.append(entity)
                if tier == 1:
                    stats["tier1_passed"] += 1
                else:
                    stats["tier2_passed"] += 1
            else:
                # Tier 3: Apply filters
                should_drop, reason = lf.should_drop(entity)
                
                if should_drop:
                    stats["tier3_dropped"] += 1
                    stats["drop_reasons"][reason] += 1
                    note_dropped += 1
                    
                    # Collect examples for verbose output
                    if len(stats["dropped_examples"]) < 20:
                        stats["dropped_examples"].append({
                            "text": entity.get('text', ''),
                            "reason": reason,
                            "note_id": note_id
                        })
                else:
                    kept.append(entity)
                    stats["tier3_passed"] += 1
        
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
        
        print(f"\n[LinguisticFilter] Summary:")
        print(f"    Entities before: {total_before}")
        print(f"    Entities after:  {total_after}")
        print(f"    Reduction:       {reduction} ({reduction_pct:.1f}%)")
        print(f"\n    By Tier:")
        print(f"      Tier 1 (passed): {stats['tier1_passed']}")
        print(f"      Tier 2 (passed): {stats['tier2_passed']}")
        print(f"      Tier 3 (passed): {stats['tier3_passed']}")
        print(f"      Tier 3 (dropped): {stats['tier3_dropped']}")
        print(f"\n    Drop Reasons:")
        print(f"      Headers:          {stats['drop_reasons']['header']}")
        print(f"      Ghosts:           {stats['drop_reasons']['ghost']}")
        print(f"      Lonely Modifiers: {stats['drop_reasons']['lonely_modifier']}")
        
        if stats["dropped_examples"]:
            print(f"\n    Sample Dropped Entities:")
            for ex in stats["dropped_examples"][:10]:
                print(f"      - \"{ex['text']}\" ({ex['reason']})")
        
        print(f"\n    Output saved to: {OUTPUT_PATH}")
    
    return output_data


def main():
    run_linguistic_filter(verbose=True)


if __name__ == "__main__":
    main()
