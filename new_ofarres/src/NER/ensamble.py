"""
Ensemble NER Pipeline for Spanish Radiology Reports.

Combines DFA (dictionary-based) and LLM (context-aware) extractors
using a "Maximal Munch" strategy for overlap resolution.
"""

import json
import os
from typing import List, Dict, Any, Tuple

# Import the extractors
from DFA import EntityExtractorDFA
from LLM import EntityExtractorLLM


class NEREnsemble:
    """
    Ensemble NER that combines DFA and LLM extractors.
    
    Strategy:
    - DFA: High recall on exact taxonomy matches
    - LLM: Context-aware, handles negations
    - Merge: Longest span wins (Maximal Munch)
    - Code Resolution: LLM entities get codes via taxonomy lookup
    """
    
    def __init__(self, taxonomy_path: str):
        """
        Initialize both extractors.
        
        Args:
            taxonomy_path: Path to the taxonomy JSON file for DFA
        """
        print("=" * 60)
        print("🔧 Initializing NER Ensemble Pipeline")
        print("=" * 60)
        
        # Initialize DFA extractor
        self.dfa_extractor = EntityExtractorDFA(taxonomy_path)
        
        # Initialize LLM extractor
        self.llm_extractor = EntityExtractorLLM()
        
        # Load taxonomy for code resolution
        self._load_taxonomy_for_lookup(taxonomy_path)
        
        print("=" * 60)
        print("✅ Ensemble ready!")
        print("=" * 60 + "\n")

    def _load_taxonomy_for_lookup(self, taxonomy_path: str):
        """
        Load taxonomy and build a lookup dictionary for code resolution.
        Maps normalized aliases to codes.
        """
        import unicodedata
        
        self.alias_to_code = {}
        
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        
        for item in taxonomy:
            code = item.get('code')
            aliases = item.get('aliases', [])
            local_name = item.get('local_name', '')
            
            # Add all aliases (normalized)
            all_names = aliases + ([local_name] if local_name else [])
            
            for alias in all_names:
                if alias:
                    # Normalize: lowercase + remove accents
                    normalized = self._normalize_for_lookup(alias)
                    if normalized and normalized not in self.alias_to_code:
                        self.alias_to_code[normalized] = code
        
        print(f"📚 [Ensemble] Loaded {len(self.alias_to_code)} aliases for code lookup.")

    def _normalize_for_lookup(self, text: str) -> str:
        """Normalize text for taxonomy lookup (lowercase, no accents)."""
        import unicodedata
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text.lower().strip()

    def _resolve_code(self, entity_text: str) -> str:
        """
        Try to find a taxonomy code for an entity text.
        
        Args:
            entity_text: The extracted entity text
            
        Returns:
            Taxonomy code if found, None otherwise
        """
        normalized = self._normalize_for_lookup(entity_text)
        
        # Direct match
        if normalized in self.alias_to_code:
            return self.alias_to_code[normalized]
        
        # Try partial matches (entity might be substring of alias or vice versa)
        for alias, code in self.alias_to_code.items():
            # If the entity contains the alias or alias contains the entity
            if len(normalized) >= 3 and len(alias) >= 3:
                if normalized in alias or alias in normalized:
                    return code
        
        return None

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """
        Check if two spans overlap.
        
        Spans overlap if: start1 < end2 AND start2 < end1
        
        Args:
            span1: (start, end) tuple
            span2: (start, end) tuple
            
        Returns:
            True if spans overlap
        """
        start1, end1 = span1
        start2, end2 = span2
        return start1 < end2 and start2 < end1

    def _merge_with_maximal_munch(
        self, 
        dfa_entities: List[Dict[str, Any]], 
        llm_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge entities from both extractors using Maximal Munch strategy.
        
        Rules:
        1. If two entities overlap, keep the LONGER one
        2. If same length, prefer LLM (context-aware)
        3. Non-overlapping entities are all kept
        4. LLM entities get codes resolved via taxonomy lookup
        
        Args:
            dfa_entities: Entities from DFA extractor
            llm_entities: Entities from LLM extractor
            
        Returns:
            Merged list of entities with overlaps resolved
        """
        # Combine all entities
        all_entities = []
        
        for ent in dfa_entities:
            ent_copy = ent.copy()
            ent_copy['source'] = 'DFA'
            all_entities.append(ent_copy)
            
        for ent in llm_entities:
            ent_copy = ent.copy()
            ent_copy['source'] = 'LLM'
            # Resolve code for LLM entity
            if ent_copy.get('code') is None:
                resolved_code = self._resolve_code(ent_copy.get('text', ''))
                ent_copy['code'] = resolved_code
            all_entities.append(ent_copy)
        
        if not all_entities:
            return []
        
        # Sort by start position, then by length (descending) for tie-breaking
        all_entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        # Greedy maximal munch: keep non-overlapping entities, preferring longer ones
        merged = []
        
        for candidate in all_entities:
            candidate_span = (candidate['start'], candidate['end'])
            candidate_length = candidate['end'] - candidate['start']
            
            # Check if this candidate overlaps with any already-merged entity
            overlap_found = False
            overlap_idx = -1
            
            for idx, kept in enumerate(merged):
                kept_span = (kept['start'], kept['end'])
                
                if self._spans_overlap(candidate_span, kept_span):
                    overlap_found = True
                    overlap_idx = idx
                    break
            
            if not overlap_found:
                # No overlap, add the candidate
                merged.append(candidate)
            else:
                # Overlap found - apply maximal munch
                kept = merged[overlap_idx]
                kept_length = kept['end'] - kept['start']
                
                # Replace if candidate is longer, or same length but LLM preferred
                should_replace = False
                
                if candidate_length > kept_length:
                    should_replace = True
                elif candidate_length == kept_length:
                    # Same length: prefer LLM over DFA
                    if candidate['source'] == 'LLM' and kept['source'] == 'DFA':
                        should_replace = True
                
                if should_replace:
                    merged[overlap_idx] = candidate
        
        # Sort final results by start position
        merged.sort(key=lambda x: x['start'])
        
        return merged

    def process_text(self, text: str, field_location: str = None) -> List[Dict[str, Any]]:
        """
        Process a single text through both extractors and merge results.
        
        Args:
            text: Clinical text to analyze
            field_location: Optional field name for metadata
            
        Returns:
            Merged list of entities
        """
        if not text or not text.strip():
            return []
        
        # Get DFA results
        dfa_entities = self.dfa_extractor.predict(text)
        
        # Get LLM results
        llm_entities = self.llm_extractor.predict(text)
        
        # Merge with maximal munch
        merged = self._merge_with_maximal_munch(dfa_entities, llm_entities)
        
        # Add field location if provided
        if field_location:
            for ent in merged:
                ent['field_location'] = field_location
        
        return merged

    def process_note(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single medical note.
        
        Args:
            note: Medical note dictionary with clinical_data
            
        Returns:
            Processed note with extracted entities
        """
        note_id = note.get('id')
        clinical_data = note.get('clinical_data', {})
        
        all_entities = []
        fields_to_scan = ['history', 'findings', 'impression']
        
        for field in fields_to_scan:
            text = clinical_data.get(field, "")
            if text:
                entities = self.process_text(text, field_location=field)
                all_entities.extend(entities)
        
        return {
            "id": note_id,
            "extracted_entities": all_entities
        }

    def run_on_notes(self, notes_path: str) -> List[Dict[str, Any]]:
        """
        Process all notes from a JSON file.
        
        Args:
            notes_path: Path to the medical notes JSON file
            
        Returns:
            List of processed notes with extracted entities
        """
        if not os.path.exists(notes_path):
            raise FileNotFoundError(f"Notes file not found at: {notes_path}")

        with open(notes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            notes_list = data if isinstance(data, list) else [data]

        print(f"🚀 [Ensemble] Processing {len(notes_list)} notes...")
        
        processed_notes = []
        
        for i, note in enumerate(notes_list):
            result = self.process_note(note)
            processed_notes.append(result)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(notes_list)} notes...")
        
        print(f"✅ [Ensemble] Completed processing {len(notes_list)} notes.")
        
        return processed_notes


def print_entity_summary(entities: List[Dict[str, Any]], title: str = "Entities"):
    """Helper to print a summary of extracted entities."""
    print(f"\n📊 {title} ({len(entities)} total):")
    print("-" * 50)
    
    # Count entities with/without codes
    with_codes = [e for e in entities if e.get('code') is not None]
    without_codes = [e for e in entities if e.get('code') is None]
    
    print(f"  📌 With taxonomy codes: {len(with_codes)}")
    print(f"  ⚠️  Without codes (unresolved): {len(without_codes)}")
    
    # Group by source
    by_source = {}
    for ent in entities:
        source = ent.get('source', 'UNKNOWN')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(ent)
    
    for source, ents in sorted(by_source.items()):
        source_with_codes = len([e for e in ents if e.get('code') is not None])
        print(f"\n  [{source}] ({len(ents)} entities, {source_with_codes} with codes):")
        for ent in ents[:10]:  # Show first 10
            text = ent.get('text', '')
            code = ent.get('code', 'N/A')
            code_str = code if code else "❌ NULL"
            field = ent.get('field_location', '?')
            category = ent.get('category', '')
            cat_str = f" [{category}]" if category else ""
            print(f"    • \"{text}\"{cat_str} (code: {code_str}, field: {field})")
        if len(ents) > 10:
            print(f"    ... and {len(ents) - 10} more")
    
    # List unresolved entities
    if without_codes:
        print(f"\n  ⚠️  Unresolved entities (no taxonomy match):")
        for ent in without_codes[:5]:
            print(f"    • \"{ent.get('text', '')}\" (source: {ent.get('source', '?')})")
        if len(without_codes) > 5:
            print(f"    ... and {len(without_codes) - 5} more")


# --- Execution Block ---
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TAXONOMY_FILE = os.path.join(BASE_DIR, "data", "processed", "taxonomy.json")
    NOTES_FILE = os.path.join(BASE_DIR, "data", "medical_notes.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "ensemble_results.json")

    try:
        # Initialize ensemble
        ensemble = NEREnsemble(TAXONOMY_FILE)
        
        # Process all notes
        results = ensemble.run_on_notes(NOTES_FILE)
        
        # Save results
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {OUTPUT_FILE}")
        
        # Print detailed results for Note 1
        if results:
            note1 = results[0]
            print("\n" + "=" * 60)
            print(f"📋 DETAILED RESULTS FOR NOTE 1 (ID: {note1['id']})")
            print("=" * 60)
            
            print_entity_summary(note1['extracted_entities'], "Merged Entities")
            
            # Check for "hemorragia" - should NOT be present if negated
            hemorragia_found = [
                e for e in note1['extracted_entities'] 
                if 'hemorragia' in e.get('text', '').lower()
            ]
            
            print("\n" + "-" * 50)
            if hemorragia_found:
                print("⚠️  WARNING: 'hemorragia' was found in Note 1:")
                for h in hemorragia_found:
                    print(f"    • \"{h['text']}\" (source: {h['source']}, field: {h.get('field_location', '?')})")
                print("    This might be a false positive if it was negated in the text.")
            else:
                print("✅ GOOD: 'hemorragia' was NOT extracted (correctly filtered as negated).")
            
            # Full JSON output for Note 1
            print("\n" + "-" * 50)
            print("📄 Full JSON Output (Note 1):")
            print(json.dumps(note1, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
