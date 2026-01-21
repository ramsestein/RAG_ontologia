"""
Helper script to write ensamble.py file correctly.
Run with: python write_ensamble.py
"""

code = '''# -*- coding: utf-8 -*-
"""
Strict Hybrid Ensemble NER Pipeline for Spanish Radiology Reports.
"""

import json
import os
import unicodedata
from typing import List, Dict, Any, Tuple

from DFA import EntityExtractorDFA
from LLM import EntityExtractorLLM


class NEREnsemble:
    """Strict Hybrid Ensemble NER combining DFA and LLM extractors."""
    
    def __init__(self, taxonomy_path: str):
        print("=" * 60)
        print("[INIT] Initializing STRICT Hybrid Ensemble Pipeline")
        print("=" * 60)
        
        self.dfa_extractor = EntityExtractorDFA(taxonomy_path)
        self.llm_extractor = EntityExtractorLLM()
        self._load_taxonomy_for_lookup(taxonomy_path)
        
        print("[OK] Ensemble ready! (Strict mode: only coded entities survive)")
        print("=" * 60 + "\\n")

    def _load_taxonomy_for_lookup(self, taxonomy_path: str):
        self.alias_to_code = {}
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        
        for item in taxonomy:
            code = item.get('code')
            aliases = item.get('aliases', [])
            local_name = item.get('local_name', '')
            all_names = aliases + ([local_name] if local_name else [])
            
            for alias in all_names:
                if alias:
                    normalized = self._normalize_for_lookup(alias)
                    if normalized and normalized not in self.alias_to_code:
                        self.alias_to_code[normalized] = code
        
        print(f"[Ensemble] Loaded {len(self.alias_to_code)} aliases for code lookup.")

    def _normalize_for_lookup(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFD', text)
        text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text.lower().strip()

    def _resolve_code(self, entity_text: str) -> str:
        normalized = self._normalize_for_lookup(entity_text)
        if not normalized or len(normalized) < 2:
            return None
        
        # Direct match
        if normalized in self.alias_to_code:
            return self.alias_to_code[normalized]
        
        # Substring matches
        if len(normalized) >= 4:
            for alias, code in self.alias_to_code.items():
                if len(alias) >= 4:
                    if normalized in alias or alias in normalized:
                        return code
        return None

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        return span1[0] < span2[1] and span2[0] < span1[1]

    def _merge_with_maximal_munch(self, dfa_entities: List[Dict], llm_entities: List[Dict]) -> List[Dict]:
        all_entities = []
        
        for ent in dfa_entities:
            ent_copy = ent.copy()
            ent_copy['source'] = 'DFA'
            all_entities.append(ent_copy)
            
        for ent in llm_entities:
            ent_copy = ent.copy()
            ent_copy['source'] = 'LLM'
            if ent_copy.get('code') is None:
                ent_copy['code'] = self._resolve_code(ent_copy.get('text', ''))
            all_entities.append(ent_copy)
        
        # STRICT FILTER - Remove entities without valid codes
        coded_entities = [e for e in all_entities if e.get('code') is not None]
        discarded = len(all_entities) - len(coded_entities)
        if discarded > 0:
            print(f"   [FILTER] Discarded {discarded} entities without codes")
        
        if not coded_entities:
            return []
        
        # Sort by start, then length descending
        coded_entities.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        # Maximal munch
        merged = []
        for candidate in coded_entities:
            cspan = (candidate['start'], candidate['end'])
            clen = candidate['end'] - candidate['start']
            
            overlap_idx = -1
            for idx, kept in enumerate(merged):
                if self._spans_overlap(cspan, (kept['start'], kept['end'])):
                    overlap_idx = idx
                    break
            
            if overlap_idx == -1:
                merged.append(candidate)
            else:
                kept = merged[overlap_idx]
                klen = kept['end'] - kept['start']
                if clen > klen or (clen == klen and candidate['source'] == 'LLM'):
                    merged[overlap_idx] = candidate
        
        merged.sort(key=lambda x: x['start'])
        return merged

    def process_text(self, text: str, field_location: str = None) -> List[Dict]:
        if not text or not text.strip():
            return []
        
        dfa_entities = self.dfa_extractor.predict(text)
        llm_entities = self.llm_extractor.predict(text)
        merged = self._merge_with_maximal_munch(dfa_entities, llm_entities)
        
        if field_location:
            for ent in merged:
                ent['field_location'] = field_location
        return merged

    def process_note(self, note: Dict) -> Dict:
        note_id = note.get('id')
        clinical_data = note.get('clinical_data', {})
        all_entities = []
        
        for field in ['history', 'findings', 'impression']:
            text = clinical_data.get(field, "")
            if text:
                all_entities.extend(self.process_text(text, field_location=field))
        
        return {"id": note_id, "extracted_entities": all_entities}

    def run_on_notes(self, notes_path: str) -> List[Dict]:
        if not os.path.exists(notes_path):
            raise FileNotFoundError(f"Notes file not found: {notes_path}")
        
        with open(notes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            notes_list = data if isinstance(data, list) else [data]
        
        print(f"[Ensemble] Processing {len(notes_list)} notes...")
        results = []
        for i, note in enumerate(notes_list):
            results.append(self.process_note(note))
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(notes_list)} notes...")
        print(f"[OK] Completed processing {len(notes_list)} notes.")
        return results


def verify_note1(entities: List[Dict]) -> bool:
    print("\\n" + "=" * 60)
    print("[VERIFY] Note 1 Requirements")
    print("=" * 60)
    
    passed = True
    
    # Check 1: No hemorragia
    hem = [e for e in entities if 'hemorragia' in e.get('text', '').lower()]
    if hem:
        print(f"[FAIL] hemorragia found: {[h['text'] for h in hem]}")
        passed = False
    else:
        print("[PASS] hemorragia NOT extracted")
    
    # Check 2: No null codes
    nulls = [e for e in entities if e.get('code') is None]
    if nulls:
        print(f"[FAIL] {len(nulls)} entities with code=null")
        passed = False
    else:
        print("[PASS] Zero entities with code=null")
    
    print("=" * 60)
    return passed


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TAXONOMY_FILE = os.path.join(BASE_DIR, "data", "processed", "taxonomy.json")
    NOTES_FILE = os.path.join(BASE_DIR, "data", "medical_notes.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "ensemble_results.json")

    try:
        ensemble = NEREnsemble(TAXONOMY_FILE)
        results = ensemble.run_on_notes(NOTES_FILE)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\\n[SAVED] {OUTPUT_FILE}")
        
        if results:
            note1 = results[0]
            print(f"\\n[NOTE 1] ID: {note1['id']}, Entities: {len(note1['extracted_entities'])}")
            verify_note1(note1['extracted_entities'])
            print("\\n[JSON] Note 1:")
            print(json.dumps(note1, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
'''

# Write the file
with open('ensamble.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("ensamble.py written successfully!")
print(f"File size: {len(code)} bytes")
