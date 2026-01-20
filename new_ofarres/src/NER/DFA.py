import json
import os
import unicodedata
from flashtext import KeywordProcessor

class EntityExtractorDFA:
    def __init__(self, taxonomy_path):
        """
        Initializes the Dictionary-based NER.
        
        Args:
            taxonomy_path (str): Path to the JSON taxonomy file.
        """
        self.taxonomy_path = taxonomy_path
        # case_sensitive=False handles A vs a, but we need manual handling for accents (á vs a)
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.taxonomy_loaded = False
        
        # Load taxonomy immediately
        self._load_taxonomy()

    def _normalize_text(self, text):
        """
        Normalizes Spanish text to ensure robust matching.
        1. Decomposes characters (NFD).
        2. Removes non-spacing marks (accents).
        3. Converts to lowercase.
        
        Example: "Artería" -> "arteria"
        """
        if not text:
            return ""
        
        # NFD decomposition separates 'á' into 'a' + '´'
        text = unicodedata.normalize('NFD', text)
        # Filter out the accent marks
        text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
        return text.lower()

    def _load_taxonomy(self):
        """
        Loads the taxonomy JSON and populates the FlashText Trie.
        """
        if not os.path.exists(self.taxonomy_path):
            raise FileNotFoundError(f"Taxonomy file not found at: {self.taxonomy_path}")

        with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)

        count = 0
        for item in taxonomy:
            code = item.get('code')
            aliases = item.get('aliases', [])
            
            # We map every alias to the unique CODE.
            for alias in aliases:
                # Normalize the alias (e.g., store "estenosis" for input "Estenósis")
                clean_alias = self._normalize_text(alias)
                if clean_alias:
                    # When 'clean_alias' is found, return 'code'
                    self.keyword_processor.add_keyword(clean_alias, code)
                    count += 1
        
        self.taxonomy_loaded = True
        print(f"✅ [DFA] Loaded {len(taxonomy)} concepts with {count} aliases.")

    def predict(self, text):
        """
        Extracts entities from a single string of text.
        
        Returns:
            list: [{'code': str, 'start': int, 'end': int, 'source': 'DFA', 'text': str}]
        """
        if not text:
            return []

        # 1. Normalize input (remove accents)
        clean_text = self._normalize_text(text)

        # 2. Extract (returns list of (code, start_idx, end_idx))
        # Note: We use clean_text for matching, but indices usually map 1:1 to original text 
        # for standard Spanish/Latin characters.
        matches = self.keyword_processor.extract_keywords(clean_text, span_info=True)

        results = []
        for code, start, end in matches:
            # We grab the text from the ORIGINAL string using the indices
            original_mention = text[start:end]
            
            results.append({
                "code": code,
                "start": start,
                "end": end,
                "text": original_mention,
                "source": "DFA"  # Tagging source for the Ensemble step
            })
            
        return results

    def run_on_notes(self, notes_path):
        """
        Processes the medical_notes.json file.
        """
        if not os.path.exists(notes_path):
            raise FileNotFoundError(f"Notes file not found at: {notes_path}")

        with open(notes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle if file is a list of notes or a dict wrapping them
            notes_list = data if isinstance(data, list) else [data]

        # Process each note
        processed_notes = []
        fields_to_scan = ['history', 'findings', 'impression']

        print(f"🚀 [DFA] Processing {len(notes_list)} notes...")

        for note in notes_list:
            note_id = note.get('id')
            clinical_data = note.get('clinical_data', {})
            
            found_entities = []

            for field in fields_to_scan:
                text_content = clinical_data.get(field)
                if text_content:
                    # Get entities for this specific section
                    entities = self.predict(text_content)
                    
                    # Add field metadata (optional, but useful for debugging)
                    for ent in entities:
                        ent['field_location'] = field
                    
                    found_entities.extend(entities)

            processed_notes.append({
                "id": note_id,
                "extracted_entities": found_entities
            })

        return processed_notes

# --- Execution Block (for testing) ---
if __name__ == "__main__":
    # Define paths based on your provided images
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TAXONOMY_FILE = os.path.join(BASE_DIR, "data", "processed", "taxonomy.json")
    NOTES_FILE = os.path.join(BASE_DIR, "data", "medical_notes.json")

    try:
        # Initialize
        extractor = EntityExtractorDFA(TAXONOMY_FILE)
        
        # Test on the actual file
        results = extractor.run_on_notes(NOTES_FILE)
        
        # Print first result as sanity check
        if results:
            print("\n--- Sample Output (Note 1) ---")
            print(json.dumps(results[0], indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that your file paths in the '__main__' block match your folder structure.")