"""
LLM-based Entity Extractor for Spanish Radiology Reports.

Uses OpenAI GPT-4o-mini to extract pathological findings with contextual understanding.
Handles negations and ignores normal anatomy.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
# Try multiple locations for .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Try to load from new_ofarres/.env first, then from root
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
load_dotenv()  # Also check current directory


class EntityExtractorLLM:
    """
    LLM-based Named Entity Recognition for clinical text.
    
    Uses GPT-4o-mini to extract ATOMIC pathological findings while:
    - Ignoring negated entities
    - Ignoring normal/patent anatomy
    - Extracting SHORT, precise spans mappable to a medical taxonomy
    """
    
    SYSTEM_PROMPT = """Eres un radiólogo experto extrayendo entidades médicas ATÓMICAS de informes radiológicos en español.

TAREA: Extrae entidades médicas CORTAS y PRECISAS que puedan mapearse a una taxonomía médica estándar (como SNOMED-CT o RadLex).

FORMATO DE SALIDA OBLIGATORIO:
Cada entidad debe seguir el formato: (TEXTO_EXACTO) [CATEGORÍA]
Categorías válidas: [FINDING], [ANATOMY], [PROCEDURE]

REGLA DE BREVEDAD (CRÍTICA):
- Extrae el SPAN MÁS CORTO posible: solo el término médico base.
- NUNCA incluyas adjetivos calificativos como: crítica, crítico, leve, bilateral, crónico, crónica, severo, severa, moderado, agudo, aguda, significativo, completa, completo, parcial, total, extensa, discreto, discreta, importante, marcado.
- Separa SIEMPRE hallazgos patológicos de localizaciones anatómicas.

EJEMPLOS CORRECTOS:
Texto: "Estenosis crítica de aspecto crónico de la ACI izquierda"
✅ CORRECTO: (estenosis) [FINDING], (ACI izquierda) [ANATOMY]
❌ INCORRECTO: (estenosis crítica) [FINDING]
❌ INCORRECTO: (Estenosis crítica de aspecto crónico de la ACI izquierda) [FINDING]

Texto: "Oclusión completa de la arteria basilar"
✅ CORRECTO: (oclusión) [FINDING], (arteria basilar) [ANATOMY]
❌ INCORRECTO: (oclusión completa) [FINDING]

Texto: "Hematoma intraparenquimatoso agudo en hemisferio cerebeloso derecho"
✅ CORRECTO: (hematoma intraparenquimatoso) [FINDING], (hemisferio cerebeloso derecho) [ANATOMY]

Texto: "Hipodensidades parcheadas en sustancia blanca"
✅ CORRECTO: (hipodensidades) [FINDING], (sustancia blanca) [ANATOMY]
❌ INCORRECTO: (hipodensidades parcheadas) [FINDING]

Texto: "ASPECTS 9" o "ASPECTS 10"
✅ CORRECTO: (ASPECTS 9) [FINDING] - Mantener el número porque es parte del score

REGLAS DE EXCLUSIÓN ESTRICTAS:

1. NEGACIONES - NO EXTRAER NADA cuando el texto indique ausencia:
   - "No se observa hemorragia" → NO EXTRAER hemorragia
   - "Sin signos de isquemia aguda" → NO EXTRAER isquemia
   - "sin evidencia de lesión" → NO EXTRAER lesión
   - "no hay hemorragia" → NO EXTRAER hemorragia
   - "sin hemorragia" → NO EXTRAER hemorragia
   - "descarta hemorragia" → NO EXTRAER hemorragia

2. ANATOMÍA NORMAL - NO EXTRAER cuando sea normal/permeable:
   - "Arterias permeables" → NO EXTRAER
   - "Sistema ventricular de tamaño normal" → NO EXTRAER
   - "sin signos de estenosis" → NO EXTRAER
   - "Arteria basilar permeable" → NO EXTRAER

3. NO EXTRAER entidades duplicadas - si ya extrajiste "estenosis", no la extraigas de nuevo del mismo texto.

FORMATO JSON DE RESPUESTA:
{"entities": ["(texto1) [CATEGORY]", "(texto2) [CATEGORY]"]}

Si no hay entidades patológicas válidas, devuelve: {"entities": []}"""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize the LLM extractor.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        print(f"✅ [LLM] Initialized with model: {self.model}")

    def _call_llm(self, text: str) -> str:
        """
        Call the OpenAI API with the given text.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            Raw response content from the LLM
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,  # Deterministic output
                response_format={"type": "json_object"}  # Force JSON output
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ [LLM] API call failed: {e}")
            return '{"entities": []}'

    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the JSON response from the LLM with structured format.
        
        Expected format: {"entities": ["(texto) [CATEGORY]", ...]}
        
        Args:
            response: Raw JSON string from LLM
            
        Returns:
            List of dicts with 'text' and 'category' keys
        """
        try:
            data = json.loads(response)
            raw_entities = data.get("entities", [])
            
            parsed_entities = []
            # Regex to extract: (text_inside_parens) [CATEGORY]
            pattern = re.compile(r'\(([^)]+)\)\s*\[([A-Z]+)\]')
            
            for raw in raw_entities:
                if not raw:
                    continue
                    
                match = pattern.search(str(raw))
                if match:
                    text = match.group(1).strip()
                    category = match.group(2).strip()
                    if text:
                        parsed_entities.append({
                            "text": text,
                            "category": category
                        })
                else:
                    # Fallback: if no pattern match, use raw text (strip any brackets)
                    clean_text = re.sub(r'[\(\)\[\]]', '', str(raw)).strip()
                    if clean_text:
                        parsed_entities.append({
                            "text": clean_text,
                            "category": "UNKNOWN"
                        })
            
            return parsed_entities
            
        except json.JSONDecodeError as e:
            print(f"⚠️ [LLM] JSON parse error: {e}")
            return []

    def _find_entity_spans(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Find the character spans (start, end) for each entity in the original text.
        
        Handles:
        - Case-insensitive matching
        - Multiple occurrences (returns first valid match per entity)
        - Category metadata preservation
        
        Args:
            text: Original text to search in
            entities: List of dicts with 'text' and 'category' keys
            
        Returns:
            List of entity dictionaries with span information
        """
        results = []
        used_spans = set()  # Track used character positions to avoid duplicates
        
        for entity_info in entities:
            entity_text = entity_info.get("text", "")
            category = entity_info.get("category", "UNKNOWN")
            
            if not entity_text or not entity_text.strip():
                continue
                
            # Escape special regex characters in the entity
            escaped_entity = re.escape(entity_text.strip())
            
            # Try exact match first (case-insensitive)
            pattern = re.compile(escaped_entity, re.IGNORECASE)
            
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                span_key = (start, end)
                
                # Avoid duplicate spans
                if span_key not in used_spans:
                    used_spans.add(span_key)
                    results.append({
                        "code": None,  # LLM doesn't provide taxonomy codes
                        "start": start,
                        "end": end,
                        "text": text[start:end],  # Use original text casing
                        "category": category,
                        "source": "LLM"
                    })
                    break  # Take first valid match for this entity
        
        return results

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract pathological entities from clinical text.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            List of entity dictionaries:
            [{"code": None, "start": int, "end": int, "text": str, "source": "LLM"}]
        """
        if not text or not text.strip():
            return []
        
        # 1. Call the LLM
        raw_response = self._call_llm(text)
        
        # 2. Parse the JSON response
        entities = self._parse_llm_response(raw_response)
        
        # 3. Find spans in original text
        results = self._find_entity_spans(text, entities)
        
        return results

    def run_on_notes(self, notes_path: str) -> List[Dict[str, Any]]:
        """
        Process the medical_notes.json file.
        
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

        processed_notes = []
        fields_to_scan = ['history', 'findings', 'impression']

        print(f"🚀 [LLM] Processing {len(notes_list)} notes...")

        for i, note in enumerate(notes_list):
            note_id = note.get('id')
            clinical_data = note.get('clinical_data', {})
            
            found_entities = []

            for field in fields_to_scan:
                text_content = clinical_data.get(field)
                if text_content:
                    # Get entities for this specific section
                    entities = self.predict(text_content)
                    
                    # Add field metadata
                    for ent in entities:
                        ent['field_location'] = field
                    
                    found_entities.extend(entities)

            processed_notes.append({
                "id": note_id,
                "extracted_entities": found_entities
            })
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(notes_list)} notes...")

        return processed_notes


# --- Execution Block (for testing) ---
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    NOTES_FILE = os.path.join(BASE_DIR, "data", "medical_notes.json")

    try:
        # Initialize extractor
        extractor = EntityExtractorLLM()
        
        # Load notes
        with open(NOTES_FILE, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        
        # Test on first note only
        first_note = notes[0]
        note_id = first_note.get('id')
        clinical_data = first_note.get('clinical_data', {})
        
        print(f"\n--- Testing on Note 1 (ID: {note_id}) ---\n")
        
        all_entities = []
        
        for field in ['history', 'findings', 'impression']:
            text = clinical_data.get(field, "")
            if text:
                print(f"📄 Processing field: {field}")
                print(f"   Text length: {len(text)} chars")
                
                entities = extractor.predict(text)
                
                for ent in entities:
                    ent['field_location'] = field
                
                all_entities.extend(entities)
                print(f"   Found {len(entities)} entities\n")
        
        # Print results
        result = {
            "id": note_id,
            "extracted_entities": all_entities
        }
        
        print("\n--- LLM Extraction Output (Note 1) ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
