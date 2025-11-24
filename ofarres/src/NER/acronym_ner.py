import json
from flashtext import KeywordProcessor
from typing import List, Dict
from pathlib import Path

class AcronymNER:
    """
    Worker Especialista en Acrónimos (Dinámico).
    
    ESTRATEGIA:
    1. Carga la ontología completa (JSON).
    2. Filtra términos que parecen acrónimos (longitud corta).
    3. Los indexa en modo CASE SENSITIVE (Estricto).
    
    Esto permite detectar 'CT' (Tomografía) ignorando 'ct' (final de 'act'),
    sin necesidad de hardcodear ninguna lista.
    """
    
    def __init__(self, ontology_path: str = None, max_len: int = 6, **kwargs):
        # 1. Resolución de Rutas (Igual que el NER principal)
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent.parent / "ontology" / "multilingual_ontology.json"
        else:
            ontology_path = Path(ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = Path(__file__).parent.parent.parent / ontology_path
            
        self.ontology_path = ontology_path
        self.max_len = max_len
        
        print(f"[AcronymNER] Extrayendo siglas dinámicamente de: {self.ontology_path}")
        
        # 2. CONFIGURACIÓN CRÍTICA: Case Sensitive = True
        # Esto es lo que diferencia este worker del OntologyNER normal.
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        
        self._load_dynamic_acronyms()

    def _is_likely_acronym(self, term: str) -> bool:
        """
        Heurística para decidir si un término de la ontología debe tratarse como acrónimo estricto.
        """
        term = term.strip()
        
        # Regla 1: Longitud (Los acrónimos suelen ser cortos, ej: 2-6 letras)
        if not (2 <= len(term) <= self.max_len):
            return False
            
        # Regla 2: Composición
        # - Si es todo mayúsculas (CT, MRI, ACV) -> SÍ
        # - Si mezcla mayúsculas/minúsculas (tPA, HbA1c, mRNA) -> SÍ
        # - Si es todo minúsculas (stroke, ictus) -> NO (Eso es trabajo del OntologyNER general)
        if term.islower():
            return False
            
        return True

    def _load_dynamic_acronyms(self):
        if not self.ontology_path.exists():
            print(f"[ERROR] No existe {self.ontology_path}")
            return

        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] JSON corrupto: {e}")
            return

        count = 0
        for entry in data:
            cid = entry['concept_id']
            
            # Recorremos todos los idiomas
            candidates = set()
            for lang in ["es", "en", "ca"]:
                terms = entry.get("languages", {}).get(lang, {}).get("terms", [])
                candidates.update(terms)
            
            # Filtramos y añadimos
            for term in candidates:
                if self._is_likely_acronym(term):
                    # FlashText: (Term, ID)
                    self.keyword_processor.add_keyword(term, cid)
                    count += 1
        
        print(f"[AcronymNER] ✅ Motor listo. {count} siglas detectadas y cargadas dinámicamente.")

    def extract_entities(self, text: str) -> List[Dict]:
        if not self.keyword_processor:
            return []
            
        found = self.keyword_processor.extract_keywords(text, span_info=True)
        
        predictions = []
        for concept_id, start, end in found:
            predictions.append({
                "start": start,
                "end": end,
                "span_text": text[start:end], 
                "label": "ACRONYM", # Etiqueta diferenciada para debug
                "concept_id": concept_id
            })
            
        return predictions