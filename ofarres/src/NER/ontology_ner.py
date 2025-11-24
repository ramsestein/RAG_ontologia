import json
import re
from flashtext import KeywordProcessor
from typing import List, Dict, Set
from pathlib import Path

class OntologyNER:
    """
    Worker NER SOTA con Expansión Morfológica Generalizable.
    NO usa listas hardcodeadas. Usa reglas lingüísticas y médicas universales.
    """
    
    def __init__(self, ontology_path: str = None, min_term_len: int = 2, **kwargs):
        # --- 1. Resolución de Rutas ---
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent.parent / "ontology" / "multilingual_ontology.json"
        else:
            ontology_path = Path(ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = Path(__file__).parent.parent.parent / ontology_path
            
        self.ontology_path = ontology_path
        self.min_term_len = min_term_len
        print(f"[NER Worker] Loading ontology from: {self.ontology_path}")
        
        # --- 2. Configuración FlashText ---
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        # Permitir que guiones y barras sean parte de la palabra (ej: "check-up", "t/c")
        # Pero NO añadimos caracteres que queremos que rompan (como espacios)
        self.keyword_processor.add_non_word_boundary('-') 
        self.keyword_processor.add_non_word_boundary('/')
        
        self._load_and_expand()

    def _generate_generic_variations(self, term: str) -> Set[str]:
        """
        Genera variaciones lingüísticas universales (sin hardcoding médico específico).
        """
        variations = {term}
        clean_term = term.strip()
        
        # 1. Pluralización Simple (EN/ES)
        # Regla heurística segura: añadir 's' o quitar 's' final
        if clean_term.lower().endswith('s'):
            variations.add(clean_term[:-1]) 
        else:
            variations.add(clean_term + 's') 
            
        # 2. Descomposición de Frases (Head Word Extraction Heurístico)
        # Si el término es "Acute myocardial infarction", extraemos "Infarction"
        # Regla: Si la última palabra tiene >4 letras, es candidata a ser el núcleo.
        parts = clean_term.split()
        if len(parts) > 1:
            last_word = parts[-1]
            if len(last_word) >= 4:
                variations.add(last_word)
                # Y su plural
                variations.add(last_word + 's')

        # 3. Normalización de Puntuación
        # "Stroke-like" -> "Stroke like"
        if '-' in clean_term:
            variations.add(clean_term.replace('-', ' '))
            
        return variations

    def _load_and_expand(self):
        if not self.ontology_path.exists():
            print(f"[ERROR] Ontology file not found: {self.ontology_path}")
            return

        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {e}")
            return

        term_count = 0
        
        for entry in data:
            cid = entry['concept_id']
            all_raw_terms = set()
            
            # Recolectar términos de todos los idiomas disponibles
            # Esto hace que el sistema sea agnóstico al idioma de entrada
            for lang_code in ["es", "en", "ca"]:
                lang_data = entry.get("languages", {}).get(lang_code, {})
                terms = lang_data.get("terms", [])
                if isinstance(terms, list):
                    all_raw_terms.update(terms)
                elif isinstance(terms, str):
                    all_raw_terms.add(terms)

            # Procesar y Expandir
            for raw_term in all_raw_terms:
                # Limpieza básica
                if not raw_term or len(raw_term) < self.min_term_len: 
                    continue
                
                # Generar variaciones
                expanded_set = self._generate_generic_variations(raw_term)
                
                for final_term in expanded_set:
                    # Filtrar basura generada (ej: "de", "la")
                    if len(final_term) < self.min_term_len:
                        continue
                        
                    self.keyword_processor.add_keyword(final_term, cid)
                    term_count += 1
                        
        print(f"[NER Worker] ✅ Engine Ready. {term_count} terms indexed (generic expansion).")

    def extract_entities(self, text: str) -> List[Dict]:
        if not self.keyword_processor:
            return []
            
        # FlashText devuelve: [(ID_ENCONTRADO, start, end), ...]
        found = self.keyword_processor.extract_keywords(text, span_info=True)
        
        predictions = []
        for concept_id, start, end in found:
            predictions.append({
                "start": start,
                "end": end,
                "span_text": text[start:end], 
                "label": "ONTOLOGY",
                "concept_id": concept_id
            })
            
        return predictions