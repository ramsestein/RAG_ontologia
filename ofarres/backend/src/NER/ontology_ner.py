import json
import spacy
from flashtext import KeywordProcessor
from typing import List, Dict, Set
from pathlib import Path

class OntologyNER:
    """
    Worker NER Genérico y Robusto.
    
    ESTRATEGIA "NO-CHEAT":
    1. Sin reglas morfológicas ad-hoc (ni 'tion' -> 't', ni 'rhage' -> 'gic').
    2. Sin listas negras arbitrarias (ni 'artery', ni 'disease').
    3. Usa puramente:
       - Variaciones exactas de la ontología.
       - Pluralización estándar del inglés.
       - Extracción de núcleo (Head Word) filtrada SOLO por stopwords estándar.
    """
    
    def __init__(self, ontology_path: str = None, min_term_len: int = 2, **kwargs):
        # 1. Resolución de Rutas
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent.parent / "ontology" / "multilingual_ontology.json"
        else:
            ontology_path = Path(ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = Path(__file__).parent.parent.parent / ontology_path
            
        self.ontology_path = ontology_path
        self.min_term_len = min_term_len 
        
        print(f"[OntologyNER] Loading ontology from: {self.ontology_path}")
        
        # 2. Carga de Stopwords Estándar (SpaCy)
        # Esta es la ÚNICA fuente de filtrado. No hay listas hardcodeadas.
        try:
            self.nlp = spacy.blank("en")
            self.stopwords = self.nlp.Defaults.stop_words
        except:
            # Fallback mínimo si spaCy falla
            self.stopwords = {"of", "in", "at", "on", "to", "by", "is", "it", "no", "us", "am", "pm", "the", "a", "an"}

        # 3. Configuración FlashText
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.keyword_processor.add_non_word_boundary('-') 
        self.keyword_processor.add_non_word_boundary('/')
        
        self._load_and_expand()

    def _generate_generic_variations(self, term: str) -> Set[str]:
        """
        Genera variaciones lingüísticas universales.
        """
        variations = {term}
        clean_term = term.strip()
        
        # 1. Pluralización Estándar (Universal en Inglés/Español)
        if clean_term.lower().endswith('s'):
            variations.add(clean_term[:-1]) 
        else:
            variations.add(clean_term + 's') 
            
        # 2. Normalización de Puntuación
        if '-' in clean_term:
            variations.add(clean_term.replace('-', ' '))

        # 3. Head Word Extraction (Extracción de Núcleo Sintáctico)
        # En inglés médico, la palabra más importante suele ir al final.
        # "Acute Myocardial Infarction" -> "Infarction"
        # "Basilar Artery" -> "Artery"
        parts = clean_term.split()
        if len(parts) > 1:
            last_word = parts[-1]
            
            # FILTRO PURO: Solo indexamos si:
            # a) Tiene longitud suficiente
            # b) NO es una stopword estándar (evita indexar "finding" de "Clinical finding" si fuera stopword)
            if len(last_word) >= self.min_term_len and last_word.lower() not in self.stopwords:
                variations.add(last_word)
                variations.add(last_word + 's')

        return variations

    def _load_and_expand(self):
        if not self.ontology_path.exists():
            print(f"[ERROR] Ontology path not found: {self.ontology_path}")
            return

        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] JSON load failed: {e}")
            return

        term_count = 0
        
        for entry in data:
            cid = entry['concept_id']
            all_raw_terms = set()
            
            # Recolección Multi-idioma
            for lang_code in ["es", "en", "ca"]:
                lang_data = entry.get("languages", {}).get(lang_code, {})
                terms = lang_data.get("terms", [])
                if isinstance(terms, list):
                    all_raw_terms.update(terms)
                elif isinstance(terms, str):
                    all_raw_terms.add(terms)

            # Procesamiento
            for raw_term in all_raw_terms:
                if not raw_term: continue
                
                # Check inicial de longitud
                if len(raw_term) < self.min_term_len:
                    continue
                
                expanded_set = self._generate_generic_variations(raw_term)
                
                for final_term in expanded_set:
                    # FILTRO 1: Longitud mínima
                    if len(final_term) < self.min_term_len:
                        continue
                    
                    # FILTRO 2: Stopwords
                    # Evita que 'In' (de Indium) o 'At' se indexen.
                    if final_term.lower() in self.stopwords:
                        continue
                        
                    self.keyword_processor.add_keyword(final_term, cid)
                    term_count += 1
                        
        print(f"[OntologyNER] [OK] Engine Ready. {term_count} terms indexed (No-Cheat Mode).")

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
                "label": "ONTOLOGY",
                "concept_id": concept_id
            })
            
        return predictions