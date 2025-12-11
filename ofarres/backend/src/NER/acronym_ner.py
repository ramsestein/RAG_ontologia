import json
import spacy
from flashtext import KeywordProcessor
from typing import List, Dict
from pathlib import Path

class AcronymNER:
    """
    Worker Especialista en Acrónimos (Stopword-Aware + Boundary Fix).
    
    CORRECCIÓN:
    1. Incluye límites de palabra (-, /, .) para detectar "CT-scan" o "C.T.".
    2. Usa Stopwords para permitir siglas cortas ("CT") bloqueando basura ("AT", "IN").
    """
    
    def __init__(self, ontology_path: str = None, max_len: int = 6, **kwargs):
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent.parent / "ontology" / "multilingual_ontology.json"
        else:
            ontology_path = Path(ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = Path(__file__).parent.parent.parent / ontology_path
            
        self.ontology_path = ontology_path
        self.max_len = max_len
        
        print(f"[AcronymNER] Iniciando motor con lógica Stopword-Aware...")

        # 1. Carga de Stopwords (Universal English)
        try:
            self.nlp = spacy.blank("en")
            self.stopwords = self.nlp.Defaults.stop_words
        except Exception:
            # Fallback seguro
            self.stopwords = {
                "of", "in", "at", "on", "to", "by", "is", "it", "no", "us", "am", "pm", 
                "do", "be", "an", "as", "if", "or", "so", "up", "my", "he", "we", "go",
                "me", "my", "et", "al", "vs"
            }
        
        # 2. Configuración FlashText (Case Sensitive = True)
        self.keyword_processor = KeywordProcessor(case_sensitive=True)
        
        # IMPORTANTE: Permitir que siglas pegadas a puntuación sean detectadas
        # Ej: "CT-scan", "N/A", "C.T."
        self.keyword_processor.add_non_word_boundary('-')
        self.keyword_processor.add_non_word_boundary('/')
        self.keyword_processor.add_non_word_boundary('.')
        
        self._load_dynamic_acronyms()

    def _load_dynamic_acronyms(self):
        if not self.ontology_path.exists():
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
            candidates = set()
            
            # Recolectar de todos los idiomas
            for lang in ["es", "en", "ca"]:
                terms = entry.get("languages", {}).get(lang, {}).get("terms", [])
                if isinstance(terms, list):
                    candidates.update(terms)
                elif isinstance(terms, str):
                    candidates.add(terms)
            
            for term in candidates:
                term_clean = term.strip()
                
                # Filtro de Longitud (Solo siglas)
                if not (2 <= len(term_clean) <= self.max_len):
                    continue
                
                is_stopword = term_clean.lower() in self.stopwords
                
                # Caso A: Sigla Segura (No es stopword: "CT", "MRI")
                if not is_stopword:
                    self.keyword_processor.add_keyword(term_clean.lower(), cid) # ct
                    self.keyword_processor.add_keyword(term_clean.upper(), cid) # CT
                    self.keyword_processor.add_keyword(term_clean.title(), cid) # Ct
                    count += 1
                
                # Caso B: Sigla Peligrosa (Es stopword: "NO", "US")
                # Solo añadimos si es coincidencia exacta con ontología (usualmente mayúsculas)
                else:
                    self.keyword_processor.add_keyword(term_clean, cid)
                    count += 1
        
        print(f"[AcronymNER] [OK] Motor listo. {count} variantes cargadas.")

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
                "label": "ACRONYM",
                "concept_id": concept_id
            })
        return predictions