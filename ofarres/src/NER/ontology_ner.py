import json
import spacy
from flashtext import KeywordProcessor
from typing import List, Dict, Set
from pathlib import Path

class OntologyNER:
    """
    Worker NER SOTA Genérico.
    
    CORRECCIONES:
    1. min_term_len = 2 (Recupera 'CT', 'TIA').
    2. Usa Stopwords para filtrar basura corta ('of', 'in').
    3. Ignora núcleos genéricos ('Artery', 'Disease') para evitar colisiones.
    """
    
    def __init__(self, ontology_path: str = None, min_term_len: int = 2, **kwargs):
        if not ontology_path:
            ontology_path = Path(__file__).parent.parent.parent / "ontology" / "multilingual_ontology.json"
        else:
            ontology_path = Path(ontology_path)
            if not ontology_path.is_absolute():
                ontology_path = Path(__file__).parent.parent.parent / ontology_path
            
        self.ontology_path = ontology_path
        self.min_term_len = min_term_len # RESTAURADO A 2
        
        print(f"[OntologyNER] Loading ontology from: {self.ontology_path}")
        
        # Carga de Stopwords para seguridad con palabras cortas
        try:
            self.nlp = spacy.blank("en")
            self.stopwords = self.nlp.Defaults.stop_words
        except:
            self.stopwords = {"of", "in", "at", "on", "to", "by", "is", "it", "no", "us"}

        # Configuración FlashText
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self.keyword_processor.add_non_word_boundary('-') 
        self.keyword_processor.add_non_word_boundary('/')
        
        self._load_and_expand()

    def _generate_generic_variations(self, term: str) -> Set[str]:
        variations = {term}
        clean_term = term.strip()
        
        # 1. Pluralización
        if clean_term.lower().endswith('s'):
            variations.add(clean_term[:-1]) 
        else:
            variations.add(clean_term + 's') 
            
        # 2. Morfología Médica (Generic Suffix Stripping)
        if clean_term.endswith("tion"): # Infarction -> Infarct
            base = clean_term[:-4]
            variations.add(base + "t") 
            variations.add(base + "ted")
            
        if clean_term.endswith("rhage"): # Hemorrhage -> Hemorrhagic
            variations.add(clean_term[:-1] + "gic")
            
        if clean_term.endswith("sis"): # Stenosis -> Stenotic
            variations.add(clean_term[:-3] + "tic")

        if clean_term.endswith("itis"): # Arthritis -> Arthritic
            variations.add(clean_term[:-4] + "itic")
            
        if clean_term.endswith("ia"): # Ischemia -> Ischemic
            variations.add(clean_term[:-2] + "ic")

        # 3. Head Word Extraction (Mejorado)
        parts = clean_term.split()
        if len(parts) > 1:
            last_word = parts[-1]
            
            # LISTA NEGRA DE NÚCLEOS: Palabras que son demasiado genéricas para ser indexadas solas.
            ignored_heads = {
                "left", "right", "acute", "chronic", "mild", "severe", "upper", "lower",
                "artery", "vein", "nerve", "muscle", "ligament", "bone", # Anatomía genérica
                "disease", "disorder", "syndrome", "condition", "problem", # Patología genérica
                "sign", "symptom", "finding"
            }
            
            # Indexamos la última palabra SOLO si no es genérica y es lo suficientemente larga
            # Ej: "Basilar Artery" -> Ignora "Artery".
            # Ej: "Atrial Fibrillation" -> Indexa "Fibrillation".
            if len(last_word) >= 4 and last_word.lower() not in ignored_heads:
                variations.add(last_word)
                variations.add(last_word + 's')
                if last_word.endswith("tion"):
                    variations.add(last_word[:-4] + "t")

        # 4. Puntuación
        if '-' in clean_term:
            variations.add(clean_term.replace('-', ' '))
            
        return variations

    def _load_and_expand(self):
        if not self.ontology_path.exists():
            return

        try:
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        term_count = 0
        
        for entry in data:
            cid = entry['concept_id']
            all_raw_terms = set()
            
            for lang_code in ["es", "en", "ca"]:
                terms = entry.get("languages", {}).get(lang_code, {}).get("terms", [])
                if isinstance(terms, list):
                    all_raw_terms.update(terms)
                elif isinstance(terms, str):
                    all_raw_terms.add(terms)

            for raw_term in all_raw_terms:
                if not raw_term: continue
                
                # Check inicial de longitud
                if len(raw_term) < self.min_term_len:
                    continue
                
                expanded_set = self._generate_generic_variations(raw_term)
                
                for final_term in expanded_set:
                    # FILTRO FINAL: Longitud y Stopwords
                    if len(final_term) < self.min_term_len:
                        continue
                    
                    # Si es una palabra corta, verificar que no sea stopword ("in", "on")
                    if final_term.lower() in self.stopwords:
                        continue
                        
                    self.keyword_processor.add_keyword(final_term, cid)
                    term_count += 1
                        
        print(f"[OntologyNER] ✅ Engine Ready. {term_count} terms indexed.")

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