import pandas as pd
from flashtext import KeywordProcessor
from typing import List, Dict
from pathlib import Path
import re

class OntologyNER:
    """
    Worker NER Simbólico SOTA para CSVs con formato 'tripletas' de texto.
    """
    
    def __init__(self, ontology_path: str, min_length: int = 2, **kwargs):
        print(f"[NER Worker: Ontology] Cargando ontología desde {ontology_path}...")
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        try:
            df = pd.read_csv(ontology_path)
            
            # Detectar columnas
            col_narrativa = 'narrativa' if 'narrativa' in df.columns else 'term'
            
            # Lista de patrones basura a eliminar
            # (Basado en tu CSV: "230690007 tiene sinónimo ...")
            patterns_to_remove = [
                r'^\d+\s+tiene\s+sinónimo\s+',
                r'^\d+\s+tiene\s+término\s+preferido\s+',
                r'^\d+\s+se\s+define\s+como:\s*',
                r'^\d+\s+tiene\s+código\s+\d+', # Ignorar líneas de solo código
                r'\s*\(estructura corporal\)',
                r'\s*\(anomalía morfológica\)',
                r'\s*\(hallazgo\)',
                r'\s*\(célula\)',
                r'\s*\[como un todo\]'
            ]
            
            combined_pattern = re.compile('|'.join(patterns_to_remove), re.IGNORECASE)
            
            raw_terms = df[col_narrativa].dropna().unique().tolist()
            
            # --- FASE DE LIMPIEZA ---
            clean_terms = set()
            
            # Añadir términos críticos manuales (siempre útiles)
            clean_terms.update(["CT", "MRI", "MRA", "tPA", "NIHSS", "ASPECTS", "TICI", "LVO"])

            for raw_text in raw_terms:
                # Tu CSV tiene muchas frases en una sola celda a veces?
                # Si es una lista de frases separadas, iteramos.
                # Si es una sola frase por fila, procesamos.
                
                # Limpiar el string
                cleaned = combined_pattern.sub('', str(raw_text)).strip()
                
                # A veces quedan residuos o la frase era solo metadatos
                if not cleaned or cleaned.isdigit():
                    continue
                    
                # Validación extra: Longitud mínima
                if len(cleaned) >= min_length:
                    clean_terms.add(cleaned)
            
            print(f"[NER Worker: Ontology] Indexando {len(clean_terms)} términos limpios...")
            
            for term in clean_terms:
                self.keyword_processor.add_keyword(term)
                
            print(f"[NER Worker: Ontology] Motor listo.")
            
        except Exception as e:
            print(f"[ERROR] Fallo cargando ontología: {e}")
            import traceback
            traceback.print_exc()
            self.keyword_processor = None

    def extract_entities(self, text: str) -> List[Dict]:
        if not self.keyword_processor:
            return []
            
        # span_info=True devuelve [(keyword, start, end)]
        # Nota: FlashText a veces devuelve el nombre canónico si se configuró mapping.
        # Aquí añadimos keywords tal cual, así que devuelve el string encontrado.
        keywords_found = self.keyword_processor.extract_keywords(text, span_info=True)
        
        predictions = []
        for keyword, start, end in keywords_found:
            predictions.append({
                "start": start,
                "end": end,
                "span_text": text[start:end], 
                "label": "ONTOLOGY_EXACT"
            })
            
        return predictions