# src/NER/spacy_ner.py

import spacy
import warnings
from typing import List, Dict

class ScispaCyNER:
    """
    Un "worker" NER que usa un modelo scispaCy pre-entrenado.
    Esta clase es cargada dinámicamente por el script de benchmark.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Inicializa el worker cargando el modelo scispaCy especificado.
        
        Args:
            model_name (str): El nombre del modelo scispaCy a cargar 
                              (ej: "en_ner_bc5cdr_md"), pasado desde
                              el 'ner_registry.json'.
        """
        print(f"[NER Worker: ScispaCy] Cargando modelo: {model_name}...")
        
        # Ignorar warnings de transformers que a veces saltan
        warnings.filterwarnings("ignore", category=UserWarning)
        
        try:
            self.nlp = spacy.load(model_name)
        except IOError:
            print(f"[ERROR] Modelo '{model_name}' no encontrado.")
            print("Por favor, instálalo con:")
            print(f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/{model_name}-0.5.1.tar.gz")
            raise
            
        print(f"[NER Worker: ScispaCy] Modelo '{model_name}' cargado.")

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extrae entidades usando el modelo scispaCy cargado.
        
        Args:
            text (str): El texto de la nota a procesar.
            
        Returns:
            List[Dict]: Una lista de spans en el formato {'start': int, 'end': int}
                        requerido por el benchmark de IoU.
        """
        
        # Procesa el texto con el pipeline de scispaCy
        doc = self.nlp(text)
        
        predictions = []
        
        # Itera sobre las entidades (spans) encontradas
        for ent in doc.ents:
            # Convierte al formato estándar de tu benchmark
            predictions.append({
                "start": ent.start_char,
                "end": ent.end_char,
                # (scispaCy también da la etiqueta, ej: 'DISEASE')
                "label": ent.label_ 
            })
            
        return predictions