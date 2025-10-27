"""
RAG+GPT Pipeline - Orquestador principal
Implementa el pipeline completo de NER -> RAG -> Coding
"""

import pandas as pd
from typing import List, Dict

from .core.ner import NERExtractor
from .core.rag import RAGRetriever
from .core.coding import SNOMEDCoder
from .utils.config import (
    load_prompt,
    setup_openai_client,
    get_model_config,
    get_assets_dir
)
from .utils.text_processing import find_span_in_text


class RAGGPTPipeline:
    """
    Pipeline modular para extracción y codificación de entidades médicas
    
    Arquitectura:
        1. NER: Extracción de entidades (GPT-4o)
        2. RAG: Recuperación de conceptos SNOMED (FAISS)
        3. Coding: Codificación SNOMED-CT (RAG + GPT-4o)
        4. Span Matching: Localización de spans en texto
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa el pipeline completo
        
        Args:
            verbose: Si True, imprime logs detallados
        """
        self.verbose = verbose
        
        if verbose:
            print("="*80)
            print("RAG+GPT Pipeline - Inicialización")
            print("="*80)
        
        # 1. Configuración
        self.client = setup_openai_client()
        self.model_config = get_model_config()
        assets_dir = get_assets_dir()
        
        # 2. Cargar prompts
        ner_prompt = load_prompt("ner_prompt")
        coding_prompt = load_prompt("coding_prompt")
        system_prompt_data = load_prompt("system_prompt")
        system_prompt = system_prompt_data['content']
        
        # 3. Inicializar componentes
        self.rag = RAGRetriever(assets_dir)
        self.ner = NERExtractor(self.client, ner_prompt, self.model_config)
        self.coder = SNOMEDCoder(
            self.client, 
            self.rag, 
            coding_prompt, 
            system_prompt,
            self.model_config
        )
        
        if verbose:
            print("[OK] Pipeline inicializado correctamente")
            print("="*80)
    
    def process_note(self, text: str, note_id: int = None) -> List[Dict]:
        """
        Procesa una nota médica completa
        
        Pipeline:
            text -> NER -> entities -> RAG+Coding -> coded_entities
        
        Args:
            text: Texto de la nota médica
            note_id: ID de la nota (opcional, para logging)
            
        Returns:
            Lista de entidades codificadas con spans localizados
        """
        if self.verbose and note_id:
            print(f"\n{'='*80}")
            print(f"Procesando nota {note_id}")
            print(f"{'='*80}")
        
        # Paso 1: NER - Extraer entidades
        entities = self.ner.extract_entities(text)
        
        if not entities:
            if self.verbose:
                print("[WARNING] No se detectaron entidades")
            return []
        
        # Paso 2: RAG + Coding - Codificar entidades
        coded_entities = self.coder.code_entities(entities, verbose=self.verbose)
        
        # Paso 3: Span Matching - Localizar en texto
        final_entities = self._locate_spans(coded_entities, text)
        
        if self.verbose:
            print(f"\n[OK] Procesamiento completado: {len(final_entities)} entidades")
        
        return final_entities
    
    def _locate_spans(self, entities: List[Dict], text: str) -> List[Dict]:
        """
        Localiza los spans en el texto original
        
        Args:
            entities: Entidades codificadas
            text: Texto original
            
        Returns:
            Entidades con start/end positions
        """
        located_entities = []
        last_search_idx = {}
        
        for entity in entities:
            span_text = entity['span_text']
            
            # Determinar desde dónde buscar
            start_from = last_search_idx.get(span_text, 0)
            
            # Buscar span
            result = find_span_in_text(span_text, text, start_from)
            
            if result:
                start, end = result
                last_search_idx[span_text] = end
                
                # Añadir posiciones
                entity['start'] = start
                entity['end'] = end
                entity['span_text_real'] = text[start:end]
                
                located_entities.append(entity)
            else:
                if self.verbose:
                    print(f"[WARNING] No se encontró span: '{span_text[:50]}'")
        
        return located_entities
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa múltiples notas y genera predicciones
        
        Args:
            notes_df: DataFrame con columnas 'note_id' y 'text'
            
        Returns:
            DataFrame con predicciones en formato de benchmark
        """
        print(f"[Pipeline] Procesando {len(notes_df)} notas...")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            # Procesar nota
            entities = self.process_note(text, note_id)
            
            # Convertir a formato de predicción
            for entity in entities:
                predictions.append({
                    'note_id': note_id,
                    'start': entity['start'],
                    'end': entity['end'],
                    'concept_id': str(entity['entity_code']),
                    'span_text': entity.get('span_text_real', entity['span_text']),
                    'confidence': 0.85,
                    'entity_description': entity.get('entity_description', ''),
                    'anatomy_code': entity.get('anatomy_code', ''),
                    'presence_code': entity.get('presence_code', ''),
                    'llm_used': 'GPT-4o'
                })
        
        print(f"[Pipeline] [OK] Completado: {len(predictions)} predicciones generadas")
        
        return pd.DataFrame(predictions)
