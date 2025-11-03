"""
RAG+GPT Pipeline - Orquestador principal
Implementa el pipeline completo de NER -> RAG -> Coding
"""

import pandas as pd
from typing import List, Dict
import sys
from pathlib import Path

# Configurar imports absolutos
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.core.ner import NERExtractor
from strategies.rag_gpt.core.rag import RAGRetriever
from strategies.rag_gpt.core.coding import SNOMEDCoder
from strategies.rag_gpt.utils.config import (
    load_prompt,
    setup_openai_client,
    get_model_config,
    get_assets_dir
)
from strategies.rag_gpt.utils.text_processing import (
    find_span_in_text,
    find_all_spans_in_text,
    find_exact_span,
    find_exact_span_near,
    find_first_case_insensitive
)


class RAGGPTPipeline:
    """
    Pipeline modular para extracción y codificación de entidades médicas

    Arquitectura:
        1. NER: Extracción de entidades (GPT-4o) con spans VERBATIM + offsets
        2. RAG: Recuperación de conceptos SNOMED (FAISS)
        3. Coding: Codificación SNOMED-CT (selección determinista + validación opcional)
        4. Span Matching: Uso de offsets del NER con corrección exacta (sin expansión masiva)
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
        self.ner = NERExtractor(self.client, ner_prompt, self.model_config, system_prompt=system_prompt)
        self.coder = SNOMEDCoder(self.rag, self.client, system_prompt=system_prompt)

        if verbose:
            print("[OK] Pipeline inicializado correctamente")
            print("="*80)

    def _chunk_text(self, text: str):
        """Divide texto en chunks con overlap y devuelve (chunk_text, base_offset)."""
        chunk_size = 3000
        overlap = 300

        if len(text) <= chunk_size:
            return [(text, 0)]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((text[start:end], start))
            if end >= len(text):
                break
            start = end - overlap

        if self.verbose:
            print(f"[CHUNKING] {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

        return chunks


    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Elimina duplicados SOLO si son la misma ocurrencia (mismo start/end).
        """
        seen = set()
        unique: List[Dict] = []

        for e in entities:
            start = e.get("start")
            end = e.get("end")
            key = (
                e.get('full_span', e['span_text']),
                e.get('anatomical_location', ''),
                e.get('presence', ''),
                start if isinstance(start, int) else None,
                end if isinstance(end, int) else None
            )
            if key not in seen:
                seen.add(key)
                unique.append(e)

        if self.verbose and len(entities) > len(unique):
            print(f"[DEDUP] {len(entities)} -> {len(unique)} entidades (por ocurrencia)")

        return unique

    def process_note(self, text: str, note_id: int = None) -> List[Dict]:
        """
        Procesa una nota médica completa

        Pipeline:
            text -> Chunking -> NER (por chunk) -> Dedup -> RAG+Coding -> Span Matching

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

            chunks = self._chunk_text(text)

            all_entities = []
            for i, (chunk, base) in enumerate(chunks):
                if self.verbose and len(chunks) > 1:
                    print(f"\n[NER] Procesando chunk {i+1}/{len(chunks)} (base={base})...")

                chunk_entities = self.ner.extract_entities(chunk)

                # FIX (1): rebasa offsets relativos del chunk a offsets globales del documento
                for e in chunk_entities:
                    if isinstance(e.get("start"), int):
                        e["start"] += base
                    if isinstance(e.get("end"), int):
                        e["end"] += base

                all_entities.extend(chunk_entities)


        if not all_entities:
            if self.verbose:
                print("[WARNING] No se detectaron entidades")
            return []

        # Paso 1.5: Deduplicación - por ocurrencia (mantiene instancias con distintos offsets)
        entities = self._deduplicate_entities(all_entities)

        # Paso 2: RAG + Coding - Codificar entidades (determinista + validación opcional)
        coded_entities = self.coder.code_entities(entities, verbose=self.verbose)

        # Paso 3: Span Matching - Localización estricta (sin expansión masiva)
        final_entities = self._locate_spans(coded_entities, text)

        if self.verbose:
            print(f"\n[OK] Procesamiento completado: {len(final_entities)} entidades")

        return final_entities

    def _locate_spans(self, entities: List[Dict], text: str) -> List[Dict]:
        """
        Localiza spans con política ESTRICTA orientada a 'exact match':
          1) Si la entidad trae start/end válidos y el snippet coincide EXACTO con full_span → aceptar.
          2) Si hay offsets pero el snippet no cuadra, intentamos corregir cerca del offset con búsqueda EXACTA.
          3) Si no hay offsets, buscamos UNA única ocurrencia EXACTA en todo el texto.
          4) Como último recurso, usamos coincidencia case-insensitive (UNA sola ocurrencia).
          5) NO expandimos a todas las ocurrencias (evita FPs y offsets no anotados).
        """
        located_entities: List[Dict] = []

        for entity in entities:
            core_entity = entity['span_text']
            full_span = (entity.get('full_span') or core_entity) or core_entity

            start = entity.get("start")
            end = entity.get("end")

            # (1) Usar offsets proporcionados si son válidos y EXACTOS (case-sensitive)
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                snippet = text[start:end]
                if snippet == full_span:
                    ent = dict(entity)
                    ent['start'] = start
                    ent['end'] = end
                    ent['span_text_real'] = snippet
                    located_entities.append(ent)
                    continue
                # (2) Corregir cerca del offset con búsqueda EXACTA
                nearby = find_exact_span_near(full_span, text, approx_start=start, window=80)
                if nearby:
                    s2, e2 = nearby
                    ent = dict(entity)
                    ent['start'] = s2
                    ent['end'] = e2
                    ent['span_text_real'] = text[s2:e2]
                    located_entities.append(ent)
                    continue
                # Intentar con el core_entity si el full_span falla
                nearby_core = find_exact_span_near(core_entity, text, approx_start=start, window=80)
                if nearby_core:
                    s3, e3 = nearby_core
                    ent = dict(entity)
                    ent['start'] = s3
                    ent['end'] = e3
                    ent['span_text_real'] = text[s3:e3]
                    located_entities.append(ent)
                    continue
                # Últimos recursos: global exacta
                global_match = find_exact_span(full_span, text)
                if global_match:
                    s4, e4 = global_match
                    ent = dict(entity)
                    ent['start'] = s4
                    ent['end'] = e4
                    ent['span_text_real'] = text[s4:e4]
                    located_entities.append(ent)
                    continue
                # Case-insensitive global (una única ocurrencia)
                ci = find_first_case_insensitive(full_span, text)
                if ci:
                    s5, e5 = ci
                    ent = dict(entity)
                    ent['start'] = s5
                    ent['end'] = e5
                    ent['span_text_real'] = text[s5:e5]
                    located_entities.append(ent)
                    continue

                if self.verbose:
                    print(f"[SPAN] Descarta entidad por offsets no fiables y sin match exacto: '{full_span[:40]}'")
                continue

            # (3) Sin offsets → buscar UNA ocurrencia exacta global
            exact_global = find_exact_span(full_span, text)
            if exact_global:
                s, e = exact_global
                ent = dict(entity)
                ent['start'] = s
                ent['end'] = e
                ent['span_text_real'] = text[s:e]
                located_entities.append(ent)
                continue

            # (4) Último recurso: case-insensitive UNA ocurrencia
            ci_global = find_first_case_insensitive(full_span, text)
            if ci_global:
                s2, e2 = ci_global
                ent = dict(entity)
                ent['start'] = s2
                ent['end'] = e2
                ent['span_text_real'] = text[s2:e2]
                located_entities.append(ent)
                continue

            # (5) No forzar regex flexible ni expansión
            if self.verbose:
                print(f"[SPAN] Sin offsets y sin match exacto: descarta '{full_span[:40]}'")

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
