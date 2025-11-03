"""
RAG+GPT Pipeline - Orquestador principal
Implementa el pipeline completo de NER -> RAG -> Coding
"""

import os
import pandas as pd
from typing import List, Dict, Tuple
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
    get_assets_dir,
    EVAL_OFFSETS,  # <— NUEVO
)
from strategies.rag_gpt.utils.text_processing import (
    find_span_in_text,
    find_all_spans_in_text,
    find_exact_span,
    find_exact_span_near,
    find_first_case_insensitive,
    tighten_span_boundaries,
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
            print("=" * 80)
            print("RAG+GPT Pipeline - Inicialización")
            print("=" * 80)

        # 1. Configuración
        self.client = setup_openai_client()
        self.model_config = get_model_config()
        assets_dir = get_assets_dir()

        # Flag de recorte de bordes (activado por defecto)
        self.span_tighten = os.getenv("RAG_SPAN_TIGHTEN", "true").lower() == "true"

        # 2. Cargar prompts
        ner_prompt = load_prompt("ner_prompt")
        coding_prompt = load_prompt("coding_prompt")
        system_prompt_data = load_prompt("system_prompt")
        system_prompt = system_prompt_data["content"]

        # 3. Inicializar componentes
        self.rag = RAGRetriever(assets_dir)
        self.ner = NERExtractor(self.client, ner_prompt, self.model_config, system_prompt=system_prompt)
        self.coder = SNOMEDCoder(self.rag, self.client, system_prompt=system_prompt)

        if verbose:
            print("[OK] Pipeline inicializado correctamente")
            print("=" * 80)

    # ----------------------------------------------------------------------------------
    # Normalización con mapeo de offsets (CRLF -> LF) SIN desalinear respecto al original
    # ----------------------------------------------------------------------------------
    def _normalize_text_with_mapping(self, original_text: str) -> Tuple[str, List[int]]:
        """
        Devuelve:
          - texto_normalizado: reemplaza \r\n y \r por \n
          - mapping: lista de longitud len(texto_normalizado) que, para cada índice i
                     del texto normalizado, da el índice correspondiente en el texto original.
        También añadimos un "pivote" final implícito (len(original)) para calcular 'end'.
        """
        norm_chars: List[str] = []
        mapping: List[int] = []

        i = 0
        n = len(original_text)
        while i < n:
            ch = original_text[i]
            if ch == "\r":
                # Caso CRLF -> un solo '\n'
                if i + 1 < n and original_text[i + 1] == "\n":
                    norm_chars.append("\n")
                    mapping.append(i)  # mapeamos el '\n' normalizado al inicio del par \r\n
                    i += 2
                else:
                    # CR suelto -> normalizamos a '\n'
                    norm_chars.append("\n")
                    mapping.append(i)
                    i += 1
            else:
                norm_chars.append(ch)
                mapping.append(i)
                i += 1

        normalized = "".join(norm_chars)
        return normalized, mapping

    def _map_norm_span_to_original(self, s_norm: int, e_norm: int, mapping: List[int], orig_len: int) -> Tuple[int, int]:
        """
        Convierte un span [s_norm, e_norm) del texto normalizado a offsets del texto original.
        Regla: start_orig = mapping[s_norm] ; end_orig = mapping[e_norm-1] + 1 (si e_norm > 0)
        """
        if not mapping:
            return s_norm, e_norm  # sin normalización
        # start
        if s_norm < 0:
            s_norm = 0
        if s_norm >= len(mapping):
            start_orig = orig_len
        else:
            start_orig = mapping[s_norm]

        # end
        if e_norm <= 0:
            end_orig = 0
        elif e_norm - 1 >= len(mapping):
            end_orig = orig_len
        else:
            end_orig = mapping[e_norm - 1] + 1

        # sanity
        if end_orig < start_orig:
            end_orig = start_orig
        return start_orig, end_orig

    # ----------------------------------------------------------------------------------

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
                e.get("full_span", e["span_text"]),
                e.get("anatomical_location", ""),
                e.get("presence", ""),
                start if isinstance(start, int) else None,
                end if isinstance(end, int) else None,
            )
            if key not in seen:
                seen.add(key)
                unique.append(e)

        if self.verbose and len(entities) > len(unique):
            print(f"[DEDUP] {len(entities)} -> {len(unique)} entidades (por ocurrencia)")

        return unique

    # Helper para añadir entidad localizada con "tighten"
    def _append_located(self, located_entities: List[Dict], base_entity: Dict, text: str, s: int, e: int):
        if self.span_tighten:
            s, e = tighten_span_boundaries(text, s, e)
        ent = dict(base_entity)
        ent["start"] = s
        ent["end"] = e
        ent["span_text_real"] = text[s:e]
        located_entities.append(ent)

    def process_note(self, text: str, note_id: int = None) -> List[Dict]:
        """
        Procesa una nota médica completa

        Pipeline:
            original_text -> normalización+mapa -> Chunking -> NER -> Dedup -> RAG+Coding -> Span Matching
            (y finalmente mapeo inverso de offsets al texto original)
        """
        if self.verbose and note_id:
            print(f"\n{'=' * 80}")
            print(f"Procesando nota {note_id}")
            print(f"{'=' * 80}")

        # --- Normalización con mapeo (CRLF/LF) ---
        original_text = text
        text_norm, mapping = self._normalize_text_with_mapping(original_text)
        text = text_norm  # a partir de aquí trabajamos SIEMPRE sobre el normalizado

        # Chunking SIEMPRE (antes estaba dentro de un if)
        chunks = self._chunk_text(text)

        all_entities: List[Dict] = []
        for i, (chunk, base) in enumerate(chunks):
            if self.verbose and len(chunks) > 1:
                print(f"\n[NER] Procesando chunk {i + 1}/{len(chunks)} (base={base})...")

            chunk_entities = self.ner.extract_entities(chunk)

            # Rebasa offsets relativos del chunk a offsets globales del documento normalizado
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

        # Paso 3: Span Matching - Localización estricta sobre el TEXTO NORMALIZADO
        final_entities_norm = self._locate_spans(coded_entities, text)

        # Paso 4: MAPEO INVERSO de offsets al TEXTO ORIGINAL (clave para no romper el benchmark)
        final_entities: List[Dict] = []
        for ent in final_entities_norm:
            s_norm = ent["start"]
            e_norm = ent["end"]
            s_orig, e_orig = self._map_norm_span_to_original(s_norm, e_norm, mapping, len(original_text))

            ent_out = dict(ent)
            ent_out["start"] = s_orig
            ent_out["end"] = e_orig
            ent_out["span_text_real"] = original_text[s_orig:e_orig]  # texto original exacto
            final_entities.append(ent_out)

        if self.verbose:
            print(f"\n[OK] Procesamiento completado: {len(final_entities)} entidades")

        return final_entities

    def _locate_spans(self, entities: List[Dict], text: str) -> List[Dict]:
        """
        Localiza spans con política ESTRICTA orientada a 'exact match' y recorte opcional:
          1) Si la entidad trae start/end válidos y el snippet coincide EXACTO con full_span → aceptar (y recortar bordes).
          2) Si hay offsets pero el snippet no cuadra, intentamos corregir cerca del offset con búsqueda EXACTA.
          3) Si no hay offsets, buscamos UNA única ocurrencia EXACTA en todo el texto.
          4) Como último recurso, usamos coincidencia case-insensitive (UNA sola ocurrencia).
          5) NO expandimos a todas las ocurrencias (evita FPs y offsets no anotados).
        """
        located_entities: List[Dict] = []

        for entity in entities:
            core_entity = entity["span_text"]
            full_span = (entity.get("full_span") or core_entity) or core_entity

            start = entity.get("start")
            end = entity.get("end")

            # (1) Usar offsets proporcionados si son válidos y EXACTOS (case-sensitive)
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                snippet = text[start:end]
                if snippet == full_span:
                    self._append_located(located_entities, entity, text, start, end)
                    continue
                # (2) Corregir cerca del offset con búsqueda EXACTA
                nearby = find_exact_span_near(full_span, text, approx_start=start, window=80)
                if nearby:
                    s2, e2 = nearby
                    self._append_located(located_entities, entity, text, s2, e2)
                    continue
                # Intentar con el core_entity si el full_span falla
                nearby_core = find_exact_span_near(core_entity, text, approx_start=start, window=80)
                if nearby_core:
                    s3, e3 = nearby_core
                    self._append_located(located_entities, entity, text, s3, e3)
                    continue
                # Últimos recursos: global exacta
                global_match = find_exact_span(full_span, text)
                if global_match:
                    s4, e4 = global_match
                    self._append_located(located_entities, entity, text, s4, e4)
                    continue
                # Case-insensitive global (una única ocurrencia)
                ci = find_first_case_insensitive(full_span, text)
                if ci:
                    s5, e5 = ci
                    self._append_located(located_entities, entity, text, s5, e5)
                    continue

                if self.verbose:
                    print(f"[SPAN] Descarta entidad por offsets no fiables y sin match exacto: '{full_span[:40]}'")
                continue

            # (3) Sin offsets → buscar UNA ocurrencia exacta global
            exact_global = find_exact_span(full_span, text)
            if exact_global:
                s, e = exact_global
                self._append_located(located_entities, entity, text, s, e)
                continue

            # (4) Último recurso: case-insensitive UNA ocurrencia
            ci_global = find_first_case_insensitive(full_span, text)
            if ci_global:
                s2, e2 = ci_global
                self._append_located(located_entities, entity, text, s2, e2)
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
            note_id = row["note_id"]
            text = row["text"]

            # Procesar nota
            entities = self.process_note(text, note_id)

            # Convertir a formato de predicción con política de offsets del benchmark
            for entity in entities:
                start_out = int(entity["start"])
                end_out = int(entity["end"])

                # Ajuste de inclusividad del 'end'
                if EVAL_OFFSETS.get("end_inclusive", False):
                    # end es exclusivo internamente -> inclusivo para el benchmark
                    end_out = max(0, end_out - 1)

                # Base 1 vs base 0
                if int(EVAL_OFFSETS.get("base", 0)) == 1:
                    start_out += 1
                    end_out += 1

                predictions.append({
                    "note_id": note_id,
                    "start": start_out,
                    "end": end_out,
                    "concept_id": str(entity["entity_code"]),
                    "span_text": entity.get("span_text_real", entity["span_text"]),
                    "confidence": 0.85,
                    "entity_description": entity.get("entity_description", ""),
                    "anatomy_code": entity.get("anatomy_code", ""),
                    "presence_code": entity.get("presence_code", ""),
                    "llm_used": "GPT-4o",
                })

        print(f"[Pipeline] [OK] Completado: {len(predictions)} predicciones generadas")

        return pd.DataFrame(predictions)
