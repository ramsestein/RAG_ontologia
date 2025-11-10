"""Codificación SNOMED-CT usando RAG + GPT-4o (selección determinista + validación opcional)"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

# Setup paths for absolute imports
COMPONENTS_DIR = Path(__file__).parent.resolve()
SRC_DIR = COMPONENTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from components.rag import RAGRetriever


class SNOMEDCoder:
    """
    Codificador de entidades médicas usando SNOMED-CT.

    Cambios clave (solución robusta):
      - Selección determinista del mejor código basada en similitud (SapBERT/FAISS) en Python.
      - El LLM (GPT-4o) pasa a ser VALIDACIÓN opcional, y NUNCA puede forzar fallback si hay candidatos.
      - Umbrales y parámetros configurables por variables de entorno (sin parchear ficheros).
      - Uso real de system prompt en llamadas al LLM.
      - Render seguro de plantillas (sin usar str.format en prompts con llaves JSON).
    """

    FALLBACK_CODE = "404684003"   # Clinical finding (genérico)
    DEFAULT_ANATOMY = "12738006"  # Brain structure
    PRESENCE_MAP = {
        "presente": "52101004",    # Present
        "ausente": "272519000",    # Absent
        "incierto": "261665006"    # Unknown
    }

    def __init__(self, rag_retriever: RAGRetriever, openai_client: OpenAI, system_prompt: Optional[str] = None):
        self.rag = rag_retriever
        self.client = openai_client
        self.system_prompt = system_prompt or "You are a precise SNOMED-CT coding assistant."

        # Cargar prompts desde la nueva ubicación
        from config import load_prompt
        self.prompt_config = load_prompt("coding")
        self.filter_config = load_prompt("filter")

        # Config runtime por ENV (robusto para optimización)
        self.cfg = {
            "TOP_K": int(os.getenv("RAG_TOP_K", "30")),
            "THRESHOLD": float(os.getenv("RAG_THRESHOLD", "0.35")),
            "MAX_DISPLAY": int(os.getenv("RAG_MAX_DISPLAY", "12")),
            "QUERY_SUFFIX": os.getenv("RAG_QUERY_SUFFIX", "disorder finding"),
            "USE_LLM_VALIDATION": os.getenv("RAG_USE_LLM_VALIDATION", "false").lower() == "true",
            "LLM_MODEL": os.getenv("RAG_LLM_MODEL", "gpt-4o"),
            "LLM_TEMPERATURE": float(os.getenv("RAG_LLM_TEMPERATURE", "0.0")),
        }

    # --------------------------
    # API pública
    # --------------------------
    def code_entities(self, entities: List[Dict], verbose: bool = True) -> List[Dict]:
        """Codifica entidades usando SNOMED-CT con selección determinista (+ validación opcional)."""
        if verbose:
            print(f"[CODING] Codificando {len(entities)} entidades...")

        coded_entities = []
        for entity in entities:
            codes = self.assign_codes(
                entity=entity['span_text'],
                location=entity.get('anatomical_location', 'No especificado'),
                presence=entity.get('presence', 'presente'),
                verbose=verbose
            )
            coded_entity = {**entity, **codes}
            coded_entities.append(coded_entity)

        return coded_entities

    def assign_codes(self, entity: str, location: str, presence: str, verbose: bool = False) -> dict:
        """
        Selección determinista:
          1) Recupera candidatos para ENTITY (double query) y ANATOMY.
          2) Elige top-1 por similitud (ya filtrado por THRESHOLD).
          3) (Opcional) Valida con LLM, restringido a los candidatos.
        """
        # 1) Recuperación de candidatos
        ent_results = self._retrieve_candidates(query=entity, context_type="ENTITY", verbose=verbose)
        anat_results = self._retrieve_candidates(query=location, context_type="ANATOMY", verbose=verbose) \
            if location and location != "No especificado" else []

        # 2) Selección determinista top-1 o fallback/default
        entity_code = self._pick_top_code(ent_results, self.cfg["THRESHOLD"]) or self.FALLBACK_CODE
        anatomy_code = self._pick_top_code(anat_results, self.cfg["THRESHOLD"]) or self.DEFAULT_ANATOMY
        presence_code = self.PRESENCE_MAP.get(presence, self.PRESENCE_MAP["presente"])

        # 3) Validación opcional con LLM (nunca puede forzar fallback si hay candidatos)
        if self.cfg["USE_LLM_VALIDATION"]:
            if verbose:
                print(f"[CODING]   -> Validación con {self.cfg['LLM_MODEL']} (restringido a candidatos)...")

            contexto_entity = self._format_context("ENTITY", entity, ent_results)
            contexto_anatomy = self._format_context("ANATOMY", location, anat_results) if anat_results else "--- ANATOMY NOT SPECIFIED ---\n"
            valid_entity_list = [c for c, _, _ in ent_results]
            valid_anatomy_list = [c for c, _, _ in anat_results]

            # Render SEGURO del prompt (sin str.format)
            prompt = self._render_template(
                self.prompt_config["template"],
                {
                    "entity": entity,
                    "location": location,
                    "presence": presence,
                    "contexto_entity": contexto_entity,
                    "contexto_anatomy": contexto_anatomy,
                    "valid_entity_codes": json.dumps(valid_entity_list, ensure_ascii=False),
                    "valid_anatomy_codes": json.dumps(valid_anatomy_list, ensure_ascii=False)
                }
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.cfg["LLM_MODEL"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.cfg["LLM_TEMPERATURE"],
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)

                # Post-procesado restrictivo (no permitir inventar/fallback si hay candidatos)
                proposed_entity = str(result.get("entity_code", entity_code))
                proposed_anatomy = str(result.get("anatomy_code", anatomy_code))
                proposed_presence = str(result.get("presence_code", presence_code))

                if valid_entity_list:
                    if proposed_entity.isdigit() and proposed_entity in valid_entity_list:
                        entity_code = proposed_entity
                    else:
                        # Mantener determinista si LLM se sale del conjunto o propone fallback
                        pass
                else:
                    # No hay candidatos → aceptar fallback o lo que proponga (si es numérico)
                    if proposed_entity.isdigit():
                        entity_code = proposed_entity

                if valid_anatomy_list:
                    if proposed_anatomy.isdigit() and proposed_anatomy in valid_anatomy_list:
                        anatomy_code = proposed_anatomy
                    else:
                        pass
                else:
                    if proposed_anatomy.isdigit():
                        anatomy_code = proposed_anatomy

                if proposed_presence.isdigit():
                    presence_code = proposed_presence

            except Exception as e:
                if verbose:
                    print(f"[CODING]   [WARNING] Error en validación LLM: {e}")

        # Normalización final y logs
        if not str(entity_code).isdigit():
            if verbose:
                print(f"[CODING]   [WARNING] entity_code no numérico: '{entity_code}', usando fallback")
            entity_code = self.FALLBACK_CODE

        if not str(anatomy_code).isdigit():
            if verbose:
                print(f"[CODING]   [WARNING] anatomy_code no numérico: '{anatomy_code}', usando default")
            anatomy_code = self.DEFAULT_ANATOMY

        if not str(presence_code).isdigit():
            if verbose:
                print(f"[CODING]   [WARNING] presence_code no numérico: '{presence_code}', usando default")
            presence_code = self.PRESENCE_MAP["presente"]

        if verbose:
            print(f"[CODING]   [OK] Códigos: entity={entity_code}, anatomy={anatomy_code}, presence={presence_code}")

        return {
            "entity_code": str(entity_code),
            "anatomy_code": str(anatomy_code),
            "presence_code": str(presence_code)
        }

    # --------------------------
    # Helpers internos
    # --------------------------
    def _retrieve_candidates(self, query: str, context_type: str, verbose: bool) -> List[Tuple[str, str, float]]:
        """
        Recupera, deduplica y filtra candidatos por THRESHOLD.
        Para ENTITY aplica double-query (query y query+suffix).
        """
        if not query or query == "No especificado":
            return []

        TOP_K = self.cfg["TOP_K"]
        THRESHOLD = self.cfg["THRESHOLD"]

        if context_type == "ENTITY":
            results_main = self.rag.retrieve(query, k=TOP_K)
            query_clinical = f"{query} {self.cfg['QUERY_SUFFIX']}".strip()
            results_clinical = self.rag.retrieve(query_clinical, k=TOP_K)

            combined = {}
            for concepto, narrativa, sim in (results_main + results_clinical):
                # Mantener la mayor similitud por concepto
                if concepto not in combined or sim > combined[concepto][1]:
                    combined[concepto] = (narrativa, sim)

            results = [(c, n, s) for c, (n, s) in combined.items()]
        else:
            results = self.rag.retrieve(query, k=min(TOP_K, 15))

        # Filtrado y orden
        filtered = [(c, n, s) for c, n, s in results if s >= THRESHOLD]
        filtered.sort(key=lambda x: x[2], reverse=True)

        if verbose:
            if filtered:
                best_code, _, best_sim = filtered[0]
                print(f"[CODING]   -> RAG {context_type} [MULTI-Q]: {len(filtered)} conceptos (mejor: {best_code}, dist: {best_sim:.3f})")
            else:
                print(f"[CODING]   -> RAG {context_type}: 0 resultados (dist < {THRESHOLD})")

        return filtered

    def _pick_top_code(self, results: List[Tuple[str, str, float]], threshold: float) -> Optional[str]:
        """Elige top-1 si existe y supera threshold. Si no, None."""
        if not results:
            return None
        best_code, _, best_sim = results[0]
        return best_code if best_sim >= threshold and str(best_code).isdigit() else None

    def _format_context(self, context_type: str, query: str, results: List[Tuple[str, str, float]]) -> str:
        """Construye el bloque de contexto para el LLM (solo informativo)."""
        if not results:
            return f"--- {context_type} CODES for '{query}' ---\n--- NO CODES FOUND ---\n"

        MAX_DISPLAY = self.cfg["MAX_DISPLAY"]
        context = f"\n--- {context_type} CODES for '{query}' ---\n"
        for idx, (concepto, narrativa, sim) in enumerate(results[:MAX_DISPLAY], 1):
            context += f"OPCIÓN {idx} [SIM: {sim:.2f}]: CÓDIGO: {concepto} | {narrativa[:120]}\n"
        return context

    def _render_template(self, template: str, variables: Dict[str, str]) -> str:
        """
        Render muy simple para prompts con JSON: sustituye solo {clave} conocidas,
        sin interpretar el resto de llaves que forman parte del texto.
        """
        s = template
        for k, v in variables.items():
            s = s.replace("{" + k + "}", str(v))
        return s
