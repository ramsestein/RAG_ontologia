"""Codificación SNOMED-CT usando RAG + GPT-4o"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from openai import OpenAI

# Configurar imports absolutos
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCHMARK_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.core.rag import RAGRetriever


class SNOMEDCoder:
    """Codificador de entidades médicas usando SNOMED-CT"""
    
    FALLBACK_CODE = "404684003"
    DEFAULT_ANATOMY = "12738006"
    PRESENCE_MAP = {
        "presente": "52101004",
        "ausente": "272519000",
        "incierto": "261665006"
    }
    
    def __init__(self, rag_retriever, openai_client):
        self.rag = rag_retriever
        self.client = openai_client
        
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "coding_prompt.json")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_config = json.load(f)
        
        filter_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "filter_prompt.json")
        with open(filter_path, 'r', encoding='utf-8') as f:
            self.filter_config = json.load(f)
    
    def code_entities(self, entities: List[Dict], verbose: bool = True) -> List[Dict]:
        """Codifica entidades usando SNOMED-CT con contexto RAG"""
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
        """RAG + GPT single-phase"""
        contexto_entity = self._build_context(entity, "ENTITY", verbose)
        contexto_anatomy = "--- ANATOMY NOT SPECIFIED ---\n"
        
        if location and location != "No especificado":
            contexto_anatomy = self._build_context(location, "ANATOMY", verbose)
        
        if verbose:
            print(f"[CODING]   -> Consultando GPT-4o...")
        
        prompt = self.prompt_config["template"].format(
            entity=entity,
            location=location,
            presence=presence,
            contexto_entity=contexto_entity,
            contexto_anatomy=contexto_anatomy
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        entity_code = result.get("entity_code", self.FALLBACK_CODE)
        anatomy_code = result.get("anatomy_code", self.DEFAULT_ANATOMY)
        presence_code = result.get("presence_code", self.PRESENCE_MAP.get(presence, self.PRESENCE_MAP["presente"]))
        
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
    
    def _build_context(self, query: str, context_type: str, verbose: bool) -> str:
        """Double-query RAG - SapBERT cosine similarity"""
        TOP_K = 15
        THRESHOLD = 0.3  # Cosine similarity: higher threshold = more selective
        MAX_DISPLAY = 12
        
        # Double-query for entity codes
        if context_type == "ENTITY":
            # Query 1: Original
            results_main = self.rag.retrieve(query, k=TOP_K)
            
            # Query 2: Clinical context
            query_clinical = f"{query} disorder finding"
            results_clinical = self.rag.retrieve(query_clinical, k=TOP_K)
            
            # Combinar y deduplicar - KEEP HIGHER SIMILARITY (cosine)
            combined_results = {}
            for concepto, narrativa, dist in (results_main + results_clinical):
                if concepto not in combined_results or dist > combined_results[concepto][1]:  # FIXED: > for cosine
                    combined_results[concepto] = (narrativa, dist)
            
            results = [(concepto, narrativa, dist) for concepto, (narrativa, dist) in combined_results.items()]
        else:
            results = self.rag.retrieve(query, k=15)
        
        if not results:
            return "--- NO CODES FOUND ---\n"
        
        # Filtrar y ordenar - FIXED: >= for cosine similarity
        filtered_results = [(concepto, narrativa, dist) for concepto, narrativa, dist in results if dist >= THRESHOLD]
        
        if not filtered_results:
            if verbose:
                print(f"[CODING]   -> RAG {context_type}: 0 resultados (dist < {THRESHOLD})")
            return "--- NO CODES FOUND ---\n"
        
        # Ordenar DESCENDING (mayor score = mejor match con cosine)
        filtered_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)
            
        if verbose:
            best_code, _, best_dist = filtered_results[0]
            print(f"[CODING]   -> RAG {context_type} [MULTI-Q]: {len(filtered_results)} conceptos (mejor: {best_code}, dist: {best_dist:.3f})")

        # Contexto simple - NO TAGS
        context = f"\n--- {context_type} CODES for '{query}' ---\n"
        for idx, (concepto, narrativa, dist) in enumerate(filtered_results[:MAX_DISPLAY], 1):
            context += f"OPCIÓN {idx} [SIM: {dist:.2f}]: CÓDIGO: {concepto} | {narrativa[:120]}\n"
            
        return context

