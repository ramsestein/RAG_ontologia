"""Codificación SNOMED-CT usando RAG + GPT-4o"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from .rag import RAGRetriever


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
                entity=entity['full_span'],
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
        """RAG con threshold bajo y más opciones"""
        TOP_K = 20
        THRESHOLD = 1.65
        
        results = self.rag.retrieve(query, k=TOP_K)
        
        if not results:
            return "--- NO CODES FOUND ---\n"
        
        # Filtrar y ordenar por distancia
        filtered_results = [(concepto, narrativa, dist) for concepto, narrativa, dist in results if dist <= THRESHOLD]
        
        if not filtered_results:
            if verbose:
                print(f"[CODING]   -> RAG {context_type}: 0 resultados (dist > {THRESHOLD})")
            return "--- NO CODES FOUND ---\n"
        
        # Ordenar por mejor match (menor distancia)
        filtered_results = sorted(filtered_results, key=lambda x: x[2])
            
        if verbose:
            best_code, _, best_dist = filtered_results[0]
            print(f"[CODING]   -> RAG {context_type}: {len(filtered_results)} conceptos (mejor: {best_code}, dist: {best_dist:.3f})")

        # Contexto con top resultados
        context = f"\n--- {context_type} CODES for '{query}' ---\n"
        for idx, (concepto, narrativa, dist) in enumerate(filtered_results[:12], 1):
            context += f"OPCIÓN {idx}: CÓDIGO: {concepto} | {narrativa[:120]}\n"
            
        return context

