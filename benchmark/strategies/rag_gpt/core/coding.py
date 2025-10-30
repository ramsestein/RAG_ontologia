"""
Módulo de Codificación SNOMED-CT usando RAG + GPT-4o
"""

import json
import re
from typing import List, Dict
from openai import OpenAI
from .rag import RAGRetriever


class SNOMEDCoder:
    """Codificador de entidades médicas usando SNOMED-CT"""
    
    def __init__(self, client: OpenAI, rag_retriever: RAGRetriever, 
                 prompt_config: Dict, system_prompt: str, model_config: Dict):
        """
        Args:
            client: Cliente de OpenAI
            rag_retriever: Sistema RAG para recuperación de conceptos
            prompt_config: Configuración del prompt de codificación
            system_prompt: Prompt de sistema
            model_config: Configuración del modelo GPT
        """
        self.client = client
        self.rag = rag_retriever
        self.prompt_template = prompt_config['template']
        self.system_prompt = system_prompt
        self.model_config = model_config
        
        # Mapeo de presencia a código SNOMED
        self.presence_map = {
            "presente": "52101004",
            "ausente": "272519000",
            "incierto": "261665006"
        }
    
    def code_entities(self, entities: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Codifica entidades usando SNOMED-CT con contexto RAG
        
        Args:
            entities: Lista de entidades del NER
            verbose: Si True, imprime logs detallados
            
        Returns:
            Lista de entidades codificadas con códigos SNOMED
        """
        if verbose:
            print(f"[CODING] Codificando {len(entities)} entidades...")
        
        coded_entities = []
        
        for idx, entity in enumerate(entities):
            if verbose:
                print(f"\n[CODING] Entidad {idx+1}/{len(entities)}: '{entity['span_text']}'")
            
            coded = self._code_single_entity(entity, verbose)
            coded_entities.append(coded)
        
        if verbose:
            print(f"\n[CODING] [OK] Codificación completada: {len(coded_entities)} entidades")
        
        return coded_entities
    
    def _code_single_entity(self, entity: Dict, verbose: bool) -> Dict:
        """Codifica una sola entidad"""
        
        span_text = entity['span_text']
        location = entity['anatomical_location']
        presence = entity['presence']
        
        # 1. Recuperar contexto RAG para la entidad
        contexto_entity = self._build_context(span_text, "ENTITY", verbose)
        
        # 2. Recuperar contexto RAG para la ubicación anatómica
        contexto_anatomy = ""
        if location and location != "No especificado":
            contexto_anatomy = self._build_context(location, "ANATOMY", verbose)
        
        # 3. NO combinar contextos - mantenerlos separados para evitar confusión
        # Asegurar que ambos contextos existan (aunque sea vacío)
        if not contexto_entity.strip():
            contexto_entity = "--- NO SPECIFIC ENTITY CODES AVAILABLE ---\nUse default entity code."
        if not contexto_anatomy.strip():
            contexto_anatomy = "--- NO SPECIFIC ANATOMY CODES AVAILABLE ---\nUse default anatomy code (12738006)."
        
        # 4. Preparar prompt con contextos SEPARADOS
        prompt = self.prompt_template.format(
            entity=span_text,
            location=location,
            presence=presence,
            contexto_entity=contexto_entity,
            contexto_anatomy=contexto_anatomy
        )
        
        # 5. Llamar a GPT-4o
        if verbose:
            print(f"[CODING]   -> Consultando GPT-4o...")
        
        response = self._call_gpt4o(prompt)
        
        # 6. Parsear respuesta
        codes = self._parse_coding_response(response, presence, verbose)
        
        # 7. Construir entidad codificada
        coded_entity = {
            **entity,  # Mantener datos originales
            'entity_code': codes['entity_code'],
            'anatomy_code': codes['anatomy_code'],
            'presence_code': codes['presence_code'],
            'entity_description': span_text,
            'anatomy_description': location
        }
        
        return coded_entity
    
    def _build_context(self, query: str, context_type: str, verbose: bool) -> str:
        """Construye contexto ontológico usando RAG"""
        
        results = self.rag.retrieve(query, k=5)
        
        if not results:
            return ""
        
        if verbose:
            best_code, _, best_dist = results[0]
            print(f"[CODING]   -> RAG {context_type}: {len(results)} conceptos (mejor: {best_code}, dist: {best_dist:.3f})")
        
        context = f"\n--- {context_type} CODES for '{query}' ---\n"
        for idx, (concepto, narrativa, dist) in enumerate(results, 1):
            context += f"{idx}. CÓDIGO: {concepto} | DESCRIPCIÓN: {narrativa[:150]}\n"
        
        return context
    
    def _call_gpt4o(self, prompt: str, max_retries: int = 3) -> str:
        """Llama a GPT-4o para codificación"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=self.model_config.get("max_tokens", 1000),
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"[CODING] [ERROR] Error GPT-4o (intento {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return '{"entity_code": "404684003", "anatomy_code": "12738006", "presence_code": "261665006"}'
        
        return '{"entity_code": "404684003", "anatomy_code": "12738006", "presence_code": "261665006"}'
    
    def _parse_coding_response(self, response: str, presence: str, verbose: bool) -> Dict:
        """Parsea la respuesta de codificación"""
        
        try:
            # Buscar JSON en la respuesta
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON encontrado")
            
            data = json.loads(json_match.group(0))
            
            # Extraer códigos
            entity_code = str(data.get("entity_code", "404684003"))
            anatomy_code = str(data.get("anatomy_code", "12738006"))
            
            # Validar que son numéricos
            if not entity_code.isdigit():
                if verbose:
                    print(f"[CODING]   [WARNING] entity_code no numérico: '{entity_code}', usando default")
                entity_code = "404684003"
            
            if not anatomy_code.isdigit():
                if verbose:
                    print(f"[CODING]   [WARNING] anatomy_code no numérico: '{anatomy_code}', usando default")
                anatomy_code = "12738006"
            
            # Código de presencia (fijo según mapeo)
            presence_code = self.presence_map.get(presence.lower(), "261665006")
            
            if verbose:
                print(f"[CODING]   [OK] Códigos: entity={entity_code}, anatomy={anatomy_code}, presence={presence_code}")
            
            return {
                'entity_code': entity_code,
                'anatomy_code': anatomy_code,
                'presence_code': presence_code
            }
            
        except Exception as e:
            if verbose:
                print(f"[CODING]   [ERROR] Error parseando: {e}")
            
            return {
                'entity_code': "404684003",
                'anatomy_code': "12738006",
                'presence_code': self.presence_map.get(presence.lower(), "261665006")
            }
