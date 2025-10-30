"""Named Entity Recognition usando GPT-4o"""

import json
import re
from typing import List, Dict
from openai import OpenAI


class NERExtractor:
    """Extractor de entidades médicas usando GPT-4o"""
    
    def __init__(self, client: OpenAI, prompt_config: Dict, model_config: Dict):
        self.client = client
        self.prompt_template = prompt_config['template']
        self.model_config = model_config
        
    def extract_entities(self, texto: str) -> List[Dict]:
        """Extrae entidades médicas del texto usando GPT-4o"""
        print("[NER] Ejecutando extracción de entidades con GPT-4o...")
        
        prompt = self.prompt_template.format(informe=texto)
        response = self._call_gpt4o(prompt)
        entities = self._parse_ner_response(response)
        
        print(f"[NER] Entidades detectadas: {len(entities)}")
        for i, ent in enumerate(entities[:3]):
            print(f"  - Span: \"{ent['span_text']}\"")
        if len(entities) > 3:
            print(f"  ... y {len(entities) - 3} más")
        
        return entities
    
    def _call_gpt4o(self, prompt: str, max_retries: int = 3) -> str:
        """Llama a GPT-4o con manejo de errores"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.05,
                    max_tokens=4500,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"[NER] Error en llamada GPT-4o (intento {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return '{"entities": []}'
        
        return '{"entities": []}'
    
    def _parse_ner_response(self, response: str) -> List[Dict]:
        """Parsea la respuesta JSON del NER"""
        try:
            response_clean = response.strip()
            
            if '```json' in response_clean:
                json_start = response_clean.find('```json') + 7
                json_end = response_clean.find('```', json_start)
                response_clean = response_clean[json_start:json_end].strip()
            elif '```' in response_clean:
                json_start = response_clean.find('```') + 3
                json_end = response_clean.find('```', json_start)
                response_clean = response_clean[json_start:json_end].strip()
            
            response_clean = re.sub(r',(\s*[}\]])', r'\1', response_clean)
            
            try:
                data = json.loads(response_clean)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                if json_match:
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_match.group())
                    data = json.loads(json_str)
                else:
                    raise ValueError("No se pudo extraer JSON válido")
            
            entities = []
            if "entities" in data:
                for finding in data["entities"]:
                    core_entity = finding.get("core_entity", "")
                    full_span = finding.get("full_span", "")
                    
                    if not core_entity:
                        if full_span:
                            core_entity = full_span
                        else:
                            continue

                    if not full_span:
                        full_span = core_entity

                    entities.append({
                        "span_text": core_entity,
                        "full_span": full_span,
                        "anatomical_location": finding.get("anatomical_location", "No especificado"),
                        "presence": finding.get("presence", "presente"),
                        "value": finding.get("value")
                    })
            
            return entities
            
        except Exception as e:
            print(f"[NER] Error parseando respuesta: {e}")
            print(f"[NER] Respuesta: {response[:500]}...")
            return []
