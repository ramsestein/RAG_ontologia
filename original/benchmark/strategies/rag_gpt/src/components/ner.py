"""Named Entity Recognition usando GPT-4o (con system prompt y offsets de caracteres)
Robusto:
- Limpieza centralizada del JSON (markdown/trailing commas).
- Extracción del PRIMER objeto JSON balanceando llaves (tolerante a texto extra).
- Fallback: parseo por-objeto dentro de "entities" (equilibra llaves y repara separadores).
- Saneado de caracteres de control y conversión segura de offsets.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from openai import OpenAI

# Setup paths for absolute imports
COMPONENTS_DIR = Path(__file__).parent.resolve()
SRC_DIR = COMPONENTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# reutilizamos limpieza común
from utils.text import clean_json_response


class NERExtractor:
    """Extractor de entidades médicas usando GPT-4o"""

    def __init__(self, client: OpenAI, prompt_config: Dict, model_config: Dict, system_prompt: Optional[str] = None):
        self.client = client
        self.prompt_template = prompt_config['template']
        self.model_config = model_config
        self.system_prompt = system_prompt or "You are a precise clinical NER extractor. Respond ONLY with valid JSON."

    def extract_entities(self, texto: str) -> List[Dict]:
        """Extrae entidades médicas del texto usando GPT-4o"""
        print("[NER] Ejecutando extracción de entidades con GPT-4o...")

        # USAR render seguro (evita KeyError por llaves JSON en la plantilla)
        prompt = self._render(self.prompt_template, {"informe": texto})
        response = self._call_gpt4o(prompt)
        entities = self._parse_ner_response(response)
        
        # NOTE: Offset fixing disabled - metrics don't use spans, only concept_ids
        # entities = self._fix_offsets(entities, texto)

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
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[NER] Error en llamada GPT-4o (intento {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return '{"entities": []}'

        return '{"entities": []}'

    # -------------------------------
    # Parsing y reparación de JSON
    # -------------------------------
    def _strip_control_chars(self, s: str) -> str:
        """Elimina caracteres de control no imprimibles que rompen JSON."""
        return ''.join(ch for ch in s if ch == '\t' or ch == '\n' or ch == '\r' or ord(ch) >= 32)

    def _extract_top_level_json(self, s: str) -> Optional[str]:
        """
        Devuelve el primer objeto JSON balanceando llaves, ignorando texto extra.
        Soporta respuestas con prólogo/epílogo.
        """
        # limpiar markdown + trailing commas + control chars
        s = clean_json_response(self._strip_control_chars(s))
        # quitar elipsis sueltas al final
        s = s.rstrip('.… \n\r\t')

        # buscar primera llave
        start = s.find('{')
        if start == -1:
            return None

        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        # si no cerró, intentar cerrar brutamente
        if depth > 0:
            return s[start:] + ('}' * depth)
        return None

    def _repair_entities_array(self, text: str) -> str:
        """
        Repara separadores y cierra el array 'entities' si es evidente que quedó abierto.
        """
        # arreglar objetos consecutivos sin coma dentro de arrays
        text = re.sub(r'}\s*{', '}, {', text)

        # si abre "entities": [ pero falta ']' antes de cerrar objeto raíz,
        # insertamos un ']' justo antes del último '}'.
        ent_open = text.find('"entities"')
        if ent_open != -1:
            arr_open = text.find('[', ent_open)
            if arr_open != -1:
                arr_close = text.find(']', arr_open)
                if arr_close == -1:
                    # insertar ']' antes del último '}'
                    last_brace = text.rfind('}')
                    if last_brace != -1 and last_brace > arr_open:
                        text = text[:last_brace] + ']' + text[last_brace:]
        return text

    def _iter_objects_in_entities(self, text: str) -> List[str]:
        """
        Extrae cada objeto del array "entities" balanceando llaves.
        Útil como Fallback cuando el JSON completo no parsea.
        """
        text = self._strip_control_chars(text)
        m = re.search(r'"entities"\s*:\s*\[', text)
        if not m:
            return []
        i = text.find('[', m.end() - 1)
        if i == -1:
            return []

        objs = []
        depth = 0
        in_str = False
        escape = False
        start_obj = None

        # recorremos desde el primer carácter dentro del array
        for j in range(i + 1, len(text)):
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    if depth == 0:
                        start_obj = j
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start_obj is not None:
                        objs.append(text[start_obj:j+1])
                        start_obj = None
                elif ch == ']':
                    if depth == 0:
                        break  # fin del array

        return objs

    def _safe_int(self, x):
        if x is None:
            return None
        try:
            if isinstance(x, str):
                x = x.strip()
                if not re.match(r'^-?\d+(\.0+)?$', x):
                    return None
            return int(float(x))
        except Exception:
            return None

    def _normalize_entity(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(finding, dict):
            return None
        core_entity = (finding.get("core_entity") or finding.get("entity") or "").strip()
        full_span = (finding.get("full_span") or core_entity or "").strip()
        if not full_span:
            return None
        anatomical_location = (finding.get("anatomical_location") or "No especificado") or "No especificado"
        presence = (finding.get("presence") or "presente") or "presente"
        value = finding.get("value", None)
        start_int = self._safe_int(finding.get("start", None))
        end_int = self._safe_int(finding.get("end", None))
        return {
            "span_text": core_entity if core_entity else full_span,
            "full_span": full_span,
            "anatomical_location": anatomical_location,
            "presence": presence,
            "value": value,
            "start": start_int,
            "end": end_int
        }

    def _parse_ner_response(self, response: str) -> List[Dict]:
        """Parsea la respuesta JSON del NER y normaliza campos expected (tolerante a errores)"""
        try:
            candidate = self._extract_top_level_json(response)
            if not candidate:
                raise ValueError("No se encontró objeto JSON en la respuesta")

            candidate = self._repair_entities_array(candidate)

            # Intento 1: parse completo
            try:
                data = json.loads(candidate)
                entities_raw = data.get("entities", [])
                if not isinstance(entities_raw, list):
                    # buscar en cualquier clave una lista de dicts
                    for v in data.values():
                        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                            entities_raw = v
                            break
                entities: List[Dict[str, Any]] = []
                for finding in entities_raw:
                    ent = self._normalize_entity(finding)
                    if ent:
                        entities.append(ent)
                return entities

            except json.JSONDecodeError:
                # Intento 2: arreglos simples y reintento
                candidate2 = re.sub(r',(\s*[}\]])', r'\1', candidate)
                candidate2 = re.sub(r'}\s*{', '}, {', candidate2)
                try:
                    data = json.loads(candidate2)
                    entities_raw = data.get("entities", [])
                    if not isinstance(entities_raw, list):
                        for v in data.values():
                            if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                                entities_raw = v
                                break
                    entities = []
                    for finding in entities_raw:
                        ent = self._normalize_entity(finding)
                        if ent:
                            entities.append(ent)
                    return entities
                except json.JSONDecodeError:
                    # Fallback definitivo: parsear cada objeto del array "entities" por separado
                    objs = self._iter_objects_in_entities(candidate2)
                    entities = []
                    for obj_str in objs:
                        obj_str = re.sub(r',(\s*[}\]])', r'\1', obj_str)
                        try:
                            finding = json.loads(obj_str)
                            ent = self._normalize_entity(finding)
                            if ent:
                                entities.append(ent)
                        except Exception:
                            # ignorar objetos irrecuperables
                            continue

                    if entities:
                        return entities
                    # si no se pudo recuperar nada:
                    raise

        except Exception as e:
            print(f"[NER] Error parseando respuesta: {e}")
            print(f"[NER] Respuesta: {response[:500]}...")
            return []

    def _render(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render muy simple: sustituye solo las claves provistas {clave} por su valor,
        sin interpretar el resto de llaves del template (evita conflictos con JSON).
        """
        s = template
        for k, v in variables.items():
            s = s.replace("{" + k + "}", str(v))
        return s
    
    def _fix_offsets(self, entities: List[Dict], texto: str) -> List[Dict]:
        """
        Fix character offsets by finding the actual position of span_text in the document.
        GPT-4o often returns incorrect character positions, so we search for the text.
        Uses fuzzy matching for partial matches.
        """
        fixed_entities = []
        used_positions = set()  # Track used positions to handle duplicates
        
        for ent in entities:
            span_text = ent.get("span_text", "")
            if not span_text or len(span_text) < 2:
                continue
            
            # Try exact match first
            best_match = None
            search_start = 0
            
            while True:
                pos = texto.find(span_text, search_start)
                if pos == -1:
                    # Try case-insensitive search
                    lower_text = texto.lower()
                    lower_span = span_text.lower()
                    pos = lower_text.find(lower_span, search_start)
                    if pos == -1:
                        # Try to find a partial match (at least 80% of the span)
                        min_len = max(3, int(len(span_text) * 0.8))
                        if len(span_text) >= min_len:
                            # Try searching for the first part of the span
                            partial_span = span_text[:min_len]
                            pos = lower_text.find(partial_span.lower(), search_start)
                            if pos != -1:
                                # Extend to full word if possible
                                end_pos = pos + len(span_text)
                                if end_pos > len(texto):
                                    end_pos = len(texto)
                                # Find word boundary
                                while end_pos < len(texto) and texto[end_pos].isalnum():
                                    end_pos += 1
                                if pos not in used_positions:
                                    best_match = (pos, end_pos)
                                    used_positions.add(pos)
                                    break
                        break
                
                # Check if this position is already used
                if pos not in used_positions:
                    best_match = (pos, pos + len(span_text))
                    used_positions.add(pos)
                    break
                
                # Try next occurrence
                search_start = pos + 1
            
            if best_match:
                ent["start"] = best_match[0]
                ent["end"] = best_match[1]
                # Update full_span and span_text with actual text from document
                actual_text = texto[best_match[0]:best_match[1]]
                ent["full_span"] = actual_text
                ent["span_text"] = actual_text
                fixed_entities.append(ent)
            # Don't skip entities - just use original offsets if we can't fix them
            elif ent.get("start") is not None and ent.get("end") is not None:
                # Validate original offsets are within document bounds
                if 0 <= ent["start"] < len(texto) and ent["start"] < ent["end"] <= len(texto):
                    fixed_entities.append(ent)
        
        return fixed_entities
