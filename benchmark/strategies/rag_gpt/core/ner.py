"""Named Entity Recognition usando GPT-4o (con system prompt y offsets de caracteres)
Robusto:
- Limpieza centralizada del JSON (markdown/trailing commas).
- Extracción del PRIMER objeto JSON balanceando llaves (tolerante a texto extra).
- Reparación de separadores '}{' → '}, {' dentro de arrays.
- Cierre de arreglos y conversión segura de offsets.
"""

import json
import re
from typing import List, Dict, Optional, Any
from openai import OpenAI

# reutilizamos limpieza común
from strategies.rag_gpt.utils.text_processing import clean_json_response


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
    def _extract_top_level_json(self, s: str) -> Optional[str]:
        """
        Devuelve el primer objeto JSON balanceando llaves, ignorando texto extra.
        Soporta respuestas con prólogo/epílogo.
        """
        # limpiar markdown + trailing commas
        s = clean_json_response(s)
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
        # Heurística segura: solo si aparece '"entities": [' y NO hay ']' después.
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

    def _parse_ner_response(self, response: str) -> List[Dict]:
        """Parsea la respuesta JSON del NER y normaliza campos expected (tolerante a errores)"""
        try:
            candidate = self._extract_top_level_json(response)
            if not candidate:
                raise ValueError("No se encontró objeto JSON en la respuesta")

            candidate = self._repair_entities_array(candidate)

            # Intento 1: parse directo
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                # Intento 2: reparar comas finales y parsear de nuevo
                candidate2 = re.sub(r',(\s*[}\]])', r'\1', candidate)
                try:
                    data = json.loads(candidate2)
                except json.JSONDecodeError:
                    # Intento 3: extraer solo el array si vino como lista en bruto
                    m_list = re.search(r'\[\s*\{.*\}\s*\]', candidate2, re.DOTALL)
                    if m_list:
                        data = {"entities": json.loads(m_list.group(0))}
                    else:
                        # Intento 4: forzar comas entre objetos y reintentar
                        candidate3 = re.sub(r'}\s*{', '}, {', candidate2)
                        data = json.loads(candidate3)

            entities_raw = data.get("entities", [])
            if not isinstance(entities_raw, list):
                # fallback: buscar en cualquier clave un array de dicts con 'core_entity' o 'full_span'
                found = None
                for v in data.values():
                    if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                        found = v
                        break
                entities_raw = found if found is not None else []

            entities: List[Dict[str, Any]] = []
            for finding in entities_raw:
                if not isinstance(finding, dict):
                    continue

                core_entity = (finding.get("core_entity") or finding.get("entity") or "").strip()
                full_span = (finding.get("full_span") or core_entity or "").strip()

                anatomical_location = (finding.get("anatomical_location") or "No especificado") or "No especificado"
                presence = (finding.get("presence") or "presente") or "presente"
                value = finding.get("value", None)

                # offsets tolerantes
                start_raw = finding.get("start", None)
                end_raw = finding.get("end", None)

                def _to_int(x):
                    if x is None:
                        return None
                    try:
                        # soporta strings numéricas, floats representando enteros, etc.
                        if isinstance(x, str):
                            x = x.strip()
                            # descarta no numérico
                            if not re.match(r'^-?\d+(\.0+)?$', x):
                                return None
                        return int(float(x))
                    except Exception:
                        return None

                start_int = _to_int(start_raw)
                end_int = _to_int(end_raw)

                if not full_span:
                    continue

                entities.append({
                    "span_text": core_entity if core_entity else full_span,
                    "full_span": full_span,
                    "anatomical_location": anatomical_location,
                    "presence": presence,
                    "value": value,
                    "start": start_int,
                    "end": end_int
                })

            return entities

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
