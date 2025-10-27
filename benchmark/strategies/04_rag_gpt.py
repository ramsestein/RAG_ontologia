#!/usr/bin/env python3
"""
Tu estrategia RAG original modificada para usar GPT-4o en lugar de Llama 3.3 70B
Mantiene toda la lógica de RAG + búsqueda semántica pero cambia el LLM

REFACTORED VERSION (for F1-Score improvement):
- Updates asset paths to new '04_rag_gpt_assets' directory
- Replaces abstractive-summary NER prompt with an EXTRACTIVE-NER prompt
- Preserves the original 'span_text' through the coding pipeline
- Implements span-finding logic in predict() to get correct start/end offsets

*** ARREGLO v5: Corregir el prompt de CODIFICACIÓN para que ELIJA el código ***
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import json
import sys
import os
import pickle
import openai
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer


# --- START: Robust Path Setup (Updated for new location) ---

# Get the absolute path to THIS script's directory (.../benchmark/strategies)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root (.../RAG_ontologia)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# --- RUTA CORREGIDA ---
# Path to the NEW assets directory (basado en tu log: 04_utils/assets)
ASSETS_DIR = os.path.join(SCRIPT_DIR, '04_utils', 'assets')

# --- END: Robust Path Setup ---

class RAGWithGPT4oStrategy:
    """
    Tu estrategia RAG original pero usando GPT-4o via OpenAI API
    
    REFACTORED:
    - Carga el índice Faiss pre-construido desde la nueva ruta de assets.
    - Utiliza un pipeline NER extractivo para mejorar drásticamente el F1-Score.
    """
    
    def __init__(self):
        print("[RAG+GPT4o] Inicializando estrategia con GPT-4o...")
        
        # Configurar OpenAI
        self._setup_openai()
        
        # Cargar ontología y conceptos pre-procesados
        self._load_ontology_data()
        
        # Cargar índice Faiss pre-construido (¡RÁPIDO! [READY])
        self._load_faiss_index()
        
        # Configurar prompts
        self._setup_prompts()
        
        print("[RAG+GPT4o] [OK] Inicialización completada")
    
    def _setup_openai(self):
        """Configura la API de OpenAI con GPT-4o"""
        
        # API Key de ChatGPT - Cargar desde archivo api_keys
        try:
            api_key_path = os.path.join(PROJECT_ROOT, 'api_keys')
            with open(api_key_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("chatGPT="):
                        api_key = line.split("=")[1].strip()
                        break
        except:
            api_key = "YOUR_OPENAI_API_KEY_HERE"  # Placeholder
        
        self.client = OpenAI(api_key=api_key)
        
        # Configuración del modelo
        self.model_config = {
            "model": "gpt-4o",  # GPT-4o más reciente
            "temperature": 0.1, # Bajar temperatura para NER más predecible
            "max_tokens": 2048, # Aumentar para informes largos
            "top_p": 0.9
        }
        
        print("[RAG+GPT4o] OpenAI configurado con GPT-4o")
    
    def _load_ontology_data(self):
        """
        Carga conceptos y narrativas pre-procesados desde archivos pickle
        en la NUEVA ubicación de assets.
        """
        concepts_path = os.path.join(ASSETS_DIR, 'ontology_concepts.pkl')
        narratives_path = os.path.join(ASSETS_DIR, 'ontology_narratives.pkl')
        
        if os.path.exists(concepts_path) and os.path.exists(narratives_path):
            try:
                print("[RAG+GPT4o] Cargando conceptos desde archivos pre-procesados...")
                
                with open(concepts_path, 'rb') as f:
                    self.conceptos = pickle.load(f)
                
                with open(narratives_path, 'rb') as f:
                    self.narrativas = pickle.load(f)
                
                print(f"[RAG+GPT4o] [OK] Cargados {len(self.conceptos)} conceptos (pre-procesados)")
                return
                
            except Exception as e:
                print(f"[RAG+GPT4o] [WARNING]  Error cargando archivos pre-procesados: {e}")
                print("[RAG+GPT4o] Intentando cargar desde CSV...")
        
        # Fallback: cargar desde CSV (si los pickle no existen)
        print(f"[RAG+GPT4o] [WARNING]  Archivos pre-procesados no encontrados en {ASSETS_DIR}")
        print("[RAG+GPT4o] Por favor, ejecuta primero: python strategies/04_utils/ontology_preprocessor.py")
        print("[RAG+GPT4o] Usando ontología simplificada como último recurso...")
        self._create_fallback_ontology()

    def _create_fallback_ontology(self):
        """Crea ontología simplificada si no se puede cargar la original"""
        
        fallback_concepts = {
            "230690007": "stroke cerebrovascular accident CVA brain attack acute neurological deficit sudden onset weakness speech difficulties lesión isquémica infarto cerebral",
            "50582007": "hemiparesis hemiplegia weakness paralysis motor deficit left sided right sided weakness facial droop debilidad motora",
            "87486003": "aphasia dysphasia speech difficulty language disorder communication deficit expression comprehension afasia trastorno del lenguaje",
            "8011004": "dysarthria slurred speech articulation disorder motor speech impairment disartria habla arrastrada",
            "25064002": "headache cephalgia head pain severe headache sudden onset worst headache of life cefalea dolor de cabeza",
            "50960005": "hemorrhage bleeding haemorrhage blood extravasation subarachnoid hemorrhage intracerebral hemorrhage hemorragia sangrado",
            "55342001": "infarct infarction ischemic lesion tissue death acute infarct brain infarct lesión isquémica infarto",
            "52674009": "ischemia ischaemia reduced blood flow cerebral ischemia tissue hypoxia isquemia reducción flujo sanguíneo",
            "433112001": "thrombectomy mechanical thrombectomy clot removal endovascular treatment stent retriever trombectomía extracción coágulo",
            "77343006": "angiography angiogram vessel imaging arteriography cerebral angiography contrast injection angiografía imagen vascular",
            "77477000": "CT computed tomography CAT scan tomografía computarizada escáner",
            "113091000": "MRI magnetic resonance imaging MR resonancia magnética",
            "69930009": "middle cerebral artery MCA cerebral artery M1 segment M2 segment territory arteria cerebral media ACM",
            "86547008": "internal carotid artery ICA carotid artery carotid stenosis carotid occlusion arteria carótida interna",
            "67889009": "basilar artery basilar arteria basilar",
            "450893003": "NIHSS ASPECTS TICI clinical scale neurological scale stroke scale assessment escala clínica evaluación neurológica",
        }
        
        self.conceptos = list(fallback_concepts.keys())
        self.narrativas = list(fallback_concepts.values())
        
        print(f"[RAG+GPT4o] Usando {len(self.conceptos)} conceptos de fallback")

    def _load_faiss_index(self):
        """
        Carga el índice Faiss pre-construido desde la NUEVA ubicación.
        """
        index_path = os.path.join(ASSETS_DIR, 'ontology.index')
        metadata_path = os.path.join(ASSETS_DIR, 'ontology_metadata.pkl')
        
        # Verificar que el índice existe
        if not os.path.exists(index_path):
            print("[RAG+GPT4o] [ERROR] Índice Faiss no encontrado")
            print(f"[RAG+GPT4o] Esperado en: {index_path}")
            print("[RAG+GPT4o] ")
            print("[RAG+GPT4o] [FIX] SOLUCIÓN: Ejecuta el siguiente comando:")
            print("[RAG+GPT4o]    python strategies/04_utils/ontology_preprocessor.py")
            print("[RAG+GPT4o] ")
            print("[RAG+GPT4o] [WARNING]  Usando fallback sin Faiss (búsqueda simple)...")
            
            self.faiss_index = None
            self.embedding_model = None
            return
        
        try:
            print(f"[RAG+GPT4o] Cargando índice Faiss pre-construido desde: {index_path}")
            
            self.faiss_index = faiss.read_index(index_path)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RAG+GPT4o] [OK] Índice cargado: {metadata['n_concepts']} conceptos")
                print(f"[RAG+GPT4o]     - Modelo: {metadata['model_name']}")
            
            print("[RAG+GPT4o] Cargando modelo de embeddings para consultas...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("[RAG+GPT4o] [READY] Índice listo para búsqueda semántica")
            
        except Exception as e:
            print(f"[RAG+GPT4o] [ERROR] Error cargando índice Faiss: {e}")
            print("[RAG+GPT4o] [WARNING]  Usando búsqueda simple como fallback...")
            self.faiss_index = None
            self.embedding_model = None

    def _setup_prompts(self):
        """
        Configura los prompts.
        *** PROMPT DE CODIFICACIÓN (v5) ARREGLADO ***
        """
        
        # --- PROMPT NER EN INGLÉS (OPTIMIZADO PARA SPANS CORTOS) ---
        self.ner_prompt_template = """
<task>
You are an expert clinical NER agent. Your task is to extract ALL relevant clinical entities from the English medical report.

**CRITICAL: Extract the SHORTEST, MOST SPECIFIC span for each entity**
- For "left-sided weakness" → extract just "weakness"  
- For "NIHSS score of 12" → extract just "NIHSS"
- For "mechanical thrombectomy" → extract just "thrombectomy"
- Extract SINGLE WORDS or SHORT PHRASES, not full sentences

**Entities to Extract:**
- Findings (e.g., hemorrhage, infarct, occlusion, stenosis)
- Anatomy (e.g., MCA, M1, M2, caudate, internal capsule)
- Scales (e.g., NIHSS, ASPECTS, TICI, GCS, mRS)
- Symptoms (e.g., hemiparesis, aphasia, dysarthria, weakness)
- Procedures (e.g., thrombectomy, thrombolysis, tPA, angiography)
- Comorbidities (e.g., hypertension, diabetes)

**Rules:**
- Extract EVERY instance - if "CT" appears 5 times, extract all 5
- Use the SHORTEST span possible (usually 1-2 words)
- For abbreviations, extract ONLY the abbreviation (e.g., "CT" not "computed tomography")
- For scales with values, extract ONLY the scale name (e.g., "NIHSS" not "NIHSS score of 12")
- For each entity, provide:
  1. `span_text`: The SHORTEST, MOST SPECIFIC text from the report
  2. `anatomical_location`: The location if mentioned, or 'Not specified'
  3. `presence`: "present", "absent", or "uncertain"
  4. `value`: The value if applicable (e.g., "18" for "NIHSS score was 18")
</task>

<output_format>
{{
  "entities": [
    {{
      "span_text": "shortest specific term",
      "anatomical_location": "location or 'Not specified'",
      "presence": "present | absent | uncertain",
      "value": "value or null"
    }}
  ]
}}
</output_format>

<report>
{informe}
</report>

Respond ONLY with the valid JSON:
"""
        
        # --- PROMPT DE CODIFICACIÓN (ULTRA-SIMPLIFICADO v7) ---
        # Enfoque de selección múltiple forzada
        self.coding_prompt_template = """
You are a SNOMED-CT coder. Your ONLY job is to select codes from the list below.

**ENTITY TO CODE:** "{entity}"
**ANATOMICAL LOCATION:** "{location}"
**PRESENCE:** "{presence}"

**AVAILABLE CODES (SELECT FROM THIS LIST ONLY):**
{contexto_ontologico}

**CRITICAL RULES:**
[ERROR] DO NOT use "404684003" or "12738006" if ANY other code in the list matches
[ERROR] DO NOT invent codes - ONLY use codes from the list above
[OK] COPY the exact CODE number (digits only) from the list
[OK] Choose the MOST SPECIFIC match

**PRESENCE CODES (FIXED - DO NOT CHANGE):**
- If presence is "present" → use "52101004"
- If presence is "absent" → use "272519000"
- Otherwise → use "261665006"

**YOUR TASK:**
1. Find the line in "AVAILABLE CODES" that best matches "{entity}"
2. Copy ONLY the number after "CODE:"
3. That number goes in "entity_code"
4. Do the same for "{location}" → "anatomy_code"

**OUTPUT (JSON only, no explanation):**
{{
  "entity_code": "NUMBER_FROM_LIST",
  "anatomy_code": "NUMBER_FROM_LIST_OR_12738006_IF_NO_LOCATION",
  "presence_code": "52101004_OR_272519000_OR_261665006"
}}

**EXAMPLE:**
If list has: "CODE: 230690007 | DESCRIPTION: stroke ictus cerebrovascular accident"
And entity is "ictus"
Then output: {{"entity_code": "230690007", ...}}
"""

    def _call_gpt4o(self, prompt: str, max_retries: int = 3) -> str:
        """
        Llama a GPT-4o con manejo de errores
        MEJORADO: Temperature muy baja para codificación determinística
        """
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a SNOMED-CT coding assistant. You ONLY select codes from the provided list. You NEVER invent codes. You ALWAYS respond with valid JSON containing only the requested fields. You are precise and deterministic."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,  # Totalmente determinístico
                    max_tokens=4000,   # AUMENTADO - para NER completo
                    top_p=1.0,
                    response_format={"type": "json_object"}  # Forzar salida JSON
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"[RAG+GPT4o] [ERROR] Error en llamada GPT-4o (intento {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # Devolver JSON válido con error
                    return '{"error": "GPT-4o no disponible", "entity_code": "LINKING_FAILED", "anatomy_code": "LINKING_FAILED", "presence_code": "261665006"}'
        
        return '{"error": "GPT-4o falló", "entity_code": "LINKING_FAILED", "anatomy_code": "LINKING_FAILED", "presence_code": "261665006"}'

    def _recuperar_conceptos(self, texto: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Tu función recuperar_conceptos original usando Faiss real
        """
        if self.faiss_index is None or self.embedding_model is None:
            # Fallback a búsqueda simple
            return self._simple_text_search(texto, k)
        
        try:
            # Generar embedding para la consulta
            query_embedding = self.embedding_model.encode([texto])
            
            # Buscar en índice Faiss
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            # Convertir resultados
            resultados = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.conceptos):
                    concepto = self.conceptos[idx]
                    narrativa = self.narrativas[idx]
                    distancia = distances[0][i]
                    
                    resultados.append((concepto, narrativa, distancia))
            
            return resultados
            
        except Exception as e:
            print(f"[RAG+GPT4o] Error en búsqueda Faiss: {e}")
            return self._simple_text_search(texto, k)

    def _simple_text_search(self, texto: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Búsqueda simple de texto como fallback"""
        
        resultados = []
        texto_lower = texto.lower()
        
        for i, (concepto, narrativa) in enumerate(zip(self.conceptos, self.narrativas)):
            # Calcular similitud simple
            score = 0
            for palabra in texto_lower.split():
                if palabra in narrativa.lower():
                    score += 1
            
            if score > 0:
                resultados.append((concepto, narrativa, 1.0 / (1.0 + score)))
        
        # Ordenar por similitud y tomar top k
        resultados.sort(key=lambda x: x[2])
        return resultados[:k]

    def _execute_ner_step(self, texto: str) -> List[Dict]:
        """
        Ejecuta el Paso 1: NER EXTRACTIVO con GPT-4o (SIMPLIFICADO)
        """
        
        print("[RAG+GPT4o] Paso 1: Ejecutando NER EXTRACTIVO con GPT-4o...")
        
        # Preparar prompt
        prompt_ner = self.ner_prompt_template.format(informe=texto)
        
        # Llamar a GPT-4o
        response = self._call_gpt4o(prompt_ner)
        
        # Parsear respuesta JSON
        try:
            # Limpiar respuesta - remover markdown y espacios
            response_clean = response.strip()
            
            # Buscar el JSON - puede estar envuelto en ```json ... ```
            if '```json' in response_clean:
                json_start = response_clean.find('```json') + 7
                json_end = response_clean.find('```', json_start)
                response_clean = response_clean[json_start:json_end].strip()
            elif '```' in response_clean:
                json_start = response_clean.find('```') + 3
                json_end = response_clean.find('```', json_start)
                response_clean = response_clean[json_start:json_end].strip()
            
            # NUEVO: Limpiar trailing commas que rompen el JSON
            # Patrón: ,\s*} o ,\s*]
            response_clean = re.sub(r',(\s*[}\]])', r'\1', response_clean)
            
            # Intentar parsear directamente
            try:
                entidades_basicas = json.loads(response_clean)
            except json.JSONDecodeError as json_err:
                print(f"[RAG+GPT4o] Error JSON decode: {json_err}")
                print(f"[RAG+GPT4o] JSON problemático (primeros 1000 chars):")
                print(response_clean[:1000])
                
                # Buscar con regex si falla
                json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    # Limpiar trailing commas otra vez
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    entidades_basicas = json.loads(json_str)
                else:
                    raise ValueError("No se pudo extraer JSON válido")
            
            # Extraer entidades (NUEVO FORMATO SIMPLIFICADO)
            entidades_detectadas = []
            if "entities" in entidades_basicas: 
                for finding in entidades_basicas["entities"]:
                    span = finding.get("span_text", "")
                    
                    # Requerir solo span_text para continuar
                    if span:
                        entidades_detectadas.append({
                            "span_text": span,
                            "anatomical_location": finding.get("anatomical_location", "Not specified"),
                            "presence": finding.get("presence", "present"),
                            "value": finding.get("value")
                        })
            
            print(f"[RAG+GPT4o] Entidades EXTRACTIVAS detectadas: {len(entidades_detectadas)}")
            for i, ent in enumerate(entidades_detectadas[:3]): # Imprimir solo las primeras 3
                 print(f"  - Span: \"{ent['span_text']}\"")
            if len(entidades_detectadas) > 3:
                print(f"  ... y {len(entidades_detectadas) - 3} más")
            
            return entidades_detectadas
            
        except Exception as e:
            print(f"[RAG+GPT4o] Error parseando NER: {e}")
            print(f"[RAG+GPT4o] Respuesta GPT-4o (primeros 500 chars): {response[:500]}...")
            return []

    def _execute_coding_step(self, entidades_detectadas: List[Dict], texto_original: str = "") -> List[Dict]:
        """
        Ejecuta el Paso 2: Codificación con RAG + GPT-4o
        *** VERSIÓN MEJORADA CON LOGGING DETALLADO ***
        
        Args:
            entidades_detectadas: Lista de entidades del NER
            texto_original: Texto original del informe (para debugging)
        """
        
        print("[RAG+GPT4o] Paso 2: Generando contexto OWL para codificación...")
        entidades_codificadas = []
        
        for idx, ent_data in enumerate(entidades_detectadas):
            # Extraer datos del NER
            span_text_original = ent_data["span_text"] # <-- El SPAN COMPLETO
            location = ent_data["anatomical_location"]
            presence = ent_data["presence"]
            value = ent_data.get("value")
            
            print(f"\n[RAG+GPT4o] [SEARCH] Codificando entidad {idx+1}/{len(entidades_detectadas)}")
            print(f"[RAG+GPT4o]   [NOTE] Span: '{span_text_original}'")
            print(f"[RAG+GPT4o]   [LOCATION] Ubicación: '{location}'")
            print(f"[RAG+GPT4o]   [OK] Presencia: '{presence}'")
            
            # 1. Usar el SPAN_TEXT_ORIGINAL para la búsqueda RAG
            contexto_entity = ""
            if span_text_original:
                # Usar el span_text original para la búsqueda semántica
                similares_entity = self._recuperar_conceptos(span_text_original, k=5)  # Aumentado de 3 a 5
                
                # FILTRAR códigos no numéricos
                similares_entity_limpios = [
                    (concepto, narrativa, dist) 
                    for concepto, narrativa, dist in similares_entity
                    if str(concepto).isdigit()  # Solo códigos numéricos
                ]
                
                if similares_entity_limpios:
                    contexto_entity += f"--- ENTITY CODES for '{span_text_original}' ---\n"
                    for idx, (concepto, narrativa, dist) in enumerate(similares_entity_limpios, 1):
                        # Formato numerado y limpio
                        contexto_entity += f"{idx}. CODE: {concepto} | DESCRIPTION: {narrativa[:150]}\n"
                    
                    print(f"[RAG+GPT4o]   [FIND] Contexto RAG recuperado: {len(similares_entity_limpios)} conceptos válidos")
                    # Mostrar el mejor match
                    if similares_entity_limpios:
                        best_code, best_desc, best_dist = similares_entity_limpios[0]
                        print(f"[RAG+GPT4o]   [TARGET] Mejor match: {best_code} (dist: {best_dist:.3f})")
                else:
                    print(f"[RAG+GPT4o]   [WARNING]  No se encontraron códigos numéricos válidos para '{span_text_original}'")
            
            # Buscar conceptos similares para la ubicación anatómica
            contexto_anatomy = ""
            if location and location != "Not specified":
                similares_anatomy = self._recuperar_conceptos(location, k=5)  # Aumentado de 3 a 5
                
                # FILTRAR códigos no numéricos
                similares_anatomy_limpios = [
                    (concepto, narrativa, dist) 
                    for concepto, narrativa, dist in similares_anatomy
                    if str(concepto).isdigit()
                ]
                
                if similares_anatomy_limpios:
                    contexto_anatomy += f"\n--- ANATOMY CODES for '{location}' ---\n"
                    for idx, (concepto, narrativa, dist) in enumerate(similares_anatomy_limpios, 1):
                        contexto_anatomy += f"{idx}. CODE: {concepto} | DESCRIPTION: {narrativa[:150]}\n"
            
            contexto_ontologico = contexto_entity + contexto_anatomy
            
            # Si no hay contexto válido, usar mensaje explícito
            if not contexto_ontologico.strip():
                contexto_ontologico = "--- NO SPECIFIC CODES AVAILABLE ---\nUse default codes."
            
            # 2. Preparar prompt mejorado con formato más claro
            # Mapeo de presencia a código
            presence_code_map = {
                "present": "52101004",
                "absent": "272519000",
                "uncertain": "261665006"
            }
            presence_code_fixed = presence_code_map.get(presence.lower(), "261665006")
            
            prompt_coding = self.coding_prompt_template.format(
                entity=span_text_original,
                location=location,
                presence=presence,
                contexto_ontologico=contexto_ontologico
            )
            
            # Llamar a GPT-4o para codificación
            print(f"[RAG+GPT4o]   [AI] Consultando GPT-4o...")
            response = self._call_gpt4o(prompt_coding)
            
            # 3. LOGGING DETALLADO de la respuesta
            print(f"[RAG+GPT4o]   [RESPONSE] Respuesta GPT-4o (primeros 150 chars):")
            print(f"[RAG+GPT4o]      {response[:150]}...")
            
            try:
                # Parsear la respuesta JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    raise ValueError("No se encontró JSON en la respuesta")
                
                coded_response = json.loads(json_match.group(0))
                
                # VALIDACIÓN ESTRICTA: verificar que contiene los campos necesarios
                required_fields = ["entity_code", "presence_code"]
                missing_fields = [f for f in required_fields if f not in coded_response]
                
                if missing_fields:
                    print(f"[RAG+GPT4o]   [WARNING]  WARNING: Campos faltantes en JSON: {missing_fields}")
                    print(f"[RAG+GPT4o]   [WARNING]  JSON recibido: {coded_response}")
                
                # Extraer códigos con validación
                entity_code = str(coded_response.get("entity_code", "404684003"))
                anatomy_code = str(coded_response.get("anatomy_code", "12738006"))
                
                # Validar que son códigos numéricos
                if not entity_code.isdigit():
                    print(f"[RAG+GPT4o]   [WARNING]  ALERTA: entity_code no numérico: '{entity_code}', usando default")
                    entity_code = "404684003"
                
                if not anatomy_code.isdigit():
                    print(f"[RAG+GPT4o]   [WARNING]  ALERTA: anatomy_code no numérico: '{anatomy_code}', usando default")
                    anatomy_code = "12738006"
                
                # LOGGING de los códigos asignados
                print(f"[RAG+GPT4o]   [OK] Códigos asignados:")
                print(f"[RAG+GPT4o]      • entity_code: {entity_code}")
                print(f"[RAG+GPT4o]      • anatomy_code: {anatomy_code}")
                print(f"[RAG+GPT4o]      • presence_code: {presence_code_fixed}")
                
                # Verificar si se usaron defaults (indicador de problema)
                if entity_code == "404684003":
                    print(f"[RAG+GPT4o]   [WARNING]  PROBLEMA: Usando entity_code DEFAULT (Clinical finding)")
                if anatomy_code == "12738006":
                    print(f"[RAG+GPT4o]   [INFO]   Usando anatomy_code DEFAULT (Brain structure)")

                # 4. Construir el diccionario final de la entidad
                coded_entity = {
                    "original_span_text": span_text_original,
                    "anatomical_location": location,
                    "presence": presence,
                    "value": value,
                    
                    "entity_code": entity_code,
                    "entity_description": span_text_original,  # Usar el span como descripción
                    "anatomy_code": anatomy_code,
                    "anatomy_description": location,
                    "presence_code": presence_code_fixed  # Usar el código fijo calculado
                }
                entidades_codificadas.append(coded_entity)
                
            except Exception as e:
                print(f"[RAG+GPT4o]   [ERROR] ERROR parseando respuesta: {e}")
                print(f"[RAG+GPT4o]   [ERROR] Respuesta completa: {response}")
                print(f"[RAG+GPT4o]   [ERROR] Usando códigos FALLBACK (esto causará Match=0)")
                
                # Fallback con estructura básica
                fallback_data = {
                    "entity_code": "LINKING_FAILED",  # Código especial para identificar fallos
                    "entity_description": span_text_original,
                    "anatomical_location": location,
                    "anatomy_code": "LINKING_FAILED",
                    "presence": presence,
                    "presence_code": "52101004" if presence == "present" else "272519000",
                    "value": value,
                    "original_span_text": span_text_original 
                }
                entidades_codificadas.append(fallback_data)
        
        print(f"\n[RAG+GPT4o] [OK] Codificación completada para {len(entidades_codificadas)} entidades.")
        return entidades_codificadas

    def extract_entities(self, texto: str) -> List[Dict]:
        """
        Pipeline completo REFACTORIZADO:
        1. NER EXTRACTIVO con GPT-4o (devuelve spans)
        2. RAG + Codificación con GPT-4o (preserva spans)
        """
        
        # Paso 1: NER con GPT-4o
        entidades_detectadas = self._execute_ner_step(texto)
        
        if not entidades_detectadas:
            return []
        
        # Paso 2: Codificación RAG + GPT-4o (con texto original para debugging)
        entidades_codificadas = self._execute_coding_step(entidades_detectadas, texto)
        
        return entidades_codificadas

    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice entidades usando RAG + GPT-4o
        (Lógica de búsqueda de spans flexible)
        """
        print(f"[RAG+GPT4o] Procesando {len(notes_df)} notas con RAG + GPT-4o...")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            print(f"\n[RAG+GPT4o] === Procesando nota {note_id} ({idx+1}/{len(notes_df)}) ===")
            
            entities = self.extract_entities(text)
            
            last_search_idx = {}
            
            for entity in entities:
                span_text = entity.get('original_span_text')
                
                if not span_text:
                    span_text = entity.get('entity') # Fallback por si acaso
                    if not span_text:
                        print(f"[RAG+GPT4o] WARNING: Entidad sin span_text o entity, saltando.")
                        continue
                
                # Crear un patrón regex flexible que ignore múltiples espacios/saltos de línea
                palabras = re.split(r'\s+', span_text)
                palabras_escapadas = [re.escape(palabra) for palabra in palabras if palabra]
                regex_pattern = r'\s+'.join(palabras_escapadas)
                
                start_from = last_search_idx.get(regex_pattern, 0)
                
                # Usar el nuevo regex_pattern flexible
                match = re.search(regex_pattern, text[start_from:], re.IGNORECASE)
                
                if match:
                    # Calcular 'start' y 'end' absolutos
                    start = match.start() + start_from
                    end = match.end() + start_from
                    
                    # Actualizar el índice de última búsqueda para ESTE patrón
                    last_search_idx[regex_pattern] = end
                    
                    span_text_real = text[start:end]
                    
                else:
                    # Fallback: buscar desde el inicio (solo si no se ha encontrado antes)
                    if start_from == 0:
                        match_fallback = re.search(regex_pattern, text, re.IGNORECASE)
                        if match_fallback:
                            start = match_fallback.start()
                            end = match_fallback.end()
                            last_search_idx[regex_pattern] = end
                            span_text_real = text[start:end]
                        else:
                            print(f"[RAG+GPT4o] WARNING: No se pudo encontrar el span flexible '{span_text}' (patrón: {regex_pattern}) en la nota {note_id}. Saltando entidad.")
                            continue
                    else:
                        print(f"[RAG+GPT4o] WARNING: No se pudo encontrar otra instancia del span '{span_text}' (patrón: {regex_pattern}) en la nota {note_id} después de {start_from}. Saltando entidad.")
                        continue # No añadir predicciones con start=0

                predictions.append({
                    'note_id': note_id,
                    'start': start,
                    'end': end,
                    'concept_id': str(entity.get('entity_code', '404684003')),  # El código de la entidad principal
                    'span_text': span_text_real,
                    'confidence': 0.85,  # Confianza base
                    'entity_description': entity.get('entity_description', ''),
                    'anatomy_code': entity.get('anatomy_code', ''),
                    'presence_code': entity.get('presence_code', ''),
                    'llm_used': 'GPT-4o'
                })
                
                # LOGGING para debugging
                entity_code_used = entity.get('entity_code', 'MISSING')
                if entity_code_used in ['404684003', 'LINKING_FAILED', 'MISSING']:
                    print(f"[RAG+GPT4o]   [WARNING]  ALERTA: Predicción con código genérico/fallback: {entity_code_used} para span '{span_text[:50]}'")
            
            print(f"[RAG+GPT4o] Nota {note_id}: {len([p for p in predictions if p['note_id'] == note_id])} predicciones generadas")
        
        print(f"\n[RAG+GPT4o] Completado: {len(predictions)} predicciones generadas con GPT-4o")
        return pd.DataFrame(predictions)