#!/usr/bin/env python3
"""
RAG+GPT4o Strategy Implementation
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
        
        # --- PROMPT NER EN ESPAÑOL (OPTIMIZADO PARA SPANS CORTOS) ---
        self.ner_prompt_template = """
<tarea>
Eres un agente experto en NER clínico. Tu tarea es extraer TODAS las entidades clínicas relevantes del informe médico en inglés.

**CRÍTICO: Extrae el SPAN MÁS CORTO y ESPECÍFICO para cada entidad**
- Para "left-sided weakness" → extrae solo "weakness"  
- Para "NIHSS score of 12" → extrae solo "NIHSS"
- Para "mechanical thrombectomy" → extrae solo "thrombectomy"
- Extrae PALABRAS INDIVIDUALES o FRASES CORTAS, no oraciones completas

**Entidades a Extraer:**
- Hallazgos (ej: hemorrhage, infarct, occlusion, stenosis)
- Anatomía (ej: MCA, M1, M2, caudate, internal capsule)
- Escalas (ej: NIHSS, ASPECTS, TICI, GCS, mRS)
- Síntomas (ej: hemiparesis, aphasia, dysarthria, weakness)
- Procedimientos (ej: thrombectomy, thrombolysis, tPA, angiography)
- Comorbilidades (ej: hypertension, diabetes)

**Reglas:**
- Extrae CADA instancia - si "CT" aparece 5 veces, extrae las 5
- Usa el span MÁS CORTO posible (usualmente 1-2 palabras)
- Para abreviaturas, extrae SOLO la abreviatura (ej: "CT" no "computed tomography")
- Para escalas con valores, extrae SOLO el nombre de la escala (ej: "NIHSS" no "NIHSS score of 12")
- Para cada entidad, proporciona:
  1. `span_text`: El texto MÁS CORTO y ESPECÍFICO del informe
  2. `anatomical_location`: La ubicación si se menciona, o 'No especificado'
  3. `presence`: "presente", "ausente", o "incierto"
  4. `value`: El valor si aplica (ej: "18" para "NIHSS score was 18")
</tarea>

<formato_salida>
{{
  "entities": [
    {{
      "span_text": "término específico más corto",
      "anatomical_location": "ubicación o 'No especificado'",
      "presence": "presente | ausente | incierto",
      "value": "valor o null"
    }}
  ]
}}
</formato_salida>

<informe>
{informe}
</informe>

Responde SOLO con el JSON válido:
"""
        
        # --- PROMPT DE CODIFICACIÓN (ULTRA-SIMPLIFICADO v7 - EN ESPAÑOL) ---
        # Enfoque de selección múltiple forzada
        self.coding_prompt_template = """
Eres un codificador SNOMED-CT. Tu ÚNICA tarea es seleccionar códigos de la lista a continuación.

**ENTIDAD A CODIFICAR:** "{entity}"
**UBICACIÓN ANATÓMICA:** "{location}"
**PRESENCIA:** "{presence}"

**CÓDIGOS DISPONIBLES (SELECCIONA SOLO DE ESTA LISTA):**
{contexto_ontologico}

**REGLAS CRÍTICAS:**
[ERROR] NO uses "404684003" o "12738006" si CUALQUIER otro código de la lista coincide
[ERROR] NO inventes códigos - SOLO usa códigos de la lista anterior
[OK] COPIA el número de CÓDIGO exacto (solo dígitos) de la lista
[OK] Elige la coincidencia MÁS ESPECÍFICA

**CÓDIGOS DE PRESENCIA (FIJOS - NO CAMBIAR):**
- Si presencia es "presente" → usa "52101004"
- Si presencia es "ausente" → usa "272519000"
- De lo contrario → usa "261665006"

**TU TAREA:**
1. Encuentra la línea en "CÓDIGOS DISPONIBLES" que mejor coincida con "{entity}"
2. Copia SOLO el número después de "CÓDIGO:"
3. Ese número va en "entity_code"
4. Haz lo mismo para "{location}" → "anatomy_code"

**SALIDA (solo JSON, sin explicación):**
{{
  "entity_code": "NUMERO_DE_LA_LISTA",
  "anatomy_code": "NUMERO_DE_LA_LISTA_O_12738006_SI_NO_HAY_UBICACION",
  "presence_code": "52101004_O_272519000_O_261665006"
}}

**EJEMPLO:**
Si la lista tiene: "CÓDIGO: 230690007 | DESCRIPCIÓN: ictus accidente cerebrovascular ACV"
Y la entidad es "ictus"
Entonces salida: {{"entity_code": "230690007", ...}}
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
                            "content": "Eres un asistente de codificación SNOMED-CT. SOLO seleccionas códigos de la lista proporcionada. NUNCA inventas códigos. SIEMPRE respondes con JSON válido conteniendo solo los campos solicitados. Eres preciso y determinista."
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