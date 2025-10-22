#!/usr/bin/env python3
"""
Tu estrategia RAG original modificada para usar GPT-4o en lugar de Llama 3.3 70B
Mantiene toda la lógica de RAG + búsqueda semántica pero cambia el LLM

OPTIMIZED VERSION:
- Loads pre-built Faiss index for instant initialization (no embedding generation)
- Follows Single Responsibility Principle: index creation separated to build_rag_index.py
- Adheres to Open/Closed Principle: strategy class only depends on index artifact
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


# --- START: Robust Path Setup ---

# Get the absolute path to THIS script's directory (.../benchmark/strategies)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root (.../RAG_ontologia)
# We need to go up TWO levels ('..' to benchmark, '..' to root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Path to benchmark directory
BENCHMARK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Path to assets directory (where pre-built index is stored)
ASSETS_DIR = os.path.join(BENCHMARK_DIR, 'assets')

# --- END: Robust Path Setup ---

class RAGWithGPT4oStrategy:
    """
    Tu estrategia RAG original pero usando GPT-4o via OpenAI API
    
    OPTIMIZED: Now loads pre-built Faiss index instead of building on-the-fly
    This provides instant initialization and follows SRP by separating index
    creation (build_rag_index.py) from index usage (this class).
    """
    
    def __init__(self):
        print("[RAG+GPT4o] Inicializando estrategia con GPT-4o...")
        
        # Configurar OpenAI
        self._setup_openai()
        
        # Cargar ontología y conceptos pre-procesados
        self._load_ontology_data()
        
        # Cargar índice Faiss pre-construido (¡RÁPIDO! 🚀)
        self._load_faiss_index()
        
        # Configurar prompts
        self._setup_prompts()
        
        print("[RAG+GPT4o] ✅ Inicialización completada")
    
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
            "temperature": 0.3,  # Misma configuración que tu Llama
            "max_tokens": 1500,
            "top_p": 0.9
        }
        
        print("[RAG+GPT4o] OpenAI configurado con GPT-4o")
    
    def _load_ontology_data(self):
        """
        Carga conceptos y narrativas pre-procesados desde archivos pickle.
        
        Esto es mucho más rápido que cargar el CSV completo y permite
        mantener la alineación exacta con el índice Faiss.
        """
        concepts_path = os.path.join(ASSETS_DIR, 'ontology_concepts.pkl')
        narratives_path = os.path.join(ASSETS_DIR, 'ontology_narratives.pkl')
        
        # Intentar cargar desde archivos pickle pre-construidos
        if os.path.exists(concepts_path) and os.path.exists(narratives_path):
            try:
                print("[RAG+GPT4o] Cargando conceptos desde archivos pre-procesados...")
                
                with open(concepts_path, 'rb') as f:
                    self.conceptos = pickle.load(f)
                
                with open(narratives_path, 'rb') as f:
                    self.narrativas = pickle.load(f)
                
                print(f"[RAG+GPT4o] ✅ Cargados {len(self.conceptos)} conceptos (pre-procesados)")
                return
                
            except Exception as e:
                print(f"[RAG+GPT4o] ⚠️  Error cargando archivos pre-procesados: {e}")
                print("[RAG+GPT4o] Intentando cargar desde CSV...")
        
        # Fallback: cargar desde CSV (si los pickle no existen)
        print("[RAG+GPT4o] ⚠️  Archivos pre-procesados no encontrados")
        print("[RAG+GPT4o] Por favor, ejecuta primero: python build_rag_index.py")
        print("[RAG+GPT4o] Intentando cargar desde CSV como fallback...")
        
        try:
            # Intentar cargar desde el directorio principal
            conceptos_path_csv = os.path.join(PROJECT_ROOT, 'conceptos_con_narrativas.csv')
            
            if os.path.exists(conceptos_path_csv):
                self.df_conceptos = pd.read_csv(conceptos_path_csv)
            else:
                raise FileNotFoundError(f"No se encuentra: {conceptos_path_csv}")
            
            print(f"[RAG+GPT4o] Cargados {len(self.df_conceptos)} conceptos desde CSV")
            
            # Preparar listas para búsqueda
            self.conceptos = self.df_conceptos["concepto"].tolist()
            self.narrativas = self.df_conceptos["narrativa"].tolist()
            
        except Exception as e:
            print(f"[RAG+GPT4o] ❌ Error cargando ontología desde CSV: {e}")
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
            "32603002": "basal ganglia ganglios basales núcleos basales",
            "113305005": "cerebellum cerebelo",
            "15926001": "brainstem brain stem troncoencéfalo tronco encefálico",
            "415582006": "stenosis narrowing estenosis estrechamiento",
            "26036001": "occlusion blockage oclusión bloqueo",
            "432101006": "aneurysm aneurisma dilatación arterial",
            "230691006": "penumbra penumbra isquémica tejido salvable"
        }
        
        self.conceptos = list(fallback_concepts.keys())
        self.narrativas = list(fallback_concepts.values())
        
        print(f"[RAG+GPT4o] Usando {len(self.conceptos)} conceptos de fallback")
    
    def _load_faiss_index(self):
        """
        Carga el índice Faiss pre-construido desde disco.
        
        Este método SOLO carga el índice, no lo construye. La construcción
        se hace offline con build_rag_index.py (separación de responsabilidades).
        
        Benefits:
          🚀 Instant loading (milliseconds vs minutes)
          🔄 Consistency (same index across all runs)
          🧩 Modularity (index creation logic separated)
        """
        index_path = os.path.join(ASSETS_DIR, 'ontology.index')
        metadata_path = os.path.join(ASSETS_DIR, 'ontology_metadata.pkl')
        
        # Verificar que el índice existe
        if not os.path.exists(index_path):
            print("[RAG+GPT4o] ❌ Índice Faiss no encontrado")
            print(f"[RAG+GPT4o] Esperado en: {index_path}")
            print("[RAG+GPT4o] ")
            print("[RAG+GPT4o] 🔧 SOLUCIÓN: Ejecuta el siguiente comando:")
            print("[RAG+GPT4o]    python build_rag_index.py")
            print("[RAG+GPT4o] ")
            print("[RAG+GPT4o] Esto generará el índice una sola vez (tarda ~10 min)")
            print("[RAG+GPT4o] Después, la inicialización será instantánea.")
            print("[RAG+GPT4o] ")
            print("[RAG+GPT4o] ⚠️  Usando fallback sin Faiss (búsqueda simple)...")
            
            self.faiss_index = None
            self.embedding_model = None
            return
        
        try:
            print("[RAG+GPT4o] Cargando índice Faiss pre-construido...")
            
            # Cargar índice Faiss
            self.faiss_index = faiss.read_index(index_path)
            
            # Cargar metadata (opcional, para validación)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                print(f"[RAG+GPT4o] ✅ Índice cargado: {metadata['n_concepts']} conceptos")
                print(f"[RAG+GPT4o]    - Dimensión: {metadata['embedding_dim']}")
                print(f"[RAG+GPT4o]    - Modelo: {metadata['model_name']}")
                print(f"[RAG+GPT4o]    - Creado: {metadata['created_at'][:10]}")
            else:
                print(f"[RAG+GPT4o] ✅ Índice cargado: {self.faiss_index.ntotal} vectores")
            
            # Cargar modelo de embeddings (SOLO para consultas, NO para construir índice)
            # Esto es ligero porque no genera embeddings para toda la ontología
            print("[RAG+GPT4o] Cargando modelo de embeddings para consultas...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            print("[RAG+GPT4o] 🚀 Índice listo para búsqueda semántica")
            
        except Exception as e:
            print(f"[RAG+GPT4o] ❌ Error cargando índice Faiss: {e}")
            print("[RAG+GPT4o] ⚠️  Usando búsqueda simple como fallback...")
            self.faiss_index = None
            self.embedding_model = None
    
    def _setup_prompts(self):
        """Configura los prompts (mismos que tu notebook original)"""
        
        # Prompt NER (exactamente igual que tu notebook)
        self.ner_prompt_template = """
<task>
You are an agent that recognizes clinical entities in Spanish Computed Tomography reports of patients with suspected acute stroke.

**Entidades a Extraer:**
- hemorragia, lesión isquémica, escala ASPECTS, lesiones parenquimatosas, oclusiones arteriales, grado de estenosis, retraso en los tiempos de circulación, ratio penumbra-core

**Ubicaciones Anatómicas:**
- caudado, lenticular, cápsula interna, ribete insular, segmentos M1, M2, M3, M4, M5, arterias de cabeza y cuello, cerebelo, troncoencéfalo, territorios arteriales

**Reglas:**
- Detecta solo entidades presentes en el texto
- Para cada entidad, indica ubicación anatómica, presencia y valor si aplica
</task>

<output_format>
{{
  "findings": [
    {{
      "anatomical_location": "string",
      "presence": "string", 
      "entity": "string",
      "value": "string | null"
    }}
  ]
}}
</output_format>

<informe>
{informe}
</informe>

Responde ÚNICAMENTE con el JSON válido, sin texto adicional:
"""
        
        # Prompt de codificación (exactamente igual que tu notebook)
        self.coding_prompt_template = """
<task>
Eres un experto en terminología clínica. Asigna códigos apropiados de tu ontología a esta entidad clínica específica.

Entidad detectada: {entity}
Ubicación anatómica: {location}
Presencia: {presence}
Valor: {value}

Contexto ontológico disponible:
{contexto_ontologico}

Reglas:
- Usa los conceptos y códigos exactos del contexto ontológico proporcionado
- Si no hay coincidencia exacta, usa el concepto más similar
- Para presencia: presente (52101004), ausente (272519000), unknown (261665006)
</task>

Responde ÚNICAMENTE con este JSON exacto:
{{
  "anatomical_location": "{location}",
  "anatomy_terminology": "OWL_Ontology",
  "anatomy_code": "código_del_contexto_anatomy",
  "anatomy_description": "descripción_del_contexto_anatomy",
  "presence": "{presence}",
  "presence_terminology": "SNOMED-CT",
  "presence_code": "código_presencia_apropiado",
  "presence_description": "descripción_presencia",
  "entity": "{entity}",
  "entity_terminology": "OWL_Ontology", 
  "entity_code": "código_del_contexto_entity",
  "entity_description": "descripción_del_contexto_entity",
  "value": {value}
}}
"""
    
    def _call_gpt4o(self, prompt: str, max_retries: int = 3) -> str:
        """Llama a GPT-4o con manejo de errores"""
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": "Eres un experto en terminología médica SNOMED-CT especializado en ictus. Responde siempre con JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config["temperature"],
                    max_tokens=self.model_config["max_tokens"],
                    top_p=self.model_config["top_p"]
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"[RAG+GPT4o] Error en llamada GPT-4o (intento {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    return "{\"error\": \"GPT-4o no disponible\"}"
        
        return "{\"error\": \"GPT-4o falló\"}"
    
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
        """Ejecuta el Paso 1: NER básico con GPT-4o"""
        
        print("[RAG+GPT4o] Paso 1: Ejecutando NER con GPT-4o...")
        
        # Preparar prompt
        prompt_ner = self.ner_prompt_template.format(informe=texto)
        
        # Llamar a GPT-4o
        response = self._call_gpt4o(prompt_ner)
        
        # Parsear respuesta JSON
        try:
            # Limpiar respuesta (GPT-4o a veces agrega texto extra)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_content = json_match.group()
                entidades_basicas = json.loads(json_content)
            else:
                entidades_basicas = json.loads(response)
            
            # Extraer entidades
            entidades_detectadas = []
            if "findings" in entidades_basicas:
                for finding in entidades_basicas["findings"]:
                    entity = finding.get("entity", "")
                    if entity:
                        entidades_detectadas.append({
                            "entity": entity,
                            "anatomical_location": finding.get("anatomical_location", ""),
                            "presence": finding.get("presence", ""),
                            "value": finding.get("value")
                        })
            
            print(f"[RAG+GPT4o] Entidades detectadas: {len(entidades_detectadas)}")
            for ent in entidades_detectadas:
                print(f"  - {ent['entity']} en {ent['anatomical_location']} ({ent['presence']})")
            
            return entidades_detectadas
            
        except Exception as e:
            print(f"[RAG+GPT4o] Error parseando NER: {e}")
            print(f"[RAG+GPT4o] Respuesta GPT-4o: {response[:200]}...")
            return []
    
    def _execute_coding_step(self, entidades_detectadas: List[Dict]) -> List[Dict]:
        """Ejecuta el Paso 2: Codificación con RAG + GPT-4o"""
        
        print("[RAG+GPT4o] Paso 2: Generando contexto OWL para codificación...")
        entidades_codificadas = []
        
        for ent_data in entidades_detectadas:
            entity = ent_data["entity"]
            location = ent_data["anatomical_location"]
            presence = ent_data["presence"]
            value = ent_data.get("value")
            
            # Buscar conceptos similares para la entidad (tu RAG original)
            contexto_entity = ""
            if entity:
                similares_entity = self._recuperar_conceptos(entity, k=3)
                contexto_entity += f"Entidad '{entity}':\n"
                for concepto, narrativa, dist in similares_entity:
                    contexto_entity += f"- {concepto}: {narrativa}\n"
            
            # Buscar conceptos similares para la ubicación anatómica
            contexto_anatomy = ""
            if location and location != "No especificado":
                similares_anatomy = self._recuperar_conceptos(location, k=3)
                contexto_anatomy += f"Ubicación '{location}':\n"
                for concepto, narrativa, dist in similares_anatomy:
                    contexto_anatomy += f"- {concepto}: {narrativa}\n"
            
            contexto_ontologico = contexto_entity + "\n" + contexto_anatomy
            
            # Preparar prompt de codificación
            prompt_coding = self.coding_prompt_template.format(
                entity=entity,
                location=location,
                presence=presence,
                value=json.dumps(value) if value else "null",
                contexto_ontologico=contexto_ontologico
            )
            
            # Llamar a GPT-4o para codificación
            response = self._call_gpt4o(prompt_coding)
            
            try:
                # Parsear respuesta de codificación
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    coded_entity = json.loads(json_match.group())
                else:
                    coded_entity = json.loads(response)
                
                entidades_codificadas.append(coded_entity)
                print(f"[RAG+GPT4o] Codificado: {entity} -> {coded_entity.get('entity_code', 'No asignado')}")
                
            except Exception as e:
                print(f"[RAG+GPT4o] Error codificando {entity}: {e}")
                # Fallback con estructura básica
                entidades_codificadas.append({
                    "entity": entity,
                    "entity_code": "404684003",  # Clinical finding
                    "entity_description": entity,
                    "anatomical_location": location,
                    "anatomy_code": "12738006",  # Brain structure
                    "presence": presence,
                    "presence_code": "52101004" if presence == "presente" else "272519000",
                    "value": value
                })
        
        return entidades_codificadas
    
    def extract_entities(self, texto: str) -> List[Dict]:
        """
        Pipeline completo de tu RAG original con GPT-4o:
        1. NER básico con GPT-4o
        2. RAG + Codificación con GPT-4o
        """
        
        # Paso 1: NER con GPT-4o
        entidades_detectadas = self._execute_ner_step(texto)
        
        if not entidades_detectadas:
            return []
        
        # Paso 2: Codificación RAG + GPT-4o
        entidades_codificadas = self._execute_coding_step(entidades_detectadas)
        
        return entidades_codificadas
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice entidades usando tu RAG original con GPT-4o
        """
        print(f"[RAG+GPT4o] Procesando {len(notes_df)} notas con RAG + GPT-4o...")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            print(f"\n[RAG+GPT4o] === Procesando nota {note_id} ({idx+1}/{len(notes_df)}) ===")
            
            # Aplicar tu pipeline completo con GPT-4o
            entities = self.extract_entities(text)
            
            for entity in entities:
                predictions.append({
                    'note_id': note_id,
                    'start': 0,  # GPT-4o no devuelve posiciones exactas
                    'end': len(entity.get('entity', '')),
                    'concept_id': entity.get('entity_code', '404684003'),
                    'span_text': entity.get('entity', ''),
                    'confidence': 0.85,  # Confianza típica de tu sistema
                    'entity_description': entity.get('entity_description', ''),
                    'anatomy_code': entity.get('anatomy_code', ''),
                    'presence_code': entity.get('presence_code', ''),
                    'llm_used': 'GPT-4o'
                })
            
            print(f"[RAG+GPT4o] Nota {note_id}: {len(entities)} entidades extraídas")
        
        print(f"\n[RAG+GPT4o] Completado: {len(predictions)} predicciones generadas con GPT-4o")
        return pd.DataFrame(predictions)
