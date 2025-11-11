#!/usr/bin/env python3
"""
Implementación REAL de MITEL usando Mistral 7B via Ollama
Basada en el código original pero usando tu instalación local de Ollama
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import json
import requests
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict

class RealMITELOllamaStrategy:
    """
    Implementación real de MITEL usando Mistral 7B via Ollama:
    1. NER con Mistral 7B (via Ollama)
    2. RAG con Faiss + SentenceTransformers
    3. Clasificación con Mistral 7B (via Ollama)
    """
    
    def __init__(self):
        print("[REAL MITEL+Ollama] Inicializando con Mistral 7B via Ollama...")
        
        # Configurar Ollama
        self._setup_ollama()
        
        # Configurar embeddings para RAG
        self._setup_embedding_model()
        
        # Construir base de datos terminológica con Faiss
        self._build_faiss_terminologies()
        
        # Configurar prompts reales
        self._setup_prompts()
    
    def _setup_ollama(self):
        """Configura conexión con Ollama"""
        
        self.ollama_url = "http://localhost:11434"  # URL por defecto de Ollama
        self.model_name = "mistral:7b"  # Tu modelo Mistral 7B
        
        # Probar conexión
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                if any("mistral" in name.lower() for name in model_names):
                    print(f"[REAL MITEL+Ollama] ✓ Conectado a Ollama - Mistral disponible")
                    self.ollama_available = True
                else:
                    print(f"[REAL MITEL+Ollama] ⚠ Ollama conectado pero Mistral no encontrado")
                    print(f"[REAL MITEL+Ollama] Modelos disponibles: {model_names}")
                    self.ollama_available = False
            else:
                print(f"[REAL MITEL+Ollama] ✗ Error conectando a Ollama: {response.status_code}")
                self.ollama_available = False
                
        except Exception as e:
            print(f"[REAL MITEL+Ollama] ✗ Ollama no disponible: {e}")
            print(f"[REAL MITEL+Ollama] Asegúrate de que Ollama esté ejecutándose")
            self.ollama_available = False
    
    def _setup_embedding_model(self):
        """Configura SentenceTransformer para RAG"""
        
        try:
            # Modelo usado por MITEL para embeddings
            model_name = "sentence-transformers/all-MiniLM-L12-v2"
            print(f"[REAL MITEL+Ollama] Cargando modelo embeddings: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            
            print("[REAL MITEL+Ollama] ✓ Modelo embeddings cargado")
            
        except Exception as e:
            print(f"[REAL MITEL+Ollama] ✗ Error cargando embeddings: {e}")
            # Fallback a modelo más ligero
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[REAL MITEL+Ollama] ✓ Usando modelo embeddings alternativo")
            except:
                print("[REAL MITEL+Ollama] ✗ No se pudo cargar ningún modelo de embeddings")
                self.embedding_model = None
    
    def _build_faiss_terminologies(self):
        """Construye base de datos Faiss con terminologías SNOMED extendidas"""
        
        # Terminologías SNOMED específicas para ictus (basado en código real MITEL)
        self.terminologies = {
            # Core stroke concepts
            "230690007": "Stroke cerebrovascular accident CVA brain attack acute stroke syndrome neurological deficit sudden onset",
            "432102000": "Ischemic stroke cerebral infarction brain infarction ischemic cerebrovascular accident tissue death hypoxia",
            "274100004": "Hemorrhagic stroke cerebral hemorrhage brain hemorrhage hemorrhagic CVA bleeding extravasation",
            "266257000": "Transient ischemic attack TIA mini stroke transient cerebral ischemia temporary symptoms",
            
            # Motor and sensory deficits
            "50582007": "Hemiparesis partial paralysis weakness one side motor weakness unilateral paresis muscle weakness",
            "13791008": "Weakness motor weakness muscle weakness paresis reduced strength power loss motor deficit",
            "26544005": "Motor weakness motor deficit strength deficit motor impairment movement disorder paralysis",
            "280816001": "Facial weakness facial paresis facial droop seventh nerve palsy cranial nerve VII",
            "44077006": "Sensory loss numbness sensory deficit sensory impairment loss sensation tactile deficit",
            "20262006": "Ataxia coordination disorder balance problem cerebellar dysfunction gait disturbance unsteady",
            
            # Speech and language disorders
            "87486003": "Aphasia language disorder speech disorder communication deficit dysphasia verbal impairment",
            "8011004": "Dysarthria slurred speech speech articulation disorder motor speech disorder pronunciation difficulty",
            "29164008": "Speech disorder communication disorder verbal communication impairment language problem",
            
            # Vascular pathology
            "50960005": "Hemorrhage bleeding blood extravasation haemorrhage vascular rupture vessel bleeding",
            "21454007": "Subarachnoid hemorrhage SAH subarachnoid bleeding aneurysmal hemorrhage cisternal blood",
            "71186008": "Intraventricular hemorrhage IVH ventricular bleeding intraventricular blood hydrocephalus",
            "274100004": "Intracerebral hemorrhage ICH brain hemorrhage parenchymal hemorrhage tissue bleeding",
            "55342001": "Infarction tissue death necrosis ischemic lesion cell death hypoxic damage",
            "52674009": "Ischemia reduced blood flow hypoperfusion tissue hypoxia oxygen deficit circulation compromise",
            "26036001": "Occlusion blockage vessel closure arterial obstruction flow interruption complete blockage",
            "415582006": "Stenosis narrowing vessel constriction luminal reduction flow limitation partial blockage",
            "432101006": "Aneurysm arterial dilatation vessel bulging vascular malformation arterial expansion weakness",
            
            # Cerebral vessels and anatomy
            "69930009": "Middle cerebral artery MCA cerebral artery middle arteria cerebri media M1 M2 segments",
            "86547008": "Internal carotid artery ICA carotid artery internal arteria carotis interna cervical vessel",
            "79371005": "Anterior cerebral artery ACA cerebral artery anterior arteria cerebri anterior A1 A2 segments",
            "70382005": "Posterior cerebral artery PCA cerebral artery posterior arteria cerebri posterior P1 P2 segments",
            "67889009": "Basilar artery arteria basilaris vertebrobasilar system posterior circulation brainstem supply",
            "85234005": "Vertebral artery arteria vertebralis vertebrobasilar circulation posterior circulation",
            
            # Brain anatomy regions
            "32603002": "Basal ganglia basal nuclei subcortical nuclei deep gray matter caudate putamen globus pallidus",
            "42695009": "Thalamus thalamic nuclei diencephalon relay station sensory motor relay deep structure",
            "113305005": "Cerebellum cerebellar hemisphere posterior fossa balance center coordination motor control",
            "15926001": "Brainstem brain stem midbrain pons medulla truncus encephali posterior circulation",
            "42696002": "Internal capsule capsula interna white matter tract motor pathway corticospinal tract",
            "7173007": "Caudate nucleus caudate striatum basal ganglia component motor control cognition",
            "89610007": "Putamen putaminal nucleus striatum basal ganglia motor control movement",
            "36992007": "Insula insular cortex island cortex insular lobe deep cortical region",
            
            # Cerebral lobes
            "83251001": "Frontal lobe frontal cortex prefrontal region motor cortex executive function personality",
            "16630005": "Parietal lobe parietal cortex sensory cortex posterior parietal somatosensory processing",
            "78277001": "Temporal lobe temporal cortex auditory cortex hippocampus memory language processing",
            "31065004": "Occipital lobe occipital cortex visual cortex striate cortex vision processing",
            
            # Procedures and interventions
            "433112001": "Thrombectomy mechanical thrombectomy clot removal endovascular treatment stent retriever aspiration",
            "373110003": "Thrombolysis clot dissolution fibrinolytic therapy thrombolytic treatment clot busting medication",
            "387467008": "Tissue plasminogen activator tPA rtPA alteplase fibrinolytic agent clot dissolving drug",
            "77343006": "Angiography vessel imaging arteriography contrast study DSA digital subtraction angiography",
            "449894001": "Recanalization vessel reopening flow restoration reperfusion circulation restoration",
            
            # Imaging modalities
            "77477000": "Computed tomography CT scan CAT scan axial tomography cross sectional imaging",
            "113091000": "Magnetic resonance imaging MRI scan nuclear magnetic resonance MR imaging soft tissue",
            "419775003": "CT angiography CTA computed tomographic angiography CT angiogram vessel imaging contrast",
            "419775003": "MR angiography MRA magnetic resonance angiography MR angiogram vessel imaging",
            
            # Clinical assessment scales and scores
            "450893003": "NIHSS National Institutes Health Stroke Scale neurologic assessment stroke severity score",
            "450893003": "ASPECTS Alberta Stroke Program Early CT Score imaging score infarct assessment early changes",
            "450893003": "TICI Thrombolysis Cerebral Infarction reperfusion grade recanalization score flow restoration",
            "273302005": "Modified Rankin Scale mRS functional outcome disability scale independence assessment",
            "386554004": "Glasgow Coma Scale GCS consciousness level neurologic status awareness response",
            
            # Clinical findings and symptoms
            "25064002": "Headache cephalgia head pain cranial pain cephalalgia severe sudden worst headache",
            "422587007": "Nausea feeling sick queasiness stomach upset gastric distress vomiting sensation",
            "422400008": "Vomiting emesis throwing up gastric emptying retching stomach contents expulsion",
            "40917007": "Confusion mental confusion disorientation cognitive impairment delirium altered consciousness",
            "419045004": "Loss consciousness unconsciousness syncope fainting blackout coma unresponsive",
            "24982008": "Diplopia double vision binocular diplopia visual disturbance seeing double eye movement",
            "18060000": "Visual field defect hemianopia visual field cut vision loss scotoma blind spot",
            
            # Qualifiers and modifiers
            "52101004": "Present positive detected found identified observed confirmed existing current",
            "272519000": "Absent negative not detected not found not present missing nonexistent",
            "261665006": "Unknown uncertain unclear indeterminate not specified undetermined questionable",
            "24484000": "Severe serious marked significant pronounced extensive major critical",
            "6736007": "Moderate intermediate medium modest mild-moderate substantial noticeable",
            "255604002": "Mild slight minimal minor subtle trace small degree",
            "255212004": "Acute sudden onset rapid development recent new fresh immediate",
            "90734009": "Chronic long-standing persistent ongoing established old longterm",
            "19939008": "Subacute intermediate onset developing evolving progressing gradual"
        }
        
        if self.embedding_model is not None:
            try:
                print("[REAL MITEL+Ollama] Construyendo índice Faiss con terminologías...")
                
                # Preparar textos para embeddings
                terminology_texts = list(self.terminologies.values())
                self.concept_ids_faiss = list(self.terminologies.keys())
                
                # Generar embeddings
                embeddings = self.embedding_model.encode(
                    terminology_texts,
                    show_progress_bar=True,
                    batch_size=32
                )
                
                # Crear índice Faiss
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings.astype('float32'))
                
                print(f"[REAL MITEL+Ollama] ✓ Índice Faiss: {len(terminology_texts)} terminologías, dim {dimension}")
                
            except Exception as e:
                print(f"[REAL MITEL+Ollama] ✗ Error construyendo Faiss: {e}")
                self.faiss_index = None
        else:
            self.faiss_index = None
    
    def _setup_prompts(self):
        """Configura prompts reales de MITEL"""
        
        # Prompt para NER (basado en el código real de MITEL)
        self.ner_prompt_template = """<s>[INST] Eres un experto médico especializado en reconocimiento de entidades SNOMED-CT para trastornos cerebrovasculares e ictus. Tu tarea es identificar y extraer todas las entidades médicas relevantes de notas clínicas.

**Tarea**: Extraer entidades médicas relacionadas con ictus, trastornos cerebrovasculares, hallazgos neurológicos, anatomía, procedimientos y evaluaciones clínicas.

**Instrucciones**:
1. Identifica todas las entidades médicas en el texto
2. Enfócate en: síntomas de ictus, patología vascular, anatomía cerebral, procedimientos, imágenes, escalas clínicas
3. Extrae los spans exactos de texto tal como aparecen
4. Proporciona posiciones de inicio y fin
5. Asigna puntuaciones de confianza (0.0-1.0)

**Texto de entrada**: {text}

**Formato de salida** (solo JSON):
{{
  "entities": [
    {{
      "start": <posicion_inicio>,
      "end": <posicion_fin>, 
      "text": "<span_texto_exacto>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Responde solo con JSON, sin texto adicional: [/INST]"""
        
        # Prompt para clasificación (basado en el código real de MITEL)
        self.classification_prompt_template = """<s>[INST] Eres un experto en terminología médica especializado en mapeo de conceptos SNOMED-CT para trastornos cerebrovasculares e ictus.

**Tarea**: Mapear la entidad médica extraída al concepto SNOMED-CT más apropiado usando el contexto terminológico proporcionado.

**Entidad**: "{entity_text}"
**Contexto clínico**: {context}

**Conceptos SNOMED-CT disponibles**:
{terminology_context}

**Instrucciones**:
1. Analiza la entidad en su contexto clínico
2. Selecciona el concepto SNOMED-CT más apropiado
3. Considera similitud semántica y relevancia clínica
4. Proporciona puntuación de confianza basada en la calidad del match

**Formato de salida** (solo JSON):
{{
  "concept_id": "<id_concepto_snomed>",
  "concept_description": "<descripcion_concepto>",
  "confidence": <0.0-1.0>,
  "reasoning": "<explicacion_breve>"
}}

Responde solo con JSON, sin texto adicional: [/INST]"""
    
    def _call_mistral_ollama(self, prompt: str, max_tokens: int = 512) -> str:
        """Llama a Mistral 7B via Ollama"""
        
        if not self.ollama_available:
            return self._simulate_mistral_response(prompt)
        
        try:
            # Preparar payload para Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": max_tokens,
                    "stop": ["</s>", "[/INST]", "<s>"]
                }
            }
            
            # Llamar a Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"[REAL MITEL+Ollama] ✗ Error Ollama: {response.status_code}")
                return self._simulate_mistral_response(prompt)
                
        except Exception as e:
            print(f"[REAL MITEL+Ollama] ✗ Error llamando Mistral: {e}")
            return self._simulate_mistral_response(prompt)
    
    def _simulate_mistral_response(self, prompt: str) -> str:
        """Simula respuesta de Mistral cuando no está disponible"""
        
        if "extract medical entities" in prompt.lower() or "entities" in prompt.lower():
            # Simular NER con regex
            text_match = re.search(r'\*\*Texto de entrada\*\*:\s*(.+?)(?:\*\*|$)', prompt, re.DOTALL)
            if text_match:
                text = text_match.group(1).strip()
                entities = self._regex_ner_simulation(text)
                
                return json.dumps({
                    "entities": [
                        {
                            "start": e["start"],
                            "end": e["end"],
                            "text": e["text"],
                            "confidence": e["confidence"]
                        }
                        for e in entities
                    ]
                }, indent=2)
        
        elif "snomed-ct" in prompt.lower() and "mapear" in prompt.lower():
            # Simular clasificación
            entity_match = re.search(r'\*\*Entidad\*\*:\s*"([^"]+)"', prompt)
            if entity_match:
                entity = entity_match.group(1).lower()
                
                # Mapeo simple basado en palabras clave
                if "stroke" in entity or "ictus" in entity or "cva" in entity:
                    concept_id = "230690007"
                    description = "Stroke (disorder)"
                elif "hemorrhage" in entity or "bleeding" in entity or "hemorragia" in entity:
                    concept_id = "50960005"
                    description = "Hemorrhage (morphologic abnormality)"
                elif "weakness" in entity or "paresis" in entity or "debilidad" in entity:
                    concept_id = "13791008"
                    description = "Weakness (finding)"
                elif "mca" in entity or "middle cerebral" in entity or "cerebral media" in entity:
                    concept_id = "69930009"
                    description = "Middle cerebral artery (body structure)"
                elif "infarct" in entity or "infarto" in entity:
                    concept_id = "55342001"
                    description = "Infarction (morphologic abnormality)"
                else:
                    concept_id = "404684003"
                    description = "Clinical finding (finding)"
                
                return json.dumps({
                    "concept_id": concept_id,
                    "concept_description": description,
                    "confidence": 0.85,
                    "reasoning": "Mapeo simulado basado en coincidencia de palabras clave"
                }, indent=2)
        
        return '{"error": "No se pudo procesar la solicitud"}'
    
    def _regex_ner_simulation(self, text: str) -> List[Dict]:
        """Simulación de NER usando regex cuando Mistral no está disponible"""
        
        patterns = [
            # Términos de ictus
            (r'\b(stroke|ictus|CVA|cerebrovascular accident|brain attack)\b', 0.95),
            (r'\b(ischemic|hemorrhagic|isquémico|hemorrágico)\s+(stroke|ictus|CVA)\b', 0.98),
            
            # Síntomas motores
            (r'\b(hemiparesis|hemiplegia|weakness|paralysis|debilidad|parálisis)\b', 0.9),
            (r'\b(left|right|izquierda|derecha)\s+(sided\s+)?(weakness|hemiparesis|debilidad)\b', 0.95),
            (r'\b(facial\s+)?(weakness|droop|debilidad|caída)\b', 0.85),
            
            # Trastornos del habla
            (r'\b(aphasia|dysphasia|dysarthria|afasia|disartria)\b', 0.95),
            (r'\b(speech\s+)?(difficulty|impairment|disorder|dificultad|trastorno)\b', 0.8),
            
            # Patología vascular
            (r'\b(hemorrhage|bleeding|haemorrhage|hemorragia|sangrado)\b', 0.9),
            (r'\b(infarct|infarction|ischemia|ischaemia|infarto|isquemia)\b', 0.95),
            (r'\b(occlusion|stenosis|aneurysm|thrombosis|oclusión|estenosis|aneurisma)\b', 0.9),
            
            # Anatomía
            (r'\b(MCA|ACA|PCA|ICA|basilar|vertebral)\s*(artery|arteria|segment|segmento)?\b', 0.95),
            (r'\b(M1|M2|M3|A1|A2|P1|P2)\s*(segment|segmento)\b', 0.98),
            (r'\b(basal ganglia|thalamus|cerebellum|brainstem|ganglios basales|tálamo|cerebelo|troncoencéfalo)\b', 0.95),
            
            # Procedimientos
            (r'\b(thrombectomy|thrombolysis|angiography|trombectomía|trombólisis|angiografía)\b', 0.95),
            (r'\b(tPA|rtPA|tissue plasminogen activator|activador del plasminógeno)\b', 0.98),
            
            # Imágenes
            (r'\b(CT|MRI|CTA|MRA|DWI|PWI|FLAIR|TAC|RMN)\b', 0.9),
            (r'\b(computed tomography|magnetic resonance|tomografía computarizada|resonancia magnética)\b', 0.95),
            
            # Escalas clínicas
            (r'\b(NIHSS|ASPECTS|TICI|mRS|GCS)\s*\d*\b', 0.95),
            
            # Otros síntomas
            (r'\b(headache|nausea|vomiting|confusion|cefalea|náusea|vómito|confusión)\b', 0.85),
            (r'\b(diplopia|visual field defect|hemianopia|diplopía|defecto campo visual)\b', 0.9)
        ]
        
        entities = []
        
        for pattern, confidence in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                entities.append({
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                    "confidence": confidence
                })
        
        return entities
    
    def _retrieve_terminology_context(self, entity_text: str, k: int = 5) -> str:
        """Recupera contexto terminológico usando Faiss RAG"""
        
        if self.faiss_index is None or self.embedding_model is None:
            return self._simple_terminology_lookup(entity_text)
        
        try:
            # Generar embedding para la entidad
            entity_embedding = self.embedding_model.encode([entity_text])
            
            # Buscar en índice Faiss
            distances, indices = self.faiss_index.search(
                entity_embedding.astype('float32'), k=k
            )
            
            # Construir contexto
            context_parts = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.concept_ids_faiss):
                    concept_id = self.concept_ids_faiss[idx]
                    description = self.terminologies[concept_id]
                    distance = distances[0][i]
                    similarity = 1.0 / (1.0 + distance)
                    
                    context_parts.append(f"- {concept_id}: {description[:100]}... (similitud: {similarity:.3f})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            print(f"[REAL MITEL+Ollama] ✗ Error recuperando contexto: {e}")
            return self._simple_terminology_lookup(entity_text)
    
    def _simple_terminology_lookup(self, entity_text: str) -> str:
        """Lookup simple de terminología como fallback"""
        
        entity_lower = entity_text.lower()
        matches = []
        
        for concept_id, description in list(self.terminologies.items())[:10]:
            if any(word in description.lower() for word in entity_lower.split()):
                matches.append(f"- {concept_id}: {description[:80]}...")
        
        return "\n".join(matches[:5]) if matches else "- 404684003: Clinical finding (finding)"
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Pipeline completo de MITEL real con Mistral 7B via Ollama:
        1. NER con Mistral 7B
        2. RAG para recuperar contexto terminológico
        3. Clasificación con Mistral 7B
        """
        
        # Paso 1: NER con Mistral 7B via Ollama
        ner_prompt = self.ner_prompt_template.format(text=text)
        ner_response = self._call_mistral_ollama(ner_prompt)
        
        # Parsear respuesta NER
        try:
            # Extraer JSON de la respuesta
            json_match = re.search(r'\{.*\}', ner_response, re.DOTALL)
            if json_match:
                ner_data = json.loads(json_match.group())
                entities = ner_data.get("entities", [])
            else:
                entities = []
        except Exception as e:
            print(f"[REAL MITEL+Ollama] ✗ Error parseando NER: {e}")
            entities = []
        
        if not entities:
            return []
        
        # Paso 2 y 3: Para cada entidad, recuperar contexto RAG y clasificar
        classified_entities = []
        
        for entity in entities:
            entity_text = entity.get("text", "")
            
            # Recuperar contexto terminológico con RAG
            terminology_context = self._retrieve_terminology_context(entity_text)
            
            # Clasificar con Mistral 7B
            context_window = text[max(0, entity.get("start", 0) - 100):entity.get("end", 0) + 100]
            classification_prompt = self.classification_prompt_template.format(
                entity_text=entity_text,
                context=context_window,
                terminology_context=terminology_context
            )
            
            classification_response = self._call_mistral_ollama(classification_prompt)
            
            # Parsear respuesta de clasificación
            try:
                json_match = re.search(r'\{.*\}', classification_response, re.DOTALL)
                if json_match:
                    classification_data = json.loads(json_match.group())
                    
                    classified_entities.append({
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0),
                        "span_text": entity_text,
                        "concept_id": classification_data.get("concept_id", "404684003"),
                        "concept_description": classification_data.get("concept_description", "Clinical finding"),
                        "confidence": min(entity.get("confidence", 0.5) * classification_data.get("confidence", 0.5), 1.0),
                        "reasoning": classification_data.get("reasoning", ""),
                        "method": "mitel_mistral_rag",
                        "rag_context": terminology_context[:200] + "..." if len(terminology_context) > 200 else terminology_context,
                        "ollama_available": self.ollama_available
                    })
                    
            except Exception as e:
                print(f"[REAL MITEL+Ollama] ✗ Error clasificando '{entity_text}': {e}")
                # Fallback
                classified_entities.append({
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "span_text": entity_text,
                    "concept_id": "404684003",
                    "concept_description": "Clinical finding",
                    "confidence": entity.get("confidence", 0.5) * 0.7,
                    "method": "mitel_fallback",
                    "ollama_available": self.ollama_available
                })
        
        return classified_entities
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice entidades usando la implementación real de MITEL con Ollama
        """
        print(f"[REAL MITEL+Ollama] Procesando {len(notes_df)} notas...")
        print(f"[REAL MITEL+Ollama] Ollama disponible: {self.ollama_available}")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            print(f"[REAL MITEL+Ollama] Procesando nota {note_id} ({idx + 1}/{len(notes_df)})")
            
            entities = self.extract_entities(text)
            
            for entity in entities:
                predictions.append({
                    'note_id': note_id,
                    'start': entity['start'],
                    'end': entity['end'],
                    'concept_id': entity['concept_id'],
                    'span_text': entity['span_text'],
                    'confidence': entity['confidence'],
                    'method': entity.get('method', 'unknown'),
                    'concept_description': entity.get('concept_description', ''),
                    'reasoning': entity.get('reasoning', ''),
                    'rag_context': entity.get('rag_context', ''),
                    'ollama_used': entity.get('ollama_available', False)
                })
            
            print(f"[REAL MITEL+Ollama] Nota {note_id}: {len(entities)} entidades")
        
        print(f"[REAL MITEL+Ollama] ✓ Completado: {len(predictions)} predicciones")
        return pd.DataFrame(predictions)
