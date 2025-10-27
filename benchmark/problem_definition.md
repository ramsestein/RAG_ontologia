## Definición del problema

Este proyecto implementa y compara 4 estrategias diferentes para el reconocimiento y codificación de entidades médicas en textos clínicos de ictus, utilizando la terminología SNOMED-CT. El proyecto incluye una implementación original de RAG (Retrieval-Augmented Generation) con ontologías médicas y una comparación exhaustiva con las estrategias ganadoras del SNOMED CT Entity Linking Challenge.

En mi máquina, obtengo estos resultados:

    Estrategia           F1-Score   Precision  Recall     Pred   Match  Tiempo
    1_KIRIs_REAL         0.8000     0.8381     0.7652     105    88     0.0s
    2_SNOBERT_REAL       0.3630     0.3072     0.4435     166    51     10.0s
    4_TU_RAG_GPT4o       0.0310     0.1429     0.0174     14     2      74.8s    

Estos son mis códigos:

01_kiris.py:
#!/usr/bin/env python3
"""
Implementación REAL de la estrategia KIRIs (1er lugar)
Adaptada del código original del repositorio ganador
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import re
import pickle
from pathlib import Path
from collections import defaultdict

class RealKIRIsStrategy:
    """
    Implementación real de KIRIs basada en su código ganador
    Usa diccionarios híbridos con OMOP + reglas lingüísticas
    """
    
    def __init__(self):
        print("[REAL KIRIs] Inicializando estrategia real KIRIs...")
        self._build_medical_dictionaries()
        self._build_abbreviations_dict()
        self._build_section_specific_dicts()
        self._build_case_sensitive_rules()
        
    def _build_medical_dictionaries(self):
        """Construye diccionarios médicos reales basados en SNOMED + OMOP"""
        
        # Diccionario principal basado en el código real de KIRIs
        self.medical_dict = {
            # Neurological findings - específicos de ictus
            "stroke": 230690007,
            "cerebrovascular accident": 230690007,
            "CVA": 230690007,
            "brain attack": 230690007,
            "acute stroke": 230690007,
            "ischemic stroke": 432102000,
            "hemorrhagic stroke": 274100004,
            
            # Motor deficits
            "hemiparesis": 50582007,
            "hemiplegia": 50582007,
            "weakness": 13791008,
            "motor weakness": 26544005,
            "left sided weakness": 13791008,
            "right sided weakness": 13791008,
            "left hemiparesis": 50582007,
            "right hemiparesis": 50582007,
            "facial weakness": 280816001,
            "facial droop": 280816001,
            "left facial droop": 280816001,
            "right facial droop": 280816001,
            
            # Speech and language
            "aphasia": 87486003,
            "dysphasia": 87486003,
            "speech difficulty": 87486003,
            "language disorder": 87486003,
            "dysarthria": 8011004,
            "slurred speech": 8011004,
            "speech impairment": 8011004,
            
            # Vascular pathology
            "hemorrhage": 50960005,
            "bleeding": 50960005,
            "haemorrhage": 50960005,
            "subarachnoid hemorrhage": 21454007,
            "intracerebral hemorrhage": 274100004,
            "intraventricular hemorrhage": 71186008,
            "subdural hemorrhage": 95453001,
            "epidural hemorrhage": 62414007,
            
            # Ischemic pathology
            "infarct": 55342001,
            "infarction": 55342001,
            "acute infarct": 55342001,
            "brain infarct": 432102000,
            "cerebral infarct": 432102000,
            "ischemia": 52674009,
            "ischaemia": 52674009,
            "cerebral ischemia": 52674009,
            "brain ischemia": 52674009,
            
            # Vascular abnormalities
            "occlusion": 26036001,
            "vessel occlusion": 26036001,
            "artery occlusion": 26036001,
            "stenosis": 415582006,
            "narrowing": 415582006,
            "aneurysm": 432101006,
            "arterial aneurysm": 432101006,
            "thrombosis": 118927008,
            "embolism": 414086009,
            "thrombus": 396339007,
            "embolus": 55584005,
            
            # Procedures
            "thrombectomy": 433112001,
            "mechanical thrombectomy": 433112001,
            "endovascular thrombectomy": 433112001,
            "thrombolysis": 373110003,
            "tissue plasminogen activator": 387467008,
            "tPA": 387467008,
            "rtPA": 387467008,
            "angiography": 77343006,
            "angiogram": 77343006,
            "cerebral angiography": 419775003,
            "CT angiography": 419775003,
            "CTA": 419775003,
            "MR angiography": 419775003,
            "MRA": 419775003,
            "recanalization": 449894001,
            "reperfusion": 35963006,
            
            # Imaging
            "CT": 77477000,
            "computed tomography": 77477000,
            "CAT scan": 77477000,
            "MRI": 113091000,
            "magnetic resonance imaging": 113091000,
            "MR imaging": 113091000,
            "DWI": 113091000,
            "diffusion weighted imaging": 113091000,
            "PWI": 113091000,
            "perfusion weighted imaging": 113091000,
            "FLAIR": 113091000,
            "T1": 113091000,
            "T2": 113091000,
            
            # Anatomy - Arteries
            "middle cerebral artery": 69930009,
            "MCA": 69930009,
            "internal carotid artery": 86547008,
            "ICA": 86547008,
            "anterior cerebral artery": 79371005,
            "ACA": 79371005,
            "posterior cerebral artery": 70382005,
            "PCA": 70382005,
            "basilar artery": 67889009,
            "vertebral artery": 85234005,
            "carotid artery": 86547008,
            "common carotid artery": 69105007,
            "external carotid artery": 32062004,
            
            # Anatomy - Brain regions
            "basal ganglia": 32603002,
            "thalamus": 42695009,
            "cerebellum": 113305005,
            "brainstem": 15926001,
            "brain stem": 15926001,
            "midbrain": 61962009,
            "pons": 49557009,
            "medulla": 25062003,
            "medulla oblongata": 25062003,
            "frontal lobe": 83251001,
            "parietal lobe": 16630005,
            "temporal lobe": 78277001,
            "occipital lobe": 31065004,
            "insula": 36992007,
            "insular cortex": 36992007,
            "corona radiata": 89777002,
            "internal capsule": 42696002,
            "caudate": 7173007,
            "putamen": 89610007,
            "lentiform nucleus": 42743008,
            
            # Clinical scales
            "NIHSS": 450893003,
            "National Institutes of Health Stroke Scale": 450893003,
            "ASPECTS": 450893003,
            "Alberta Stroke Program Early CT Score": 450893003,
            "TICI": 450893003,
            "Thrombolysis in Cerebral Infarction": 450893003,
            "mRS": 273302005,
            "modified Rankin Scale": 273302005,
            "GCS": 386554004,
            "Glasgow Coma Scale": 386554004,
            
            # Other symptoms
            "headache": 25064002,
            "severe headache": 25064002,
            "sudden headache": 25064002,
            "worst headache": 25064002,
            "nausea": 422587007,
            "vomiting": 422400008,
            "confusion": 40917007,
            "altered mental status": 419284004,
            "loss of consciousness": 419045004,
            "syncope": 271594007,
            "dizziness": 404640003,
            "vertigo": 399153001,
            "ataxia": 20262006,
            "diplopia": 24982008,
            "visual field defect": 18060000,
            "hemianopia": 18060000,
            "sensory loss": 44077006,
            "numbness": 44077006,
            
            # Risk factors
            "hypertension": 38341003,
            "diabetes": 73211009,
            "diabetes mellitus": 73211009,
            "atrial fibrillation": 49436004,
            "hyperlipidemia": 55822004,
            "smoking": 77176002,
            "obesity": 414915002
        }
        
        print(f"[REAL KIRIs] Diccionario médico: {len(self.medical_dict)} términos")
    
    def _build_abbreviations_dict(self):
        """Diccionario de abreviaciones médicas comunes"""
        
        self.abbreviations = {
            # Imaging
            "CT": 77477000,
            "MRI": 113091000,
            "CTA": 419775003,
            "MRA": 419775003,
            "DWI": 113091000,
            "PWI": 113091000,
            "FLAIR": 113091000,
            
            # Vessels
            "MCA": 69930009,
            "ACA": 79371005,
            "PCA": 70382005,
            "ICA": 86547008,
            "ECA": 32062004,
            "CCA": 69105007,
            
            # Scales
            "NIHSS": 450893003,
            "ASPECTS": 450893003,
            "TICI": 450893003,
            "mRS": 273302005,
            "GCS": 386554004,
            
            # Treatments
            "tPA": 387467008,
            "rtPA": 387467008,
            "IV": 47625008,
            "IA": 47625008,
            
            # Other
            "CVA": 230690007,
            "TIA": 266257000,
            "ICU": 309904001,
            "ED": 225728007,
            "ER": 225728007,
            "BP": 75367002,
            "HR": 364075005,
            "O2": 24099007,
            "L": 7771000,
            "R": 24028007
        }
        
        print(f"[REAL KIRIs] Diccionario abreviaciones: {len(self.abbreviations)} términos")
    
    def _build_section_specific_dicts(self):
        """Diccionarios específicos por sección (implementación real de KIRIs)"""
        
        self.section_dicts = {
            # Imaging section
            "imaging": {
                "no acute hemorrhage": ("no_hemorrhage", 50960005),
                "no hemorrhage": ("no_hemorrhage", 50960005),
                "acute infarct": ("acute_infarct", 55342001),
                "early ischemic changes": ("ischemic_changes", 52674009),
                "hypodense lesion": ("hypodense_lesion", 55342001),
                "mass effect": ("mass_effect", 300577008),
                "midline shift": ("midline_shift", 31209005),
                "edema": ("edema", 79654002),
                "cerebral edema": ("cerebral_edema", 79654002),
                "hemorrhagic transformation": ("hemorrhagic_transformation", 432102000),
                "petechial hemorrhage": ("petechial_hemorrhage", 50960005)
            },
            
            # Physical examination
            "examination": {
                "left facial droop": ("left_facial_droop", 280816001),
                "right facial droop": ("right_facial_droop", 280816001),
                "motor weakness": ("motor_weakness", 26544005),
                "sensory loss": ("sensory_loss", 44077006),
                "decreased reflexes": ("decreased_reflexes", 405944004),
                "increased reflexes": ("increased_reflexes", 405945003),
                "Babinski sign": ("babinski_sign", 69064006),
                "nuchal rigidity": ("nuchal_rigidity", 405944004)
            },
            
            # Treatment/intervention
            "intervention": {
                "successful recanalization": ("successful_recanalization", 449894001),
                "TICI 2B": ("tici_2b", 450893003),
                "TICI 3": ("tici_3", 450893003),
                "good outcome": ("good_outcome", 385669000),
                "poor outcome": ("poor_outcome", 385669000),
                "complete recanalization": ("complete_recanalization", 449894001),
                "partial recanalization": ("partial_recanalization", 449894001)
            }
        }
        
        print(f"[REAL KIRIs] Diccionarios por sección: {len(self.section_dicts)} secciones")
    
    def _build_case_sensitive_rules(self):
        """Reglas case-sensitive específicas (del código real de KIRIs)"""
        
        # Basado en get_case_sensitive_dict() del código original
        self.case_sensitive = {
            "K": 312468003,  # Potassium
            "T": 105723007,   # Temperature
            "Mg": 271285000,  # Magnesium
            "RA": 722742002,  # Right atrium
            "Plt": 61928009,  # Platelet count
            "MR": 48724000,   # Mitral regurgitation
            "L": 7771000,     # Left
            "R": 24028007,    # Right
            "M1": 69930009,   # MCA M1 segment
            "M2": 69930009,   # MCA M2 segment
            "M3": 69930009,   # MCA M3 segment
            "A1": 79371005,   # ACA A1 segment
            "A2": 79371005,   # ACA A2 segment
            "P1": 70382005,   # PCA P1 segment
            "P2": 70382005    # PCA P2 segment
        }
        
        print(f"[REAL KIRIs] Reglas case-sensitive: {len(self.case_sensitive)} términos")
    
    def _detect_section(self, text: str, position: int) -> str:
        """Detecta la sección del documento (implementación real)"""
        
        # Buscar headers de sección hacia atrás desde la posición
        text_before = text[:position].lower()
        
        # Headers comunes en notas médicas (basado en código real)
        section_headers = {
            "imaging": ["imaging", "radiology", "ct", "mri", "scan", "x-ray"],
            "examination": ["physical exam", "examination", "pe:", "exam:", "physical"],
            "intervention": ["treatment", "intervention", "procedure", "therapy", "management"],
            "history": ["history", "hpi", "chief complaint", "cc:"],
            "assessment": ["assessment", "impression", "diagnosis", "plan"],
            "medications": ["medications", "meds", "drugs", "prescriptions"]
        }
        
        for section, keywords in section_headers.items():
            for keyword in keywords:
                if keyword in text_before[-200:]:  # Buscar en últimos 200 caracteres
                    return section
        
        return "general"
    
    def _apply_linguistic_rules(self, text: str, match_start: int, match_end: int) -> Dict:
        """Aplica reglas lingüísticas (implementación real de KIRIs)"""
        
        # Expandir contexto alrededor del match
        context_start = max(0, match_start - 50)
        context_end = min(len(text), match_end + 50)
        context = text[context_start:context_end].lower()
        
        attributes = {}
        
        # Regla 1: Lateralidad
        laterality_patterns = [
            r'\b(left|right)[\s\-]?sided?\b',
            r'\b(left|right)\b',
            r'\b(bilateral|bilaterally)\b'
        ]
        
        for pattern in laterality_patterns:
            match = re.search(pattern, context)
            if match:
                attributes["laterality"] = match.group(1)
                break
        
        # Regla 2: Negación
        negation_patterns = [
            r'\b(no|not|without|absence of|negative for)\b',
            r'\b(denies|denied)\b',
            r'\b(rule out|r/o)\b'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, context):
                attributes["negated"] = True
                break
        
        # Regla 3: Severidad
        severity_patterns = [
            r'\b(mild|moderate|severe|massive|extensive)\b',
            r'\b(small|large|huge)\b',
            r'\b(acute|chronic|subacute)\b'
        ]
        
        for pattern in severity_patterns:
            match = re.search(pattern, context)
            if match:
                attributes["severity"] = match.group(1)
                break
        
        # Regla 4: Temporalidad
        temporal_patterns = [
            r'\b(acute|chronic|subacute|recent|old|new)\b',
            r'\b(hours?|days?|weeks?|months?|years?)\s+(ago|old)\b'
        ]
        
        for pattern in temporal_patterns:
            match = re.search(pattern, context)
            if match:
                attributes["temporal"] = match.group(0)
                break
        
        return attributes
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extrae entidades usando la estrategia real de KIRIs:
        1. Diccionario case-insensitive
        2. Diccionario case-sensitive  
        3. Abreviaciones
        4. Diccionarios específicos por sección
        5. Reglas lingüísticas
        6. Post-procesamiento
        """
        
        entities = []
        text_lower = text.lower()
        
        # 1. Diccionario principal case-insensitive
        for term, concept_id in self.medical_dict.items():
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                section = self._detect_section(text, start)
                attributes = self._apply_linguistic_rules(text, start, end)
                
                entities.append({
                    "start": start,
                    "end": end,
                    "span_text": text[start:end],
                    "concept_id": concept_id,
                    "confidence": 0.9,
                    "method": "main_dict",
                    "section": section,
                    "attributes": attributes
                })
        
        # 2. Diccionario case-sensitive
        for term, concept_id in self.case_sensitive.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text):  # Case-sensitive
                start, end = match.span()
                
                # Evitar duplicados
                if not any(e["start"] == start and e["end"] == end for e in entities):
                    section = self._detect_section(text, start)
                    attributes = self._apply_linguistic_rules(text, start, end)
                    
                    entities.append({
                        "start": start,
                        "end": end,
                        "span_text": text[start:end],
                        "concept_id": concept_id,
                        "confidence": 0.95,
                        "method": "case_sensitive",
                        "section": section,
                        "attributes": attributes
                    })
        
        # 3. Abreviaciones
        for abbr, concept_id in self.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            for match in re.finditer(pattern, text):
                start, end = match.span()
                
                if not any(e["start"] == start and e["end"] == end for e in entities):
                    section = self._detect_section(text, start)
                    attributes = self._apply_linguistic_rules(text, start, end)
                    
                    entities.append({
                        "start": start,
                        "end": end,
                        "span_text": text[start:end],
                        "concept_id": concept_id,
                        "confidence": 0.85,
                        "method": "abbreviation",
                        "section": section,
                        "attributes": attributes
                    })
        
        # 4. Diccionarios específicos por sección
        for section_name, section_dict in self.section_dicts.items():
            for term, (term_type, concept_id) in section_dict.items():
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    start, end = match.span()
                    detected_section = self._detect_section(text, start)
                    
                    # Solo aplicar si estamos en la sección correcta o es general
                    if detected_section == section_name or section_name == "general":
                        if not any(e["start"] == start and e["end"] == end for e in entities):
                            attributes = self._apply_linguistic_rules(text, start, end)
                            attributes["term_type"] = term_type
                            
                            entities.append({
                                "start": start,
                                "end": end,
                                "span_text": text[start:end],
                                "concept_id": concept_id,
                                "confidence": 0.92,
                                "method": "section_specific",
                                "section": detected_section,
                                "attributes": attributes
                            })
        
        # 5. Post-procesamiento: resolver overlaps (implementación real)
        entities = self._resolve_overlaps_real(entities)
        
        return entities
    
    def _resolve_overlaps_real(self, entities: List[Dict]) -> List[Dict]:
        """
        Resuelve overlaps usando la lógica real de KIRIs:
        - Priorizar por confianza
        - Luego por longitud del término
        - Luego por especificidad del método
        """
        
        if not entities:
            return entities
        
        # Ordenar por posición
        entities.sort(key=lambda x: (x["start"], x["end"]))
        
        # Ranking de métodos por prioridad (basado en código real)
        method_priority = {
            "case_sensitive": 4,
            "section_specific": 3,
            "main_dict": 2,
            "abbreviation": 1
        }
        
        resolved = []
        for entity in entities:
            # Verificar overlap con entidades ya resueltas
            has_overlap = False
            
            for i, resolved_entity in enumerate(resolved):
                if (entity["start"] < resolved_entity["end"] and 
                    entity["end"] > resolved_entity["start"]):
                    
                    # Hay overlap - decidir cuál mantener
                    keep_new = False
                    
                    # 1. Comparar confianza
                    if entity["confidence"] > resolved_entity["confidence"]:
                        keep_new = True
                    elif entity["confidence"] == resolved_entity["confidence"]:
                        # 2. Comparar longitud del término
                        if len(entity["span_text"]) > len(resolved_entity["span_text"]):
                            keep_new = True
                        elif len(entity["span_text"]) == len(resolved_entity["span_text"]):
                            # 3. Comparar prioridad del método
                            entity_priority = method_priority.get(entity["method"], 0)
                            resolved_priority = method_priority.get(resolved_entity["method"], 0)
                            if entity_priority > resolved_priority:
                                keep_new = True
                    
                    if keep_new:
                        resolved[i] = entity
                    
                    has_overlap = True
                    break
            
            if not has_overlap:
                resolved.append(entity)
        
        return resolved
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice entidades usando la implementación real de KIRIs
        """
        print(f"[REAL KIRIs] Procesando {len(notes_df)} notas con estrategia real...")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            entities = self.extract_entities(text)
            
            for entity in entities:
                predictions.append({
                    'note_id': note_id,
                    'start': entity['start'],
                    'end': entity['end'],
                    'concept_id': entity['concept_id'],
                    'span_text': entity['span_text'],
                    'confidence': entity['confidence'],
                    'method': entity['method'],
                    'section': entity['section'],
                    'attributes': str(entity['attributes'])  # Convertir dict a string
                })
            
            if (idx + 1) % 1 == 0:
                print(f"[REAL KIRIs] Procesadas {idx + 1}/{len(notes_df)} notas")
        
        print(f"[REAL KIRIs] Completado: {len(predictions)} predicciones generadas")
        return pd.DataFrame(predictions)



02_snobert.py:
#!/usr/bin/env python3
"""
Implementación REAL de la estrategia SNOBERT (2do lugar)
Adaptada del código original del repositorio ganador
Usa BERT para NER + SapBERT para embeddings + clasificación
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    pipeline, TokenClassificationPipeline
)
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict

class RealSNOBERTStrategy:
    """
    Implementación real de SNOBERT basada en su código ganador:
    1. Primera etapa: BERT-based NER (BiomedBERT)
    2. Segunda etapa: SapBERT embeddings + clasificación
    """
    
    def __init__(self):
        print("[REAL SNOBERT] Inicializando estrategia real SNOBERT...")
        
        # Configurar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[REAL SNOBERT] Usando dispositivo: {self.device}")
        
        # Inicializar modelos reales
        self._load_ner_model()
        self._load_embedding_model()
        self._build_snomed_index()
        self._setup_static_dictionary()
        
    def _load_ner_model(self):
        """Carga modelo BERT real para NER (primera etapa)"""
        
        try:
            # Usar BioBERT o similar modelo médico
            model_name = "dmis-lab/biobert-v1.1"
            print(f"[REAL SNOBERT] Cargando modelo NER: {model_name}")
            
            # Tokenizer
            self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Modelo personalizado para NER (basado en el código real)
            self.ner_model = self._create_ner_model(model_name)
            
            print("[REAL SNOBERT] Modelo NER cargado exitosamente")
            
        except Exception as e:
            print(f"[REAL SNOBERT] Error cargando NER model: {e}")
            print("[REAL SNOBERT] Usando NER basado en reglas como fallback")
            self.ner_model = None
            self.ner_tokenizer = None
    
    def _create_ner_model(self, model_name: str):
        """Crea modelo NER personalizado (basado en código real SNOBERT)"""
        
        class CustomNERModel(nn.Module):
            def __init__(self, model_name, num_labels=3):  # B-ENT, I-ENT, O
                super().__init__()
                self.config = AutoConfig.from_pretrained(model_name)
                self.bert = AutoModel.from_pretrained(model_name)
                self.dropout = nn.Dropout(0.3)
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)
                
            def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                sequence_output = outputs[0]
                sequence_output = self.dropout(sequence_output)
                logits = self.classifier(sequence_output)
                
                return logits
        
        model = CustomNERModel(model_name)
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_embedding_model(self):
        """Carga SapBERT para embeddings (segunda etapa)"""
        
        try:
            # SapBERT es el modelo real usado por SNOBERT
            model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
            print(f"[REAL SNOBERT] Cargando SapBERT: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            
            print("[REAL SNOBERT] SapBERT cargado exitosamente")
            
        except Exception as e:
            print(f"[REAL SNOBERT] Error cargando SapBERT: {e}")
            print("[REAL SNOBERT] Usando modelo alternativo...")
            
            # Fallback a modelo más ligero
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.embedding_model = None
    
    def _build_snomed_index(self):
        """Construye índice Faiss con conceptos SNOMED reales"""
        
        # Conceptos SNOMED específicos para ictus (basado en código real)
        self.snomed_concepts = {
            # Stroke and cerebrovascular
            "230690007": "Stroke, cerebrovascular accident",
            "432102000": "Ischemic stroke",  
            "274100004": "Hemorrhagic stroke",
            "266257000": "Transient ischemic attack",
            "71444005": "Cerebral thrombosis",
            "414086009": "Cerebral embolism",
            
            # Motor symptoms
            "50582007": "Hemiparesis",
            "13791008": "Weakness",
            "26544005": "Motor weakness", 
            "280816001": "Facial weakness",
            "44077006": "Sensory loss",
            "20262006": "Ataxia",
            
            # Speech and language
            "87486003": "Aphasia",
            "8011004": "Dysarthria",
            "29164008": "Speech disorder",
            
            # Vascular pathology
            "50960005": "Hemorrhage",
            "21454007": "Subarachnoid hemorrhage",
            "71186008": "Intraventricular hemorrhage",
            "55342001": "Infarction",
            "52674009": "Ischemia",
            "26036001": "Occlusion",
            "415582006": "Stenosis",
            "432101006": "Aneurysm",
            
            # Anatomy
            "69930009": "Middle cerebral artery",
            "86547008": "Internal carotid artery", 
            "79371005": "Anterior cerebral artery",
            "70382005": "Posterior cerebral artery",
            "67889009": "Basilar artery",
            "32603002": "Basal ganglia",
            "42695009": "Thalamus",
            "113305005": "Cerebellum",
            "15926001": "Brainstem",
            "42696002": "Internal capsule",
            
            # Procedures
            "433112001": "Thrombectomy",
            "373110003": "Thrombolysis",
            "77343006": "Angiography",
            "449894001": "Recanalization",
            
            # Imaging
            "77477000": "Computed tomography",
            "113091000": "Magnetic resonance imaging",
            "419775003": "CT angiography",
            
            # Clinical scales
            "450893003": "NIHSS, ASPECTS, TICI",
            "273302005": "Modified Rankin Scale",
            "386554004": "Glasgow Coma Scale",
            
            # Other findings
            "25064002": "Headache",
            "422587007": "Nausea", 
            "422400008": "Vomiting",
            "40917007": "Confusion",
            "419045004": "Loss of consciousness",
            "24982008": "Diplopia",
            "18060000": "Visual field defect"
        }
        
        if self.embedding_model is not None:
            try:
                print("[REAL SNOBERT] Construyendo índice Faiss con SapBERT...")
                
                # Generar embeddings para conceptos SNOMED
                concept_texts = list(self.snomed_concepts.values())
                embeddings = self.embedding_model.encode(
                    concept_texts,
                    show_progress_bar=True,
                    batch_size=32
                )
                
                # Crear índice Faiss
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Mapear índices a concept IDs
                self.concept_ids = list(self.snomed_concepts.keys())
                
                print(f"[REAL SNOBERT] Índice Faiss: {len(concept_texts)} conceptos, dim {dimension}")
                
            except Exception as e:
                print(f"[REAL SNOBERT] Error construyendo índice: {e}")
                self.faiss_index = None
        else:
            self.faiss_index = None
    
    def _setup_static_dictionary(self):
        """Diccionario estático para casos específicos (del código real)"""
        
        # Basado en static_dict.py del código real SNOBERT
        self.static_dict = {
            # Casos específicos que el modelo puede fallar
            "left MCA": ("69930009", "Left middle cerebral artery"),
            "right MCA": ("69930009", "Right middle cerebral artery"),
            "M1 segment": ("69930009", "MCA M1 segment"),
            "M2 segment": ("69930009", "MCA M2 segment"),
            "TICI 0": ("450893003", "TICI 0 - no perfusion"),
            "TICI 1": ("450893003", "TICI 1 - minimal perfusion"),
            "TICI 2A": ("450893003", "TICI 2A - partial perfusion"),
            "TICI 2B": ("450893003", "TICI 2B - partial perfusion"),
            "TICI 3": ("450893003", "TICI 3 - complete perfusion"),
            "ASPECTS 0": ("450893003", "ASPECTS score 0"),
            "ASPECTS 1": ("450893003", "ASPECTS score 1"),
            "ASPECTS 2": ("450893003", "ASPECTS score 2"),
            "ASPECTS 3": ("450893003", "ASPECTS score 3"),
            "ASPECTS 4": ("450893003", "ASPECTS score 4"),
            "ASPECTS 5": ("450893003", "ASPECTS score 5"),
            "ASPECTS 6": ("450893003", "ASPECTS score 6"),
            "ASPECTS 7": ("450893003", "ASPECTS score 7"),
            "ASPECTS 8": ("450893003", "ASPECTS score 8"),
            "ASPECTS 9": ("450893003", "ASPECTS score 9"),
            "ASPECTS 10": ("450893003", "ASPECTS score 10"),
            "NIHSS 0": ("450893003", "NIHSS score 0"),
            "mRS 0": ("273302005", "mRS score 0"),
            "mRS 1": ("273302005", "mRS score 1"),
            "mRS 2": ("273302005", "mRS score 2"),
            "mRS 3": ("273302005", "mRS score 3"),
            "mRS 4": ("273302005", "mRS score 4"),
            "mRS 5": ("273302005", "mRS score 5"),
            "mRS 6": ("273302005", "mRS score 6")
        }
        
        print(f"[REAL SNOBERT] Diccionario estático: {len(self.static_dict)} entradas")
    
    def _first_stage_ner(self, text: str) -> List[Dict]:
        """Primera etapa: NER con BERT (implementación real)"""
        
        if self.ner_model is None or self.ner_tokenizer is None:
            # Fallback a NER basado en reglas
            return self._rule_based_ner(text)
        
        try:
            # Tokenizar texto
            inputs = self.ner_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predicción
            with torch.no_grad():
                logits = self.ner_model(**inputs)
            
            # Convertir logits a predicciones
            predictions = torch.argmax(logits, dim=-1)
            
            # Decodificar entidades
            entities = self._decode_ner_predictions(text, inputs, predictions)
            
            return entities
            
        except Exception as e:
            print(f"[REAL SNOBERT] Error en NER BERT: {e}")
            return self._rule_based_ner(text)
    
    def _rule_based_ner(self, text: str) -> List[Dict]:
        """NER basado en reglas como fallback"""
        
        # Patrones para entidades médicas
        medical_patterns = [
            # Stroke terms
            r'\b(stroke|CVA|cerebrovascular accident|brain attack)\b',
            r'\b(ischemic|hemorrhagic)\s+stroke\b',
            r'\b(acute|subacute|chronic)\s+stroke\b',
            
            # Motor symptoms
            r'\b(hemiparesis|hemiplegia|weakness|paralysis)\b',
            r'\b(left|right)\s+(sided\s+)?(weakness|hemiparesis)\b',
            r'\b(facial\s+)?(weakness|droop)\b',
            
            # Speech symptoms
            r'\b(aphasia|dysphasia|dysarthria)\b',
            r'\b(speech\s+)?(difficulty|impairment|disorder)\b',
            
            # Vascular pathology
            r'\b(hemorrhage|bleeding|haemorrhage)\b',
            r'\b(infarct|infarction|ischemia|ischaemia)\b',
            r'\b(occlusion|stenosis|aneurysm|thrombosis|embolism)\b',
            
            # Anatomy
            r'\b(MCA|ACA|PCA|ICA|basilar|vertebral)\s*(artery|segment)?\b',
            r'\b(M1|M2|M3|A1|A2|P1|P2)\s*segment\b',
            r'\b(basal ganglia|thalamus|cerebellum|brainstem)\b',
            r'\b(frontal|parietal|temporal|occipital)\s+lobe\b',
            
            # Procedures
            r'\b(thrombectomy|thrombolysis|angiography|recanalization)\b',
            r'\b(mechanical\s+)?thrombectomy\b',
            r'\b(tPA|rtPA|tissue plasminogen activator)\b',
            
            # Imaging
            r'\b(CT|MRI|CTA|MRA|DWI|PWI|FLAIR)\b',
            r'\b(computed tomography|magnetic resonance)\b',
            
            # Clinical scales
            r'\b(NIHSS|ASPECTS|TICI|mRS|GCS)\s*\d*\b',
            r'\b(National Institutes of Health Stroke Scale)\b',
            
            # Other symptoms
            r'\b(headache|nausea|vomiting|confusion|dizziness)\b',
            r'\b(altered mental status|loss of consciousness)\b',
            r'\b(diplopia|visual field defect|hemianopia)\b'
        ]
        
        entities = []
        
        for pattern in medical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.span()
                
                entities.append({
                    "start": start,
                    "end": end,
                    "span_text": text[start:end],
                    "confidence": 0.8,
                    "method": "rule_based_ner"
                })
        
        return entities
    
    def _decode_ner_predictions(self, text: str, inputs: Dict, predictions: torch.Tensor) -> List[Dict]:
        """Decodifica predicciones NER a entidades"""
        
        # Esta es una implementación simplificada
        # En el código real de SNOBERT sería más compleja
        
        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        pred_labels = predictions[0].cpu().numpy()
        
        # Labels: 0=O, 1=B-ENT, 2=I-ENT
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label == 1:  # B-ENT
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    "tokens": [token],
                    "start_idx": i,
                    "confidence": 0.85
                }
            elif label == 2 and current_entity is not None:  # I-ENT
                current_entity["tokens"].append(token)
            else:  # O
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity is not None:
            entities.append(current_entity)
        
        # Convertir tokens a posiciones en texto original
        final_entities = []
        for entity in entities:
            # Simplificación: usar regex para encontrar la entidad en el texto
            entity_text = " ".join(entity["tokens"]).replace(" ##", "")
            
            for match in re.finditer(re.escape(entity_text), text, re.IGNORECASE):
                final_entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "span_text": match.group(),
                    "confidence": entity["confidence"],
                    "method": "bert_ner"
                })
                break
        
        return final_entities
    
    def _second_stage_classification(self, entities: List[Dict]) -> List[Dict]:
        """Segunda etapa: Clasificación con SapBERT (implementación real)"""
        
        if self.faiss_index is None or self.embedding_model is None:
            # Fallback a clasificación simple
            return self._simple_classification(entities)
        
        classified_entities = []
        
        for entity in entities:
            span_text = entity["span_text"]
            
            # 1. Verificar diccionario estático primero
            static_match = self._check_static_dictionary(span_text)
            if static_match:
                entity.update({
                    "concept_id": static_match[0],
                    "concept_description": static_match[1],
                    "classification_method": "static_dict",
                    "confidence": min(entity["confidence"] * 1.1, 1.0)
                })
                classified_entities.append(entity)
                continue
            
            # 2. Usar SapBERT + Faiss para clasificación
            try:
                # Generar embedding para la entidad
                entity_embedding = self.embedding_model.encode([span_text])
                
                # Buscar en índice Faiss
                distances, indices = self.faiss_index.search(
                    entity_embedding.astype('float32'), k=1
                )
                
                if len(indices[0]) > 0:
                    best_idx = indices[0][0]
                    distance = distances[0][0]
                    
                    # Convertir distancia a confianza
                    similarity_confidence = max(0.0, 1.0 - (distance / 2.0))
                    
                    concept_id = self.concept_ids[best_idx]
                    concept_description = self.snomed_concepts[concept_id]
                    
                    entity.update({
                        "concept_id": concept_id,
                        "concept_description": concept_description,
                        "classification_method": "sapbert_faiss",
                        "confidence": entity["confidence"] * similarity_confidence,
                        "similarity_score": similarity_confidence
                    })
                    
                    classified_entities.append(entity)
                
            except Exception as e:
                print(f"[REAL SNOBERT] Error clasificando '{span_text}': {e}")
                # Fallback
                entity.update({
                    "concept_id": "404684003",  # Clinical finding
                    "concept_description": "Clinical finding",
                    "classification_method": "fallback",
                    "confidence": entity["confidence"] * 0.5
                })
                classified_entities.append(entity)
        
        return classified_entities
    
    def _check_static_dictionary(self, span_text: str) -> Tuple[str, str]:
        """Verifica diccionario estático"""
        
        span_lower = span_text.lower().strip()
        
        for pattern, (concept_id, description) in self.static_dict.items():
            if pattern.lower() in span_lower or span_lower in pattern.lower():
                return (concept_id, description)
        
        return None
    
    def _simple_classification(self, entities: List[Dict]) -> List[Dict]:
        """Clasificación simple como fallback"""
        
        # Mapeo simple basado en palabras clave
        keyword_mapping = {
            "stroke": "230690007",
            "hemorrhage": "50960005", 
            "infarct": "55342001",
            "weakness": "13791008",
            "aphasia": "87486003",
            "mca": "69930009",
            "ica": "86547008",
            "ct": "77477000",
            "mri": "113091000"
        }
        
        for entity in entities:
            span_lower = entity["span_text"].lower()
            
            concept_id = "404684003"  # Default: Clinical finding
            for keyword, cid in keyword_mapping.items():
                if keyword in span_lower:
                    concept_id = cid
                    break
            
            entity.update({
                "concept_id": concept_id,
                "classification_method": "simple_keyword",
                "confidence": entity["confidence"] * 0.7
            })
        
        return entities
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Pipeline completo de SNOBERT real:
        1. Primera etapa: NER con BERT
        2. Segunda etapa: Clasificación con SapBERT
        """
        
        # Primera etapa: NER
        entities = self._first_stage_ner(text)
        
        if not entities:
            return []
        
        # Segunda etapa: Clasificación
        classified_entities = self._second_stage_classification(entities)
        
        return classified_entities
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice entidades usando la implementación real de SNOBERT
        """
        print(f"[REAL SNOBERT] Procesando {len(notes_df)} notas con estrategia real...")
        
        predictions = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
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
                    'classification_method': entity.get('classification_method', 'unknown'),
                    'concept_description': entity.get('concept_description', '')
                })
            
            if (idx + 1) % 1 == 0:
                print(f"[REAL SNOBERT] Procesadas {idx + 1}/{len(notes_df)} notas")
        
        print(f"[REAL SNOBERT] Completado: {len(predictions)} predicciones generadas")
        return pd.DataFrame(predictions)



04_rag_gpt.py:
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







Básicamente debo mejorar ahora el rendimiento para el 04_rag_gpt:
Me han dado este feedback:
Básicamente el documento conceptos_con_narrativas.csv son 40k lineas y tengo 14k chunks, esto son 3 lineas por chunk.
Hay que probar 1 chunk por linea.
Falta reorganizar ficheros-> build rag index se deberia llamar ontology_preprocessor.py y ponerlo en un directorio /04_strategy.
Además la carpeta assets debería ir dentro de 04_strategy.
Focus en mejorar el rendimiento del GPT (no es muy importante tema API calls).
Faltará volver a ejecutar el builder, ya aprovechar y hacer lo de los chunks.


Aquí tienes el builder:
#!/usr/bin/env python3
"""
Offline Index Builder for RAG Strategy

This script pre-computes the embeddings and Faiss index for the RAG+GPT4o strategy.
It separates the computationally intensive index creation from the strategy initialization,
following the Single Responsibility Principle (SRP).

Responsibilities:
  - Load ontology data
  - Generate embeddings using SentenceTransformer (can leverage GPU if available)
  - Build Faiss index
  - Save artifacts to disk for fast loading at runtime

Benefits:
  🚀 Performance: Index built once offline, loaded instantly at runtime
  🔄 Consistency: Same index used across all strategy instantiations
  🧩 Modularity: Index creation logic separated from RAG strategy
  📦 Open/Closed Principle: Strategy class doesn't need modification if index generation changes
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
ASSETS_DIR = os.path.join(SCRIPT_DIR, 'assets')

# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

# Output paths
INDEX_PATH = os.path.join(ASSETS_DIR, 'ontology.index')
CONCEPTS_PATH = os.path.join(ASSETS_DIR, 'ontology_concepts.pkl')
NARRATIVES_PATH = os.path.join(ASSETS_DIR, 'ontology_narratives.pkl')
METADATA_PATH = os.path.join(ASSETS_DIR, 'ontology_metadata.pkl')


def load_ontology_csv():
    """
    Load the ontology CSV file containing concepts and their narrative descriptions.
    
    Returns:
        pd.DataFrame: Ontology data with 'concepto' and 'narrativa' columns
    """
    print("\n" + "="*80)
    print("STEP 1: Loading Ontology Data")
    print("="*80)
    
    # Try multiple paths
    possible_paths = [
        os.path.join(PROJECT_ROOT, 'conceptos_con_narrativas.csv'),
        os.path.join('..', 'conceptos_con_narrativas.csv'),
        'conceptos_con_narrativas.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Loading from: {path}")
            df = pd.read_csv(path)
            print(f"[SUCCESS] Loaded {len(df)} concepts")
            return df
    
    raise FileNotFoundError(
        f"Could not find 'conceptos_con_narrativas.csv' in any of: {possible_paths}"
    )


def generate_embeddings(narratives, model_name='all-MiniLM-L6-v2', batch_size=32, use_gpu=True):
    """
    Generate embeddings for all narrative descriptions using SentenceTransformer.
    
    Args:
        narratives (list): List of narrative text strings
        model_name (str): Name of the SentenceTransformer model
        batch_size (int): Batch size for encoding (larger for GPU)
        use_gpu (bool): Whether to try using GPU if available
        
    Returns:
        np.ndarray: Embeddings matrix of shape (n_narratives, embedding_dim)
    """
    print("\n" + "="*80)
    print("STEP 2: Generating Embeddings")
    print("="*80)
    
    print(f"[INFO] Loading SentenceTransformer model: {model_name}")
    
    # Initialize model (will automatically use GPU if available via CUDA)
    model = SentenceTransformer(model_name)
    
    # Check device
    device = model.device
    print(f"[INFO] Using device: {device}")
    
    if 'cuda' in str(device):
        print("[INFO] 🚀 GPU detected! Encoding will be faster.")
        # Increase batch size for GPU
        batch_size = 128
    else:
        print("[INFO] Using CPU. This may take several minutes...")
    
    print(f"[INFO] Encoding {len(narratives)} narratives with batch_size={batch_size}")
    print("[INFO] This is a ONE-TIME operation. Subsequent runs will load pre-built index.")
    
    # Generate embeddings
    embeddings = model.encode(
        narratives,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # L2 distance in Faiss handles this
    )
    
    print(f"[SUCCESS] Generated embeddings with shape: {embeddings.shape}")
    print(f"[INFO] Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, model


def build_faiss_index(embeddings):
    """
    Build a Faiss index from the embeddings.
    
    Args:
        embeddings (np.ndarray): Embeddings matrix
        
    Returns:
        faiss.Index: Built Faiss index
    """
    print("\n" + "="*80)
    print("STEP 3: Building Faiss Index")
    print("="*80)
    
    dimension = embeddings.shape[1]
    n_concepts = embeddings.shape[0]
    
    print(f"[INFO] Creating IndexFlatL2 with dimension={dimension}")
    
    # Create index (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    print(f"[INFO] Adding {n_concepts} vectors to index...")
    index.add(embeddings.astype('float32'))
    
    print(f"[SUCCESS] Faiss index built with {index.ntotal} vectors")
    
    return index


def save_artifacts(index, concepts, narratives, embedding_dim):
    """
    Save all artifacts to disk for fast loading at runtime.
    
    Args:
        index (faiss.Index): Built Faiss index
        concepts (list): List of concept IDs in order
        narratives (list): List of narratives in order
        embedding_dim (int): Dimension of embeddings
    """
    print("\n" + "="*80)
    print("STEP 4: Saving Artifacts")
    print("="*80)
    
    # Save Faiss index
    print(f"[INFO] Saving Faiss index to: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    print(f"[SUCCESS] Index saved ({os.path.getsize(INDEX_PATH) / 1024 / 1024:.2f} MB)")
    
    # Save concepts list (for retrieval mapping)
    print(f"[INFO] Saving concepts list to: {CONCEPTS_PATH}")
    with open(CONCEPTS_PATH, 'wb') as f:
        pickle.dump(concepts, f)
    print(f"[SUCCESS] Concepts saved ({len(concepts)} items)")
    
    # Save narratives list (for context generation)
    print(f"[INFO] Saving narratives list to: {NARRATIVES_PATH}")
    with open(NARRATIVES_PATH, 'wb') as f:
        pickle.dump(narratives, f)
    print(f"[SUCCESS] Narratives saved ({len(narratives)} items)")
    
    # Save metadata
    metadata = {
        'n_concepts': len(concepts),
        'embedding_dim': embedding_dim,
        'model_name': 'all-MiniLM-L6-v2',
        'created_at': datetime.now().isoformat(),
        'index_type': 'IndexFlatL2'
    }
    
    print(f"[INFO] Saving metadata to: {METADATA_PATH}")
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[SUCCESS] Metadata saved")
    
    print("\n" + "="*80)
    print("✅ ALL ARTIFACTS SAVED SUCCESSFULLY")
    print("="*80)
    print(f"\nArtifacts location: {ASSETS_DIR}")
    print(f"  - {os.path.basename(INDEX_PATH)}")
    print(f"  - {os.path.basename(CONCEPTS_PATH)}")
    print(f"  - {os.path.basename(NARRATIVES_PATH)}")
    print(f"  - {os.path.basename(METADATA_PATH)}")


def main():
    """
    Main execution flow for building the RAG index offline.
    """
    print("\n" + "="*80)
    print("RAG INDEX BUILDER - Offline Pre-computation")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load ontology data (45,000+ concepts)")
    print("  2. Generate embeddings using SentenceTransformer")
    print("  3. Build Faiss index for fast similarity search")
    print("  4. Save all artifacts to disk")
    print("\n⚠️  This is a ONE-TIME operation (unless ontology data changes)")
    print("="*80)
    
    import time
    start_time = time.time()
    
    try:
        # Step 1: Load ontology
        df_ontology = load_ontology_csv()
        
        # Extract concepts and narratives
        concepts = df_ontology['concepto'].tolist()
        narratives = df_ontology['narrativa'].tolist()
        
        # Step 2: Generate embeddings
        embeddings, model = generate_embeddings(narratives)
        
        # Step 3: Build Faiss index
        faiss_index = build_faiss_index(embeddings)
        
        # Step 4: Save artifacts
        save_artifacts(faiss_index, concepts, narratives, embeddings.shape[1])
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 INDEX BUILD COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"📊 Indexed: {len(concepts)} concepts")
        print(f"📏 Embedding dimension: {embeddings.shape[1]}")
        print("\n✅ The RAG strategy will now load instantly!")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR BUILDING INDEX")
        print("="*80)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




Y el main.py:
import sys
import os
import pandas as pd
import argparse
import time
import json
import re
import importlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
STRATEGIES_DIR = os.path.join(SCRIPT_DIR, 'strategies')
EVALUATION_DIR = os.path.join(SCRIPT_DIR, 'evaluation')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Add to path
for path in [SCRIPT_DIR, STRATEGIES_DIR, EVALUATION_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import metrics calculator
from evaluation.metrics_calculator import MetricsCalculator


# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

STRATEGY_CONFIG = {
    1: {
        'id': 1,
        'name': '01_KIRIs',
        'display_name': 'KIRIs REAL (1st Place)',
        'description': 'Hybrid dictionary + linguistic rules',
        'module': '01_kiris',
        'class_name': 'RealKIRIsStrategy',
        'active': True
    },
    2: {
        'id': 2,
        'name': '02_SNOBERT',
        'display_name': 'SNOBERT REAL (2nd Place)',
        'description': 'BERT + SapBERT + embeddings',
        'module': '02_snobert',
        'class_name': 'RealSNOBERTStrategy',
        'active': True
    },
    3: {
        'id': 3,
        'name': '03_Ollama',
        'display_name': 'MITEL REAL (3rd Place)',
        'description': 'Mistral 7B via Ollama + RAG',
        'module': '03_ollama',
        'class_name': 'RealMITELOllamaStrategy',
        'active': False  # Currently disabled
    },
    4: {
        'id': 4,
        'name': '04_RAG_GPT',
        'display_name': 'RAG + GPT-4o',
        'description': 'Custom RAG system with GPT-4o',
        'module': '04_rag_gpt',
        'class_name': 'RAGWithGPT4oStrategy',
        'active': True
    }
}

# Active strategies (IDs)
ACTIVE_STRATEGIES = [sid for sid, config in STRATEGY_CONFIG.items() if config['active']]


# ============================================================================
# STRATEGY LOADING
# ============================================================================

def load_strategy(strategy_id: int) -> Tuple[Optional[object], Dict]:
    """
    Dynamically load and instantiate a strategy.
    
    Args:
        strategy_id: ID of the strategy to load
        
    Returns:
        Tuple of (strategy_instance, config_dict)
    """
    
    if strategy_id not in STRATEGY_CONFIG:
        print(f"[ERROR] Invalid strategy ID: {strategy_id}")
        return None, {}
    
    config = STRATEGY_CONFIG[strategy_id]
    
    if not config['active']:
        print(f"[WARNING] Strategy {strategy_id} ({config['display_name']}) is currently disabled")
        return None, config
    
    print(f"\n{'='*80}")
    print(f"INITIALIZING: {config['display_name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    try:
        # Dynamic import using importlib.util for modules with numeric names
        import importlib.util
        
        module_name = config['module']
        module_path = os.path.join(STRATEGIES_DIR, f"{module_name}.py")
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Strategy module not found: {module_path}")
        
        # Load module from file path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get strategy class
        strategy_class = getattr(module, config['class_name'])
        strategy_instance = strategy_class()
        
        print(f"[SUCCESS] {config['display_name']} initialized successfully")
        return strategy_instance, config
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize {config['display_name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, config

# ejecuta 1 estrategia
def run_strategy(strategy_id: int, 
                notes_df: pd.DataFrame, 
                annotations_df: pd.DataFrame,
                metrics_calc: MetricsCalculator) -> Dict:
    # Load strategy
    strategy, config = load_strategy(strategy_id)
    
    if strategy is None:
        return {
            'config': config,
            'metrics': None,
            'predictions': None,
            'execution_time': 0.0,
            'error': 'Failed to load strategy'
        }
    
    strategy_name = config['name']
    
    print(f"\n{'='*100}")
    print(f"EXECUTING: {config['display_name']}")
    print(f"{'='*100}")
    
    start_time = time.time()
    
    try:
        # Execute prediction
        print(f"[INFO] Running predictions on {len(notes_df)} notes...")
        predictions = strategy.predict(notes_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"[INFO] Generated {len(predictions)} predictions in {execution_time:.2f} seconds")
        
        # Calculate metrics
        metrics = metrics_calc.calculate_metrics(predictions, annotations_df, strategy_name)
        
        # Print single report
        report = metrics_calc.format_single_report(metrics, execution_time, config['display_name'])
        print(report)
        
        return {
            'config': config,
            'metrics': metrics,
            'predictions': predictions,
            'execution_time': execution_time,
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"[ERROR] Failed to execute {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'config': config,
            'metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'predictions': 0,
                'matches': 0,
                'partial_matches': 0,
                'ground_truth': len(annotations_df),
                'coverage': 0.0
            },
            'predictions': None,
            'execution_time': execution_time,
            'error': str(e)
        }


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def create_execution_directory() -> Tuple[str, str]:
    # Find next execution number
    exec_num = 1
    dir_pattern = re.compile(r"^(\d+)_execution_.*$")
    existing_nums = []
    
    for dirname in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, dirname)
        if os.path.isdir(full_path):
            match = dir_pattern.match(dirname)
            if match:
                existing_nums.append(int(match.group(1)))
    
    if existing_nums:
        exec_num = max(existing_nums) + 1
    
    # Create directory name with timestamp
    timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
    exec_num_str = f"{exec_num:02d}"
    dir_name = f"{exec_num_str}_execution_{timestamp_str}"
    
    # Create directory
    full_path = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(full_path, exist_ok=True)
    
    return dir_name, full_path


def save_results(results_dict: Dict, 
                notes_df: pd.DataFrame, 
                annotations_df: pd.DataFrame,
                execution_dir: str) -> None:
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Prepare summary data
    summary = {
        'timestamp': datetime.now().strftime("%m_%d_%Y_%H_%M"),
        'execution_folder': os.path.basename(execution_dir),
        'strategies_evaluated': [],
        'dataset_info': {
            'notes_count': len(notes_df),
            'annotations_count': len(annotations_df)
        },
        'results': {},
        'ranking': []
    }
    
    # Process each strategy result
    for strategy_id, result in results_dict.items():
        config = result['config']
        metrics = result['metrics']
        exec_time = result['execution_time']
        predictions = result['predictions']
        
        strategy_name = config['name']
        summary['strategies_evaluated'].append(strategy_name)
        
        # Add to summary
        summary['results'][strategy_name] = {
            'display_name': config['display_name'],
            'description': config['description'],
            'metrics': metrics,
            'execution_time': exec_time,
            'error': result.get('error')
        }
        
        # Save predictions CSV
        if predictions is not None and len(predictions) > 0:
            pred_filename = os.path.join(execution_dir, f"predictions_{strategy_name}.csv")
            predictions.to_csv(pred_filename, index=False, encoding="utf-8")
            print(f"[SAVED] Predictions: {pred_filename}")
    
    # Create ranking
    ranked = sorted(
        [(name, data['metrics']['f1']) for name, data in summary['results'].items()],
        key=lambda x: x[1],
        reverse=True
    )
    summary['ranking'] = [{'strategy': name, 'f1_score': f1} for name, f1 in ranked]
    
    # Save JSON report
    report_filename = os.path.join(execution_dir, "evaluation_report.json")
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[SAVED] Summary report: {report_filename}")
    print(f"\n[INFO] All results saved to: {execution_dir}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SNOMED-CT entity linking strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run all active strategies (1, 2, 4) - NO SAVE
  python main.py 1                            # Run only strategy 1 (KIRIs) - NO SAVE
  python main.py 2                            # Run only strategy 2 (SNOBERT) - NO SAVE
  python main.py 4                            # Run only strategy 4 (RAG GPT-4o) - NO SAVE
  python main.py 1 4                          # Run strategies 1 and 4, then compare - NO SAVE
  python main.py 1 -r                         # Run strategy 1 and SAVE results
  python main.py -s 1 4 -r                    # Run strategies 1 and 4, and SAVE results
  python main.py -r                           # Run all active strategies and SAVE results

Active Strategies:
  1: KIRIs REAL - Hybrid dictionary + linguistic rules
  2: SNOBERT REAL - BERT + SapBERT + embeddings
  4: RAG + GPT-4o - Custom RAG system with GPT-4o
  
Note: Strategy 3 (Ollama) is currently disabled.
        """
    )
    
    parser.add_argument(
        'strategy_ids',
        nargs='*',
        type=int,
        help='Strategy ID(s) to run (positional). If not specified, runs all active strategies.'
    )
    
    parser.add_argument(
        '-s', '--strategy-id',
        nargs='+',
        type=int,
        choices=[1, 2, 3, 4],
        help='Strategy ID(s) to run (alternative flag). If not specified, runs all active strategies.'
    )
    
    parser.add_argument(
        '-r', '--save-results',
        action='store_true',
        help='Save results to disk. By default, results are only printed to console.'
    )
    
    args = parser.parse_args()
    
    # Determine which strategies to run
    # Priority: positional args > -s flag > default (all active)
    if args.strategy_ids:
        # Positional arguments provided
        strategies_to_run = args.strategy_ids
        
        # Validate strategy IDs
        invalid_ids = [sid for sid in strategies_to_run if sid not in STRATEGY_CONFIG]
        if invalid_ids:
            print(f"[ERROR] Invalid strategy ID(s): {invalid_ids}")
            print(f"Valid strategy IDs are: {list(STRATEGY_CONFIG.keys())}")
            return
        
        # Filter out inactive strategies
        strategies_to_run = [sid for sid in strategies_to_run 
                            if STRATEGY_CONFIG[sid]['active']]
        
        if not strategies_to_run:
            print("[ERROR] All specified strategies are inactive.")
            return
    
    elif args.strategy_id:
        # Flag-based arguments provided
        strategies_to_run = args.strategy_id
        # Filter out inactive strategies
        strategies_to_run = [sid for sid in strategies_to_run 
                            if STRATEGY_CONFIG[sid]['active']]
        
        if not strategies_to_run:
            print("[ERROR] All specified strategies are inactive.")
            return
    else:
        # Default: run all active strategies
        strategies_to_run = ACTIVE_STRATEGIES
    
    # Print header
    print("\n" + "="*100)
    print("SNOMED-CT ENTITY LINKING - STRATEGY EVALUATION")
    print("="*100)
    print(f"\nStrategies to evaluate: {', '.join([STRATEGY_CONFIG[sid]['display_name'] for sid in strategies_to_run])}")
    print("="*100)
    
    # Load datasets
    print("\n[LOADING] Loading datasets...")
    try:
        notes_path = os.path.join(DATA_DIR, "mimic-iv_notes_training_set.csv")
        annotations_path = os.path.join(DATA_DIR, "train_annotations.csv")
        
        notes_df = pd.read_csv(notes_path)
        annotations_df = pd.read_csv(annotations_path)
        
        print(f"[SUCCESS] Loaded {len(notes_df)} notes")
        print(f"[SUCCESS] Loaded {len(annotations_df)} annotations")
        
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        return
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Run strategies
    results = {}
    
    for strategy_id in strategies_to_run:
        result = run_strategy(strategy_id, notes_df, annotations_df, metrics_calc)
        results[strategy_id] = result
    
    # Print comparison report if multiple strategies
    if len(results) > 1:
        comparison_data = {
            result['config']['name']: {
                'metrics': result['metrics'],
                'execution_time': result['execution_time']
            }
            for result in results.values()
            if result['metrics'] is not None
        }
        
        comparison_report = metrics_calc.format_comparison_report(comparison_data)
        print(comparison_report)
    
    # Save results only if -r flag is provided
    if args.save_results:
        # Create execution directory and save results
        dir_name, exec_path = create_execution_directory()
        save_results(results, notes_df, annotations_df, exec_path)
        print(f"\n[INFO] Results saved to: {exec_path}")
    else:
        print(f"\n[INFO] Results not saved. Use -r or --save-results flag to save results to disk.")
    
    # Final message
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETED")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
