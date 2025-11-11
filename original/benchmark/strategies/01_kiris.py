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

