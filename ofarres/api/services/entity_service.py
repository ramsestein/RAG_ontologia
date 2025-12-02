"""
Entity Service

Handles business logic for medical entity extraction.
Following Single Responsibility Principle.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models.schemas import EntityResponse, EntityType, SnomedDetailResponse


# SNOMED CT code to entity type mapping
SNOMED_TYPE_MAP: Dict[str, EntityType] = {
    # Disorders
    "38341003": EntityType.DISORDER,      # Hypertension
    "73211009": EntityType.DISORDER,      # Diabetes
    "230690007": EntityType.DISORDER,     # Stroke
    "21454007": EntityType.DISORDER,      # Subarachnoid hemorrhage
    "266257000": EntityType.DISORDER,     # TIA
    "50582007": EntityType.DISORDER,      # Hemiparesis
    "87486003": EntityType.DISORDER,      # Aphasia
    "8011004": EntityType.DISORDER,       # Dysarthria
    "26036001": EntityType.DISORDER,      # Occlusion
    "415582006": EntityType.DISORDER,     # Stenosis
    "432101006": EntityType.DISORDER,     # Aneurysm
    "52674009": EntityType.DISORDER,      # Ischemia
    "13791008": EntityType.DISORDER,      # Weakness
    "20262006": EntityType.DISORDER,      # Ataxia
    "55342001": EntityType.DISORDER,      # Infarct
    "230691006": EntityType.DISORDER,     # Penumbra
    "69449002": EntityType.DISORDER,      # Carotid stenosis
    "50960005": EntityType.DISORDER,      # Hemorrhage
    "25064002": EntityType.DISORDER,      # Headache
    "422587007": EntityType.DISORDER,     # Nausea
    "422400008": EntityType.DISORDER,     # Vomiting
    "49436004": EntityType.DISORDER,      # Atrial fibrillation
    
    # Anatomy
    "69930009": EntityType.ANATOMY,       # Middle cerebral artery
    "86547008": EntityType.ANATOMY,       # Internal carotid artery
    "67889009": EntityType.ANATOMY,       # Basilar artery
    
    # Procedures
    "77477000": EntityType.PROCEDURE,     # CT
    "113091000": EntityType.PROCEDURE,    # MRI
    "77343006": EntityType.PROCEDURE,     # Angiography
    "433112001": EntityType.PROCEDURE,    # Thrombectomy
    "449894001": EntityType.PROCEDURE,    # Recanalization
    
    # Medications
    "387467008": EntityType.MEDICATION,   # tPA
    
    # Observations/Scales
    "450893003": EntityType.OBSERVATION,  # NIHSS, ASPECTS, TICI
}

# SNOMED CT concept details
SNOMED_DETAILS: Dict[str, Dict[str, Any]] = {
    "38341003": {
        "preferred_term": "Hypertensive disorder",
        "description": "A disorder characterized by persistently high arterial blood pressure",
        "parents": ["Cardiovascular disease", "Disorder of cardiovascular system"]
    },
    "73211009": {
        "preferred_term": "Diabetes mellitus",
        "description": "A metabolic disorder characterized by high blood sugar levels",
        "parents": ["Metabolic disease", "Endocrine disorder"]
    },
    "230690007": {
        "preferred_term": "Cerebrovascular accident",
        "description": "Acute neurological deficit due to cerebral blood flow disruption",
        "parents": ["Cerebrovascular disease", "Neurological disorder"]
    },
    "433112001": {
        "preferred_term": "Mechanical thrombectomy",
        "description": "Endovascular procedure to remove blood clot from cerebral vessel",
        "parents": ["Endovascular procedure", "Neurosurgical procedure"]
    },
    "69930009": {
        "preferred_term": "Middle cerebral artery structure",
        "description": "Major cerebral artery supplying lateral brain surface",
        "parents": ["Cerebral artery", "Arterial structure"]
    },
    "77477000": {
        "preferred_term": "Computed tomography",
        "description": "Diagnostic imaging using X-ray computed tomography",
        "parents": ["Diagnostic imaging", "Radiological procedure"]
    },
}


class EntityService:
    """Service for managing medical entity extraction."""
    
    def __init__(self):
        self._ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        self._load_ground_truth()
    
    def _load_ground_truth(self) -> None:
        """Load ground truth annotations from JSON file."""
        gt_path = Path(__file__).parent.parent.parent / "backend" / "data" / "ground_truth.json"
        
        if gt_path.exists():
            with open(gt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self._ground_truth[item["note_id"]] = item["annotations"]
    
    def get_entities_for_note(self, note_id: str) -> List[EntityResponse]:
        """Get extracted entities for a specific note."""
        annotations = self._ground_truth.get(note_id, [])
        
        entities = []
        for idx, ann in enumerate(annotations):
            entity_type = SNOMED_TYPE_MAP.get(ann["concept_id"], EntityType.OBSERVATION)
            
            entities.append(EntityResponse(
                id=f"{note_id}-{idx}",
                text=ann["text"],
                type=entity_type,
                start=ann["start"],
                end=ann["end"],
                snomed_code=ann["concept_id"],
                confidence=0.95  # Ground truth has high confidence
            ))
        
        return entities
    
    def analyze_text(self, text: str) -> List[EntityResponse]:
        """
        Analyze text and extract medical entities.
        
        Currently uses pattern matching against known terms.
        In production, this would call the NER model.
        """
        entities = []
        text_lower = text.lower()
        
        # Simple keyword-based extraction for demo
        keywords = {
            "hypertension": ("38341003", EntityType.DISORDER),
            "diabetes": ("73211009", EntityType.DISORDER),
            "stroke": ("230690007", EntityType.DISORDER),
            "hemorrhage": ("50960005", EntityType.DISORDER),
            "thrombectomy": ("433112001", EntityType.PROCEDURE),
            "ct": ("77477000", EntityType.PROCEDURE),
            "mri": ("113091000", EntityType.PROCEDURE),
            "angiography": ("77343006", EntityType.PROCEDURE),
            "weakness": ("13791008", EntityType.DISORDER),
            "headache": ("25064002", EntityType.DISORDER),
            "atrial fibrillation": ("49436004", EntityType.DISORDER),
            "stenosis": ("415582006", EntityType.DISORDER),
            "occlusion": ("26036001", EntityType.DISORDER),
            "infarct": ("55342001", EntityType.DISORDER),
            "tpa": ("387467008", EntityType.MEDICATION),
            "nihss": ("450893003", EntityType.OBSERVATION),
        }
        
        entity_id = 0
        for keyword, (code, entity_type) in keywords.items():
            start = 0
            while True:
                pos = text_lower.find(keyword, start)
                if pos == -1:
                    break
                    
                entities.append(EntityResponse(
                    id=str(entity_id),
                    text=text[pos:pos + len(keyword)],
                    type=entity_type,
                    start=pos,
                    end=pos + len(keyword),
                    snomed_code=code,
                    confidence=0.85 + (0.1 * (entity_id % 2))  # Vary confidence slightly
                ))
                entity_id += 1
                start = pos + len(keyword)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities
    
    def get_snomed_details(self, code: str) -> Optional[SnomedDetailResponse]:
        """Get detailed information about a SNOMED CT concept."""
        details = SNOMED_DETAILS.get(code)
        
        if details:
            return SnomedDetailResponse(
                code=code,
                preferred_term=details["preferred_term"],
                description=details["description"],
                parents=details["parents"]
            )
        
        # Return generic response for unknown codes
        return SnomedDetailResponse(
            code=code,
            preferred_term=f"SNOMED Concept {code}",
            description="Clinical concept from SNOMED CT terminology",
            parents=["Clinical finding"]
        )
