"""
Pydantic Schemas for API Request/Response Models

Following Interface Segregation Principle - separate schemas for different use cases.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class EntityType(str, Enum):
    """Entity type enumeration matching SNOMED CT semantic tags."""
    DISORDER = "Disorder"
    ANATOMY = "Anatomy"
    PROCEDURE = "Procedure"
    MEDICATION = "Medication"
    OBSERVATION = "Observation"


class EntityResponse(BaseModel):
    """Response model for a single extracted entity."""
    id: str = Field(..., description="Unique entity identifier")
    text: str = Field(..., description="The extracted text span")
    type: EntityType = Field(..., description="Entity semantic type")
    start: int = Field(..., description="Start character offset in source text")
    end: int = Field(..., description="End character offset in source text")
    snomed_code: Optional[str] = Field(None, description="SNOMED CT concept ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "1",
                "text": "hypertension",
                "type": "Disorder",
                "start": 75,
                "end": 87,
                "snomed_code": "38341003",
                "confidence": 0.95
            }
        }


class NoteResponse(BaseModel):
    """Response model for a clinical note (without entities)."""
    note_id: str = Field(..., description="Unique note identifier")
    text: str = Field(..., description="Clinical note text content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "note_id": "1",
                "text": "Patient presented with acute onset of left-sided weakness..."
            }
        }


class NoteWithEntitiesResponse(BaseModel):
    """Response model for a clinical note with extracted entities."""
    note_id: str
    text: str
    entities: List[EntityResponse]


class AnalyzeNoteRequest(BaseModel):
    """Request model for note analysis."""
    text: str = Field(..., min_length=1, description="Clinical text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Patient with hypertension and diabetes mellitus presented with stroke symptoms."
            }
        }


class AnalyzeNoteResponse(BaseModel):
    """Response model for note analysis."""
    entities: List[EntityResponse]
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="NER model version used")


class SnomedDetailResponse(BaseModel):
    """Response model for SNOMED CT concept details."""
    code: str = Field(..., description="SNOMED CT concept ID")
    preferred_term: str = Field(..., description="Preferred term for the concept")
    description: str = Field(..., description="Clinical description")
    parents: List[str] = Field(default_factory=list, description="Parent concepts in hierarchy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "38341003",
                "preferred_term": "Hypertensive disorder",
                "description": "A disorder characterized by high blood pressure",
                "parents": ["Cardiovascular disease", "Disorder of cardiovascular system"]
            }
        }
