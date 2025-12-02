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


# ==============================================================================
# BENCHMARK SCHEMAS
# ==============================================================================

class BenchmarkMode(str, Enum):
    """Benchmark execution mode."""
    ALL = "all"           # Run each model individually
    ASSEMBLY = "assembly" # Run all models together (ensemble)
    SINGLE = "single"     # Run a specific model


class AnnotationInput(BaseModel):
    """Input model for a single ground truth annotation."""
    start: int
    end: int
    text: str
    concept_id: Optional[str] = None


class NoteInput(BaseModel):
    """Input model for a clinical note."""
    note_id: str
    text: str


class GroundTruthInput(BaseModel):
    """Input model for ground truth annotations."""
    note_id: str
    annotations: List[AnnotationInput]


class NERBenchmarkRequest(BaseModel):
    """Request model for NER benchmarking."""
    notes: List[NoteInput] = Field(..., description="List of clinical notes")
    ground_truth: List[GroundTruthInput] = Field(..., description="Ground truth annotations")
    model_id: str = Field(default="ground_truth", description="NER model ID to evaluate")
    iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="IoU threshold for matching")


class NoteMetrics(BaseModel):
    """Metrics for a single note."""
    note_id: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class SequentialContribution(BaseModel):
    """Sequential contribution of each model in assembly mode."""
    model_id: str
    incremental_recall: float  # New recall added by this model
    cumulative_recall: float   # Total recall up to this model


class ModelBenchmarkResult(BaseModel):
    """Result for a single model's benchmark."""
    model_id: str
    precision: float
    recall: float
    f1_micro: float
    f1_macro: float
    f1_harmonic: float  # Same as f1_macro (mean of per-note F1s)
    f1_arithmetic: float  # Mean of per-note (P+R)/2
    total_tp: int
    total_fp: int
    total_fn: int
    processing_time_s: float
    per_note_metrics: List[NoteMetrics]


class NERBenchmarkResponse(BaseModel):
    """Response model for NER benchmark results."""
    mode: BenchmarkMode
    iou_threshold: float
    
    # Results for each model (or single entry for assembly/single)
    results: List[ModelBenchmarkResult]
    
    # Sequential contribution (only for assembly mode)
    sequential_contribution: Optional[List[SequentialContribution]] = None
    
    # Processing info
    total_processing_time_ms: int
    notes_processed: int
    total_entities: int


class NERModelInfo(BaseModel):
    """Information about an available NER model."""
    id: str
    name: str
    description: str
    available: bool


class RAGMetrics(BaseModel):
    """Metrics for RAG evaluation."""
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank


class RAGBenchmarkResponse(BaseModel):
    """Response model for RAG benchmark status."""
    status: str
    message: str
    metrics: Optional[RAGMetrics] = None
