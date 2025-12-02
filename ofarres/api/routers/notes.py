"""
Notes Router

Handles clinical notes CRUD operations and analysis.
Following Single Responsibility Principle.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

from ..models.schemas import (
    NoteResponse,
    NoteWithEntitiesResponse,
    AnalyzeNoteRequest,
    AnalyzeNoteResponse
)
from ..services.note_service import NoteService
from ..services.entity_service import EntityService

router = APIRouter()

# Dependency injection - services
note_service = NoteService()
entity_service = EntityService()


@router.get("/notes", response_model=List[NoteResponse])
async def get_all_notes():
    """
    Get all clinical notes from the database.
    
    Returns a list of all notes with their metadata.
    """
    notes = note_service.get_all_notes()
    return notes


@router.get("/notes/{note_id}", response_model=NoteWithEntitiesResponse)
async def get_note_with_entities(note_id: str):
    """
    Get a specific note with its extracted entities.
    
    - **note_id**: The unique identifier of the note
    """
    note = note_service.get_note_by_id(note_id)
    if not note:
        raise HTTPException(status_code=404, detail=f"Note with id {note_id} not found")
    
    entities = entity_service.get_entities_for_note(note_id)
    
    return NoteWithEntitiesResponse(
        note_id=note["note_id"],
        text=note["text"],
        entities=entities
    )


@router.post("/notes/analyze", response_model=AnalyzeNoteResponse)
async def analyze_note(request: AnalyzeNoteRequest):
    """
    Analyze a clinical note and extract medical entities.
    
    This endpoint processes the provided text and returns
    extracted SNOMED CT entities with their positions.
    """
    # For now, we use the ground truth as the analysis result
    # In production, this would call the NER model
    entities = entity_service.analyze_text(request.text)
    
    return AnalyzeNoteResponse(
        entities=entities,
        processing_time_ms=45,
        model_version="v1.0.0-ground-truth"
    )
