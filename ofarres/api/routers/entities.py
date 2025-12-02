"""
Entities Router

Handles SNOMED CT entity operations.
Following Single Responsibility Principle.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..models.schemas import EntityResponse, SnomedDetailResponse
from ..services.entity_service import EntityService

router = APIRouter()

entity_service = EntityService()


@router.get("/entities/{note_id}", response_model=List[EntityResponse])
async def get_entities_for_note(note_id: str):
    """
    Get all extracted entities for a specific note.
    
    - **note_id**: The unique identifier of the note
    """
    entities = entity_service.get_entities_for_note(note_id)
    if entities is None:
        raise HTTPException(status_code=404, detail=f"No entities found for note {note_id}")
    return entities


@router.get("/snomed/{code}", response_model=SnomedDetailResponse)
async def get_snomed_details(code: str):
    """
    Get detailed information about a SNOMED CT concept.
    
    - **code**: The SNOMED CT concept ID
    """
    details = entity_service.get_snomed_details(code)
    if not details:
        raise HTTPException(status_code=404, detail=f"SNOMED code {code} not found")
    return details
