"""
Note Service

Handles business logic for clinical notes.
Following Single Responsibility Principle.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any


class NoteService:
    """Service for managing clinical notes."""
    
    def __init__(self):
        self._notes: List[Dict[str, Any]] = []
        self._load_notes()
    
    def _load_notes(self) -> None:
        """Load notes from JSON file."""
        notes_path = Path(__file__).parent.parent.parent / "backend" / "data" / "notes.json"
        
        if notes_path.exists():
            with open(notes_path, "r", encoding="utf-8") as f:
                self._notes = json.load(f)
        else:
            self._notes = []
    
    def get_all_notes(self) -> List[Dict[str, Any]]:
        """Get all clinical notes."""
        return [
            {"note_id": note["note_id"], "text": note["text"]}
            for note in self._notes
        ]
    
    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific note by ID."""
        for note in self._notes:
            if note["note_id"] == note_id:
                return note
        return None
    
    def get_note_count(self) -> int:
        """Get total number of notes."""
        return len(self._notes)
