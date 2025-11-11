"""
RAG+GPT4o Strategy - Wrapper for benchmark compatibility
Maintains the original interface using the new modular architecture internally
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Setup path to allow imports
SCRIPT_DIR = Path(__file__).parent
RAG_GPT_DIR = SCRIPT_DIR / "rag_gpt"
SRC_DIR = RAG_GPT_DIR / "src"

# Add src to path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import after path setup
from pipeline import RAGGPTPipeline


class RAGWithGPT4oStrategy:
    """
    Wrapper to maintain benchmark interface compatibility
    Internally uses the new modular pipeline located in ./rag_gpt/
    """
    
    def __init__(self):
        """Initialize the modular pipeline"""
        print("[RAG+GPT4o] Initializing strategy...")
        self.pipeline = RAGGPTPipeline(verbose=False)
        print("[RAG+GPT4o] [OK] Initialization complete")
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Benchmark-compatible interface
        
        Args:
            notes_df: DataFrame with 'note_id' and 'text' columns
            
        Returns:
            DataFrame with predictions in standard format
        """
        return self.pipeline.predict(notes_df)
    
    def extract_entities(self, texto: str):
        """
        Compatible interface for entity extraction
        
        Args:
            texto: Medical note text
            
        Returns:
            List of SNOMED-CT coded entities
        """
        return self.pipeline.process_note(texto)
