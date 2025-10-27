#!/usr/bin/env python3
"""
Offline Index Builder for RAG Strategy (Ontology Pre-processor)

This script pre-computes the embeddings and Faiss index for the RAG+GPT4o strategy.
It is now located inside the strategy-specific assets folder.

Responsibilities:
  - Load ontology data
  - Generate embeddings using SentenceTransformer (1 embedding per narrative)
  - Build Faiss index
  - Save artifacts to disk for fast loading at runtime
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

# --- START: Robust Path Setup (Updated for new location) ---

# Get the absolute path to THIS script's directory (.../strategies/rag_gpt/utils)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the project root (.../RAG_ontologia)
# We need to go up FOUR levels ('..' to rag_gpt, '..' to strategies, '..' to benchmark, '..' to root)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

# Path to assets directory (sibling to this script, under rag_gpt/04_utils/assets)
ASSETS_DIR = os.path.join(SCRIPT_DIR, '..', '04_utils', 'assets')
ASSETS_DIR = os.path.abspath(ASSETS_DIR)

# --- END: Robust Path Setup ---


# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

# Output paths (now relative to ASSETS_DIR)
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
    
    # UPDATED: Use hybrid ontology
    hybrid_path = os.path.join(PROJECT_ROOT, 'hybrid_ontology.csv')
    fallback_path = os.path.join(PROJECT_ROOT, 'conceptos_con_narrativas.csv')
    
    # Try hybrid ontology first (preferred)
    if os.path.exists(hybrid_path):
        print(f"[INFO] Loading HYBRID ONTOLOGY from: {hybrid_path}")
        df = pd.read_csv(hybrid_path)
        print(f"[SUCCESS] Loaded {len(df)} concepts")
        print(f"           - Contains all training concepts (32 total)")
        print(f"           - Plus 2468 noise concepts for robustness testing")
        return df
    
    # Fallback to original (incomplete for stroke task)
    if os.path.exists(fallback_path):
        print(f"[WARNING] Hybrid ontology not found, using incomplete original")
        print(f"[INFO] Loading from: {fallback_path}")
        df = pd.read_csv(fallback_path)
        print(f"[SUCCESS] Loaded {len(df)} concepts")
        print(f"[WARNING] This ontology is missing 26 critical stroke concepts!")
        return df
    
    raise FileNotFoundError(
        f"Could not find ontology CSV at: {hybrid_path} or {fallback_path}"
    )


def generate_embeddings(narratives, model_name='all-MiniLM-L6-v2', batch_size=32, use_gpu=True):
    """
    Generate embeddings for all narrative descriptions using SentenceTransformer.
    (1 narrative = 1 embedding)
    """
    print("\n" + "="*80)
    print("STEP 2: Generating Embeddings")
    print("="*80)
    
    print(f"[INFO] Loading SentenceTransformer model: {model_name}")
    
    model = SentenceTransformer(model_name)
    device = model.device
    print(f"[INFO] Using device: {device}")
    
    if 'cuda' in str(device):
        print("[INFO] [READY] GPU detected! Encoding will be faster.")
        batch_size = 128
    else:
        print("[INFO] Using CPU. This may take several minutes...")
    
    print(f"[INFO] Encoding {len(narratives)} narratives (1 embedding per narrative)")
    print("[INFO] This is a ONE-TIME operation. Subsequent runs will load pre-built index.")
    
    # Generate embeddings
    embeddings = model.encode(
        narratives,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    
    print(f"[SUCCESS] Generated embeddings with shape: {embeddings.shape}")
    print(f"[INFO] Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, model


def build_faiss_index(embeddings):
    """
    Build a Faiss index from the embeddings.
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
    print("[OK] ALL ARTIFACTS SAVED SUCCESSFULLY")
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
    print("  1. Load ontology data (conceptos_con_narrativas.csv)")
    print("  2. Generate embeddings (1 per narrative)")
    print("  3. Build Faiss index")
    print("  4. Save all artifacts to disk")
    print("\n[WARNING]  This is a ONE-TIME operation (unless ontology data changes)")
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
        print("[SUCCESS] INDEX BUILD COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"[TIME]  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"[INFO] Indexed: {len(concepts)} concepts")
        print(f"[INFO] Embedding dimension: {embeddings.shape[1]}")
        print("\n[OK] The RAG strategy will now load instantly!")
        print("="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("[ERROR] ERROR BUILDING INDEX")
        print("="*80)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()