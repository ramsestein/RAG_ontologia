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
