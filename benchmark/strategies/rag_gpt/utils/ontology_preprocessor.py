#!/usr/bin/env python3
"""
Offline Index Builder for RAG Strategy (Ontology Pre-processor)
FIXED: Uses AutoModel + [CLS] token extraction per SapBERT README
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import faiss
from datetime import datetime
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# --- START: Robust Path Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
ASSETS_DIR = os.path.join(SCRIPT_DIR, 'assets')
ASSETS_DIR = os.path.abspath(ASSETS_DIR)
# --- END: Robust Path Setup ---


# Ensure assets directory exists
os.makedirs(ASSETS_DIR, exist_ok=True)

# Output paths
INDEX_PATH = os.path.join(ASSETS_DIR, 'ontology.index')
CONCEPTS_PATH = os.path.join(ASSETS_DIR, 'ontology_concepts.pkl')
NARRATIVES_PATH = os.path.join(ASSETS_DIR, 'ontology_narratives.pkl')
METADATA_PATH = os.path.join(ASSETS_DIR, 'ontology_metadata.pkl')

# --- MODELO CORRECTO (del README) ---
MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


def load_ontology_csv():
    """Carga la ontología híbrida"""
    print("\n" + "="*80)
    print("STEP 1: Loading Ontology Data")
    print("="*80)
    
    hybrid_path = os.path.join(PROJECT_ROOT, 'ontology', 'hybrid_ontology.csv')
    
    if os.path.exists(hybrid_path):
        print(f"[INFO] Loading HYBRID ONTOLOGY from: {hybrid_path}")
        df = pd.read_csv(hybrid_path)
        print(f"[SUCCESS] Loaded {len(df)} concepts")
        return df
    
    raise FileNotFoundError(f"Could not find ontology CSV at: {hybrid_path}")


def generate_embeddings(narratives, model_name=MODEL_NAME, batch_size=64):
    """
    Genera embeddings usando AutoModel y [CLS] token (SapBERT-style).
    Normaliza los embeddings para la búsqueda de similitud de coseno (IndexFlatIP).
    """
    print("\n" + "="*80)
    print("STEP 2: Generating Embeddings (SapBERT [CLS] Token Mode)")
    print("="*80)
    
    print(f"[INFO] Loading HuggingFace model: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"[INFO] Encoding {len(narratives)} narratives in batches of {batch_size}")
    
    all_embs_list = []
    
    with torch.no_grad():
        for i in tqdm(np.arange(0, len(narratives), batch_size)):
            batch_names = narratives[i:i+batch_size]
            
            toks = tokenizer.batch_encode_plus(
                batch_names, 
                padding="max_length", 
                max_length=25,  # Max length de 25 como en el README
                truncation=True,
                return_tensors="pt"
            )
            
            toks_on_device = {k: v.to(device) for k, v in toks.items()}
            
            # --- ¡LA LÓGICA CLAVE DEL README! ---
            # Extraer la representación del token [CLS]
            # model(...)[0] son las 'last_hidden_state'
            # [:, 0, :] selecciona el token [CLS] (índice 0) para cada ítem del batch
            cls_rep = model(**toks_on_device)[0][:, 0, :]
            
            all_embs_list.append(cls_rep.cpu().numpy())

    all_embs = np.concatenate(all_embs_list, axis=0)
    
    print(f"[SUCCESS] Generated embeddings with shape: {all_embs.shape}")
    
    # --- ¡PASO CRUCIAL PARA IndexFlatIP! ---
    print("[INFO] Normalizing embeddings for Cosine Similarity (L2 normalization)...")
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    normalized_embs = all_embs / norms
    
    print("[SUCCESS] Embeddings normalized.")
    return normalized_embs, all_embs.shape[1]


def build_faiss_index(embeddings):
    """Construye el índice FAISS (IndexFlatIP para Coseno)"""
    print("\n" + "="*80)
    print("STEP 3: Building Faiss Index")
    print("="*80)
    
    dimension = embeddings.shape[1]
    n_concepts = embeddings.shape[0]
    
    print(f"[INFO] Creating IndexFlatIP (cosine similarity) with dimension={dimension}")
    index = faiss.IndexFlatIP(dimension)
    
    print(f"[INFO] Adding {n_concepts} vectors to index...")
    index.add(embeddings.astype('float32'))
    
    print(f"[SUCCESS] Faiss index built with {index.ntotal} vectors")
    return index


def save_artifacts(index, concepts, narratives, embedding_dim):
    """Guarda los artefactos en disco"""
    print("\n" + "="*80)
    print("STEP 4: Saving Artifacts")
    print("="*80)
    
    # Guardar índice
    print(f"[INFO] Saving Faiss index to: {INDEX_PATH}")
    faiss.write_index(index, INDEX_PATH)
    print(f"[SUCCESS] Index saved ({os.path.getsize(INDEX_PATH) / 1024 / 1024:.2f} MB)")
    
    # Guardar listas
    with open(CONCEPTS_PATH, 'wb') as f:
        pickle.dump(concepts, f)
    print(f"[SUCCESS] Concepts saved ({len(concepts)} items)")
        
    with open(NARRATIVES_PATH, 'wb') as f:
        pickle.dump(narratives, f)
    print(f"[SUCCESS] Narratives saved ({len(narratives)} items)")
    
    # Guardar metadata
    metadata = {
        'n_concepts': len(concepts),
        'embedding_dim': embedding_dim,
        'model_name': MODEL_NAME,
        'created_at': datetime.now().isoformat(),
        'index_type': 'IndexFlatIP' # ¡Correcto!
    }
    
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"[SUCCESS] Metadata saved")


def main():
    """Flujo principal de ejecución"""
    print("\n" + "="*80)
    print("RAG INDEX BUILDER - Offline Pre-computation")
    print("="*80)
    
    import time
    start_time = time.time()
    
    try:
        # 1. Cargar
        df_ontology = load_ontology_csv()
        concepts = df_ontology['concepto'].tolist()
        narratives = df_ontology['narrativa'].tolist()
        
        # 2. Generar Embeddings
        embeddings, dim = generate_embeddings(narratives)
        
        # 3. Construir Índice
        faiss_index = build_faiss_index(embeddings)
        
        # 4. Guardar
        save_artifacts(faiss_index, concepts, narratives, dim)
        
        elapsed_time = time.time() - start_time
        print("\n" + "="*80)
        print("[SUCCESS] INDEX BUILD COMPLETED SUCCESSFULLY")
        print(f"[TIME]  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
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