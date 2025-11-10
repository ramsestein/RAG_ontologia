#!/usr/bin/env python3
"""
Offline Index Builder for RAG Strategy (Ontology Pre-processor)
Robusto: 
- Extrae 'formas superficiales' (término preferido + sinónimos) de la narrativa.
- Usa mean pooling con ventana amplia (max_length=128, padding=True).
- Mantiene la narrativa original para mostrar contexto, pero embebe los textos depurados.
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
import re

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# Assets directory (nueva ubicación)
ASSETS_DIR = PROJECT_ROOT / "assets" / "ontology"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Output paths
INDEX_PATH = ASSETS_DIR / 'ontology.index'
CONCEPTS_PATH = ASSETS_DIR / 'ontology_concepts.pkl'
NARRATIVES_PATH = ASSETS_DIR / 'ontology_narratives.pkl'
METADATA_PATH = ASSETS_DIR / 'ontology_metadata.pkl'

# --- MODELO CORRECTO (del README) ---
MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


# ============================
# Limpieza: formas superficiales
# ============================
_PREF_RE = re.compile(
    r"tiene\s+t[eé]rmino\s+preferido\s+([^0-9]+?)(?=\s+\d{3,}|\s+tiene|\s+se define|$)",
    re.IGNORECASE
)
_SYN_RE = re.compile(
    r"tiene\s+sin[oó]nimo\s+([^0-9]+?)(?=\s+\d{3,}|\s+tiene|\s+se define|$)",
    re.IGNORECASE
)

def extract_surface_forms(narr: str, max_terms: int = 24) -> str:
    """
    Extrae 'término preferido' y todos los 'sinónimo' de la narrativa en español.
    Devuelve una cadena corta adecuada para SapBERT.
    """
    if not isinstance(narr, str) or not narr.strip():
        return ""

    terms = []

    m = _PREF_RE.search(narr)
    if m:
        terms.append(m.group(1).strip())

    syns = _SYN_RE.findall(narr)
    terms.extend([s.strip() for s in syns])

    # limpieza de colas y deduplicación case-insensitive
    clean = []
    seen = set()
    for t in terms:
        t = re.sub(r"[.;,:]+$", "", t).strip()
        tl = t.lower()
        if tl and tl not in seen:
            seen.add(tl)
            clean.append(t)

    # fallback si no encontramos nada útil
    if not clean:
        # usa la narrativa pero sin el ruido más obvio (trozos 'se define como:')
        narr_short = re.split(r"\bse\s+define\s+como\b", narr, flags=re.IGNORECASE)[0]
        return narr_short.strip()[:512]

    return " [SEP] ".join(clean[:max_terms])


def load_ontology_csv():
    """Carga la ontología híbrida"""
    print("\n" + "="*80)
    print("STEP 1: Loading Ontology Data")
    print("="*80)
    
    hybrid_path = PROJECT_ROOT / 'ontology' / 'hybrid_ontology.csv'
    
    if hybrid_path.exists():
        print(f"[INFO] Loading HYBRID ONTOLOGY from: {hybrid_path}")
        df = pd.read_csv(hybrid_path)
        print(f"[SUCCESS] Loaded {len(df)} concepts")
        return df
    
    raise FileNotFoundError(f"Could not find ontology CSV at: {hybrid_path}")


def generate_embeddings(narratives, model_name=MODEL_NAME, batch_size=64):
    """
    Genera embeddings con mean pooling (más robusto que [CLS] para SapBERT).
    Normaliza L2 para IndexFlatIP (cosine).
    Usa textos depurados (formas superficiales).
    """
    print("\n" + "="*80)
    print("STEP 2: Generating Embeddings (SapBERT mean pooling)")
    print("="*80)

    print(f"[INFO] Loading HuggingFace model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # ---- depurar narrativas ----
    cleaned = [extract_surface_forms(x) for x in narratives]

    print(f"[INFO] Encoding {len(cleaned)} narratives in batches of {batch_size}")

    all_embs_list = []

    with torch.no_grad():
        for i in tqdm(np.arange(0, len(cleaned), batch_size)):
            batch = cleaned[i:i+batch_size]

            toks = tokenizer.batch_encode_plus(
                batch,
                padding=True,          # padding dinámico
                max_length=128,        # ventana amplia para cubrir sinónimos
                truncation=True,
                return_tensors="pt"
            )
            toks_on_device = {k: v.to(device) for k, v in toks.items()}

            outputs = model(**toks_on_device)
            last_hidden = outputs.last_hidden_state           # (B, T, H)
            mask = toks_on_device["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            sum_vec = (last_hidden * mask).sum(dim=1)         # (B, H)
            len_vec = mask.sum(dim=1).clamp(min=1)            # (B, 1)
            mean_vec = sum_vec / len_vec                      # (B, H)

            all_embs_list.append(mean_vec.cpu().numpy())

    all_embs = np.concatenate(all_embs_list, axis=0)
    print(f"[SUCCESS] Generated embeddings with shape: {all_embs.shape}")

    print("[INFO] Normalizing embeddings for Cosine Similarity (L2 normalization)...")
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    normalized_embs = all_embs / np.clip(norms, 1e-12, None)

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
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[SUCCESS] Index saved ({os.path.getsize(INDEX_PATH) / 1024 / 1024:.2f} MB)")
    
    # Guardar listas (narrativas originales para mostrar contexto)
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
        'index_type': 'IndexFlatIP'
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
        
        # 2. Generar Embeddings (sobre formas superficiales)
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
