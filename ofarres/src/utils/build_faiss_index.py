import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
INPUT_JSON = PROJECT_ROOT / "ontology" / "multilingual_ontology.json"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "ontology"
INDEX_PATH = OUTPUT_DIR / "ontology_rag.index"
MAP_PATH = OUTPUT_DIR / "ontology_rag_map.pkl"

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

def main():
    print(f"[FAISS BUILDER] Leyendo {INPUT_JSON}...")
    
    if not INPUT_JSON.exists():
        print("❌ No existe el JSON multilingüe.")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    # Estrategia: Vectorizamos "Término: Definición"
    # Para cada idioma disponible.
    texts = []
    ids = []

    print("[INFO] Preparando textos...")
    for item in ontology:
        cid = item['concept_id']
        langs = item.get('languages', {})
        
        # Recorrer idiomas (es, en, ca)
        for lang, data in langs.items():
            terms = data.get('terms', [])
            definition = data.get('definition', "")
            
            if terms:
                # Usamos el primer término como representante + definición
                # Ej: "Stroke: Acute neurological deficit..."
                text_repr = f"{terms[0]}: {definition}"
                texts.append(text_repr)
                ids.append(cid)

    print(f"[INFO] {len(texts)} vectores a generar.")

    # Generar Embeddings
    print(f"[INFO] Cargando modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("[INFO] Codificando...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # Guardar Index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(MAP_PATH, 'wb') as f:
        pickle.dump(ids, f)

    print(f"[SUCCESS] Índice guardado en {INDEX_PATH}")

if __name__ == "__main__":
    main()