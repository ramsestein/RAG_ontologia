"""
RAG Retrieval Module - FAISS/SapBERT semantic search
Handles FAISS index loading and embedding generation
"""

import os
import pickle
import faiss
from typing import List, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --- MODELO CORRECTO (del README) ---
MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


class FAISSRetriever:
    """Sistema de recuperación semántica usando FAISS (SapBERT mean-pooling)"""
    
    def __init__(self, assets_dir: str):
        """
        Args:
            assets_dir: Directorio con índice FAISS y archivos pickle
        """
        self.assets_dir = assets_dir
        self.faiss_index = None
        self.conceptos = []
        self.narrativas = []
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        self._load_model_and_tokenizer()
        self._load_index()
        self._load_ontology()
        
        # Validación de coherencia
        try:
            self._validate_index_coherence()
            print("[RETRIEVAL] [OK] Validación de índice superada")
        except Exception as e:
            print(f"[RETRIEVAL] [ERROR] {e}")
            raise

    def _load_model_and_tokenizer(self):
        """Carga el modelo y tokenizador de HuggingFace (SapBERT-style)"""
        try:
            print(f"[RETRIEVAL] Cargando modelo y tokenizador: {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            print(f"[RETRIEVAL] [OK] Modelo cargado en {self.device}")
        except Exception as e:
            print(f"[RETRIEVAL] [ERROR] No se pudo cargar el modelo de HuggingFace: {e}")

    def _load_index(self):
        """Carga el índice FAISS pre-construido"""
        index_path = os.path.join(self.assets_dir, 'ontology.index')
        if not os.path.exists(index_path):
            print(f"[RETRIEVAL] [WARNING] Índice FAISS no encontrado en {index_path}")
            return
        
        try:
            print(f"[RETRIEVAL] Cargando índice FAISS...")
            self.faiss_index = faiss.read_index(index_path)
            
            metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RETRIEVAL] [OK] Índice cargado: {metadata['n_concepts']} conceptos (Modelo: {metadata.get('model_name', '??')})")
                
                # Validar que el índice se hizo con el mismo modelo
                if metadata.get('model_name') != MODEL_NAME:
                    print(f"[RETRIEVAL] [WARNING] ¡El índice fue construido con un modelo diferente ({metadata.get('model_name')})!")
                    print(f"[RETRIEVAL] [WARNING] Por favor, borra los assets y re-ejecuta build_index.py")

        except Exception as e:
            print(f"[RETRIEVAL] [ERROR] Error cargando índice: {e}")
    
    def _load_ontology(self):
        """Carga conceptos y narrativas desde pickle"""
        concepts_path = os.path.join(self.assets_dir, 'ontology_concepts.pkl')
        narratives_path = os.path.join(self.assets_dir, 'ontology_narratives.pkl')
        
        try:
            with open(concepts_path, 'rb') as f:
                self.conceptos = pickle.load(f)
            with open(narratives_path, 'rb') as f:
                self.narrativas = pickle.load(f)
            print(f"[RETRIEVAL] [OK] Ontología cargada: {len(self.conceptos)} conceptos")
        except Exception as e:
            print(f"[RETRIEVAL] [ERROR] Error cargando ontología: {e}")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Genera embedding normalizado (mean pooling con máscara)."""
        text = query if isinstance(query, str) else str(query)
        with torch.no_grad():
            toks = self.tokenizer.encode_plus(
                text,
                padding=True,
                max_length=64,   # ventana mayor para términos compuestos
                truncation=True,
                return_tensors="pt"
            )
            toks_on_device = {k: v.to(self.device) for k, v in toks.items()}

            outputs = self.model(**toks_on_device)
            last_hidden = outputs.last_hidden_state            # (B, T, H)
            mask = toks_on_device["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            # mean pooling sobre tokens válidos
            sum_vec = (last_hidden * mask).sum(dim=1)          # (B, H)
            len_vec = mask.sum(dim=1).clamp(min=1)             # (B, 1)
            mean_vec = sum_vec / len_vec

            emb = mean_vec.cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            normalized_emb = emb / np.clip(norm, 1e-12, None)

            return normalized_emb.astype("float32")

    def search(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        """
        Búsqueda FAISS para una sola query.
        
        Returns:
            Lista de tuplas (concepto, narrativa, similitud)
        """
        if self.faiss_index is None or self.model is None:
            print("[RETRIEVAL] [WARNING] Sistema RAG no disponible. Faltan índice o modelo.")
            return []

        try:
            q_emb = self._get_query_embedding(query)
            k = int(k)
            distances, indices = self.faiss_index.search(q_emb, k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                if idx < len(self.conceptos):
                    concepto = self.conceptos[idx]
                    narrativa = self.narrativas[idx]
                    similarity = float(distances[0][i])
                    if str(concepto).isdigit():
                        results.append((concepto, narrativa, similarity))
            return results
        except Exception as e:
            print(f"[RETRIEVAL] [ERROR] Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _validate_index_coherence(self):
        """Valida que el índice sea IP (coseno), tamaños coincidan y el modelo sea el esperado."""
        if self.faiss_index is None:
            raise RuntimeError("Índice FAISS no cargado.")

        # (a) métrica = inner product (cosine con embeddings normalizados)
        try:
            metric = self.faiss_index.metric_type
        except Exception:
            metric = None

        if metric != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError(
                f"Índice FAISS con métrica no soportada ({metric}). "
                f"Reconstruye el índice con IndexFlatIP y embeddings L2-normalizados."
            )

        # (b) tamaños deben coincidir
        if self.faiss_index.ntotal != len(self.conceptos) or len(self.conceptos) != len(self.narrativas):
            raise RuntimeError(
                f"Inconsistencia: index.ntotal={self.faiss_index.ntotal}, "
                f"conceptos={len(self.conceptos)}, narrativas={len(self.narrativas)}."
            )

        # (c) modelo del índice debe coincidir
        metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            idx_model = metadata.get('model_name')
            if idx_model != MODEL_NAME:
                raise RuntimeError(
                    f"El índice se construyó con '{idx_model}' y el runtime usa '{MODEL_NAME}'. "
                    f"Reconstruye el índice con el mismo modelo."
                )
