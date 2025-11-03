"""
Módulo RAG (Retrieval-Augmented Generation) usando FAISS
FIXED: Usa AutoModel + [CLS] token (SapBERT-style) y normaliza la query.
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


class RAGRetriever:
    """Sistema de recuperación semántica usando FAISS (SapBERT [CLS] mode)"""
    
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

    def _load_model_and_tokenizer(self):
        """Carga el modelo y tokenizador de HuggingFace (SapBERT-style)"""
        try:
            print(f"[RAG] Cargando modelo y tokenizador: {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            print(f"[RAG] [OK] Modelo cargado en {self.device}")
        except Exception as e:
            print(f"[RAG] [ERROR] No se pudo cargar el modelo de HuggingFace: {e}")

    
    def _load_index(self):
        """Carga el índice FAISS pre-construido"""
        index_path = os.path.join(self.assets_dir, 'ontology.index')
        if not os.path.exists(index_path):
            print(f"[RAG] [WARNING] Índice FAISS no encontrado en {index_path}")
            return
        
        try:
            print(f"[RAG] Cargando índice FAISS...")
            self.faiss_index = faiss.read_index(index_path)
            
            metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RAG] [OK] Índice cargado: {metadata['n_concepts']} conceptos (Modelo: {metadata.get('model_name', '??')})")
                
                # Validar que el índice se hizo con el mismo modelo
                if metadata.get('model_name') != MODEL_NAME:
                    print(f"[RAG] [WARNING] ¡El índice fue construido con un modelo diferente ({metadata.get('model_name')})!")
                    print(f"[RAG] [WARNING] Por favor, borra los assets y re-ejecuta ontology_preprocessor.py")

        except Exception as e:
            print(f"[RAG] [ERROR] Error cargando índice: {e}")
    
    def _load_ontology(self):
        """Carga conceptos y narrativas desde pickle"""
        concepts_path = os.path.join(self.assets_dir, 'ontology_concepts.pkl')
        narratives_path = os.path.join(self.assets_dir, 'ontology_narratives.pkl')
        
        try:
            with open(concepts_path, 'rb') as f:
                self.conceptos = pickle.load(f)
            with open(narratives_path, 'rb') as f:
                self.narrativas = pickle.load(f)
            print(f"[RAG] [OK] Ontología cargada: {len(self.conceptos)} conceptos")
        except Exception as e:
            print(f"[RAG] [ERROR] Error cargando ontología: {e}")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Genera un embedding normalizado para una sola query (SapBERT-style)"""
        with torch.no_grad():
            toks = self.tokenizer.encode_plus(
                query, 
                padding="max_length", 
                max_length=25, 
                truncation=True,
                return_tensors="pt"
            )
            toks_on_device = {k: v.to(self.device) for k, v in toks.items()}
            
            # Extraer [CLS] token
            cls_rep = self.model(**toks_on_device)[0][:, 0, :]
            
            # Normalizar para similitud de coseno
            emb = cls_rep.cpu().numpy()
            norm = np.linalg.norm(emb)
            normalized_emb = emb / norm
            
            return normalized_emb.astype('float32')

    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Recupera conceptos similares usando búsqueda semántica"""
        
        if self.faiss_index is None or self.model is None:
            print("[RAG] [WARNING] Sistema RAG no disponible. Faltan índice o modelo.")
            return []
        
        try:
            # 1. Generar embedding de la query (normalizado, estilo [CLS])
            query_embedding = self._get_query_embedding(query)
            
            # 2. Buscar en FAISS
            # k tiene que ser un int, aseguramos
            k = int(k)
            distances, indices = self.faiss_index.search(query_embedding, k)
            
            # 3. Formatear resultados
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0: # FAISS puede devolver -1 si no encuentra nada
                    continue
                if idx < len(self.conceptos):
                    concepto = self.conceptos[idx]
                    narrativa = self.narrativas[idx]
                    
                    # 'distances' con IndexFlatIP son Similitud de Coseno (ya 0.0-1.0)
                    # ¡No multiplicamos por -1! El preprocesador ya normalizó
                    # y FAISS/IP devuelve el producto punto, que ES la similitud.
                    similarity = float(distances[0][i]) 
                    
                    if str(concepto).isdigit():
                        results.append((concepto, narrativa, similarity))

            return results
            
        except Exception as e:
            print(f"[RAG] [ERROR] Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def _simple_search(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        """Búsqueda simple de texto como fallback"""
        
        results = []
        query_lower = query.lower()
        
        for concepto, narrativa in zip(self.conceptos, self.narrativas):
            if not str(concepto).isdigit():
                continue
            
            score = sum(1 for palabra in query_lower.split() 
                       if palabra in narrativa.lower())
            
            if score > 0:
                results.append((concepto, narrativa, 1.0 / (1.0 + score)))
        
        results.sort(key=lambda x: x[2])
        return results[:k]
