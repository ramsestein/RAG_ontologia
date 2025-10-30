"""
Módulo RAG (Retrieval-Augmented Generation) usando FAISS
"""

import os
import pickle
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """Sistema de recuperación semántica usando FAISS"""
    
    def __init__(self, assets_dir: str):
        """
        Args:
            assets_dir: Directorio con índice FAISS y archivos pickle
        """
        self.assets_dir = assets_dir
        self.faiss_index = None
        self.embedding_model = None
        self.conceptos = []
        self.narrativas = []
        
        self._load_index()
        self._load_ontology()
    
    def _load_index(self):
        """Carga el índice FAISS pre-construido"""
        
        index_path = os.path.join(self.assets_dir, 'ontology.index')
        
        if not os.path.exists(index_path):
            print(f"[RAG] [WARNING] Índice FAISS no encontrado en {index_path}")
            print("[RAG] Ejecuta: python strategies/04_utils/ontology_preprocessor.py")
            return
        
        try:
            print(f"[RAG] Cargando índice FAISS...")
            self.faiss_index = faiss.read_index(index_path)
            
            # Cargar metadata
            metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RAG] [OK] Índice cargado: {metadata['n_concepts']} conceptos")
            
            # Cargar modelo de embeddings
            print("[RAG] Cargando modelo de embeddings...")
            self.embedding_model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
            
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
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Recupera conceptos similares usando búsqueda semántica
        
        Args:
            query: Texto de consulta
            k: Número de resultados a devolver
            
        Returns:
            Lista de tuplas (concepto_id, narrativa, distancia)
        """
        if self.faiss_index is None or self.embedding_model is None:
            print("[RAG] [WARNING] Sistema RAG no disponible, usando búsqueda simple")
            return self._simple_search(query, k)
        
        try:
            # Generar embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Buscar en FAISS
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), k
            )
            
            # Formatear resultados
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.conceptos):
                    concepto = self.conceptos[idx]
                    narrativa = self.narrativas[idx]
                    distancia = distances[0][i]
                    
                    # Filtrar solo códigos numéricos
                    if str(concepto).isdigit():
                        results.append((concepto, narrativa, distancia))
            
            return results
            
        except Exception as e:
            print(f"[RAG] [ERROR] Error en búsqueda: {e}")
            return self._simple_search(query, k)
    
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
