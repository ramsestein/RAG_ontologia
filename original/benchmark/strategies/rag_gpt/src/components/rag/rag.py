"""
RAG Orchestrator - Coordinates Retrieval + Augmentation + (optional) Generation
Main interface for the RAG subsystem
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Setup paths for absolute imports
RAG_DIR = Path(__file__).parent.resolve()
COMPONENTS_DIR = RAG_DIR.parent
SRC_DIR = COMPONENTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from components.rag.retrieval import FAISSRetriever
from components.rag.augmentation import QueryAugmenter
from components.rag.generation import LLMValidator


class RAGRetriever:
    """
    Sistema RAG completo que orquesta:
    - Retrieval (FAISS/SapBERT)
    - Augmentation (query expansion, hints bilingües)
    - Generation (validación LLM opcional - actualmente en coding.py)
    """
    
    def __init__(self, assets_dir: str):
        """
        Args:
            assets_dir: Directorio con índice FAISS y recursos
        """
        self.assets_dir = assets_dir
        
        # Componentes
        self.retriever = FAISSRetriever(assets_dir)
        self.augmenter = QueryAugmenter(assets_dir)
        self.validator = LLMValidator()  # placeholder
        
        print("[RAG] [OK] Sistema RAG inicializado (R+A+G)")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Recupera conceptos similares usando búsqueda semántica con expansión bilingüe.
        
        Pipeline:
        1. Augmentation: Genera variantes de la query (EN/ES, lowercase)
        2. Retrieval: Búsqueda multi-query con fusión por máximo
        3. (Generation): Validación/re-ranking opcional (placeholder)
        
        Args:
            query: Query de búsqueda
            k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (concepto, narrativa, similitud) ordenadas por similitud
        """
        # 1. Augmentation: Generar variantes
        variants = self.augmenter.expand_query_variants(query)
        
        # 2. Retrieval: Multi-query con fusión
        results = self.retrieve_multi(variants, k=k)
        
        # 3. Generation: Validación opcional (por ahora no hace nada)
        validated = self.validator.validate_candidates(query, results)
        
        return validated
    
    def retrieve_multi(self, queries: List[str], k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Ejecuta varias queries y fusiona resultados por máximo de similitud por concepto.
        
        Args:
            queries: Lista de queries a ejecutar
            k: Número de resultados finales
            
        Returns:
            Lista fusionada de (concepto, narrativa, similitud)
        """
        if not queries:
            return []
        
        pool: Dict[str, Tuple[str, float]] = {}
        for q in queries:
            res = self.retriever.search(q, k)
            for concepto, narrativa, sim in res:
                # conservar la mejor similitud para cada concepto
                prev = pool.get(concepto)
                if (prev is None) or (sim > prev[1]):
                    pool[concepto] = (narrativa, sim)
        
        fused = [(c, n, s) for c, (n, s) in pool.items()]
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[:k]
