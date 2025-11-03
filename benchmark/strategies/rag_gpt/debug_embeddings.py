#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script para verificar que los embeddings se están generando correctamente
"""

import sys
from pathlib import Path
import numpy as np

# Configurar path
SCRIPT_DIR = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.core.rag import RAGRetriever
from strategies.rag_gpt.utils.config import get_assets_dir

def main():
    print("="*80)
    print("DEBUG: Verificando normalización de embeddings")
    print("="*80)
    
    # Inicializar RAG
    assets_dir = get_assets_dir()
    print(f"\n[DEBUG] Cargando RAG desde: {assets_dir}")
    rag = RAGRetriever(str(assets_dir))
    
    # Test queries - términos médicos comunes que DEBEN estar en la ontología
    test_queries = [
        "hypertension",
        "diabetes",
        "headache",
        "stroke",
        "weakness",
        "atrial fibrillation"
    ]
    
    print("\n" + "="*80)
    print("TEST: Buscando términos médicos comunes")
    print("="*80)
    
    for query in test_queries:
        print(f"\n[QUERY] '{query}'")
        print("-" * 60)
        
        # Generar embedding
        emb = rag._get_query_embedding(query)
        print(f"  Shape: {emb.shape}")
        print(f"  Norm: {np.linalg.norm(emb):.6f} (debe ser ~1.0)")
        
        # Buscar en FAISS
        results = rag.retrieve(query, k=10)
        
        if not results:
            print(f"  ⚠️  [WARNING] NO SE ENCONTRARON RESULTADOS!")
        else:
            print(f"  ✓ {len(results)} resultados encontrados:")
            for i, (concepto, narrativa, sim) in enumerate(results[:5], 1):
                print(f"    {i}. [SIM: {sim:.4f}] {concepto}: {narrativa[:80]}...")
        
        # Verificar si las similitudes son razonables
        if results and results[0][2] < 0.5:
            print(f"  ⚠️  [WARNING] Similitud muy baja! Puede haber un problema de normalización")
    
    print("\n" + "="*80)
    print("Debug completado")
    print("="*80)

if __name__ == "__main__":
    main()
