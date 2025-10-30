"""
Script para encontrar el threshold óptimo de SapBERT
Analiza las distancias de los mejores matches
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from benchmark.strategies.rag_gpt.core.rag import RAGRetriever

# Queries de ejemplo del dataset
test_queries = [
    "hypertension",
    "diabetes mellitus", 
    "stroke",
    "atrial fibrillation",
    "weakness",
    "aphasia",
    "hemiplegia",
    "nausea",
    "headache",
    "vertigo",
    "smoking",
    "hyperlipidemia"
]

print("="*80)
print("ANÁLISIS DE DISTANCIAS - SapBERT")
print("="*80)

# Inicializar RAG
assets_dir = os.path.join(os.path.dirname(__file__), '04_utils', 'assets')
rag = RAGRetriever(assets_dir)

all_distances = []

for query in test_queries:
    results = rag.retrieve(query, k=10)
    
    if results:
        best_dist = results[0][2]
        worst_dist = results[-1][2]
        all_distances.extend([r[2] for r in results])
        
        print(f"\nQuery: '{query}'")
        print(f"  Best match:  {results[0][0]} (dist: {best_dist:.3f})")
        print(f"  Worst match: {results[-1][0]} (dist: {worst_dist:.3f})")

if all_distances:
    all_distances.sort()
    print("\n" + "="*80)
    print("ESTADÍSTICAS GLOBALES")
    print("="*80)
    print(f"Distancia mínima:     {min(all_distances):.3f}")
    print(f"Distancia máxima:     {max(all_distances):.3f}")
    print(f"Distancia media:      {sum(all_distances)/len(all_distances):.3f}")
    print(f"Percentil 50:         {all_distances[len(all_distances)//2]:.3f}")
    print(f"Percentil 75:         {all_distances[int(len(all_distances)*0.75)]:.3f}")
    print(f"Percentil 90:         {all_distances[int(len(all_distances)*0.90)]:.3f}")
    
    print("\n" + "="*80)
    print("RECOMENDACIONES DE THRESHOLD")
    print("="*80)
    print(f"Muy restrictivo:  {all_distances[int(len(all_distances)*0.25)]:.2f}")
    print(f"Balanceado:       {all_distances[int(len(all_distances)*0.50)]:.2f}")
    print(f"Permisivo:        {all_distances[int(len(all_distances)*0.75)]:.2f}")
    print(f"Muy permisivo:    {all_distances[int(len(all_distances)*0.90)]:.2f}")
