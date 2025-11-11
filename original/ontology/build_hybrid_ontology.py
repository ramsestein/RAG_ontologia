"""
Script para crear una ontología híbrida combinando:
1. Conceptos ground truth (32 conceptos) de concepts.json con definiciones ricas en español
2. Conceptos aleatorios (~2469 conceptos) de conceptos_con_narrativas.csv para añadir ruido

Autor: Oriol Farrés
Fecha: 27 de octubre de 2025
"""

import pandas as pd
import json
import os
from pathlib import Path

# Rutas de archivos
ONTOLOGY_DIR = Path(__file__).parent  # Este script está en /ontology
PROJECT_ROOT = ONTOLOGY_DIR.parent     # El directorio raíz del proyecto
CONCEPTS_JSON = ONTOLOGY_DIR / "concepts.json"
FULL_ONTOLOGY_CSV = ONTOLOGY_DIR / "conceptos_con_narrativas.csv"
OUTPUT_HYBRID_CSV = PROJECT_ROOT / "hybrid_ontology.csv"  # Guardar en raíz del proyecto

# Parámetros
TARGET_TOTAL_CONCEPTS = 2500  # 32 ground truth + ~2468 ruido


def load_ground_truth_concepts():
    """
    Carga los 31 conceptos ground truth desde concepts.json
    y crea narrativas ricas en español
    """
    print("="*80)
    print("1. CARGANDO CONCEPTOS GROUND TRUTH")
    print("="*80)
    
    with open(CONCEPTS_JSON, 'r', encoding='utf-8') as f:
        concepts_data = json.load(f)
    
    print(f"   Conceptos cargados desde {CONCEPTS_JSON.name}: {len(concepts_data)}")
    
    # Crear narrativas ricas en español
    ground_truth_rows = []
    
    for concept_id, data in concepts_data.items():
        # Construir narrativa rica en español similar al formato de conceptos_con_narrativas.csv
        narrative_parts = []
        
        # Añadir código
        narrative_parts.append(f"{concept_id} tiene código {concept_id}")
        
        # Añadir término preferido
        preferred = data['preferred_term']
        narrative_parts.append(f"{concept_id} tiene término preferido {preferred}")
        
        # Añadir sinónimos
        for synonym in data['synonyms']:
            narrative_parts.append(f"{concept_id} tiene sinónimo {synonym}")
        
        # Añadir definición (la parte más importante para búsqueda semántica)
        definition = data['definition']
        narrative_parts.append(f"{concept_id} se define como: {definition}")
        
        # Añadir información de ontología
        narrative_parts.append(f"{concept_id} pertenece a la terminología snomed")
        narrative_parts.append(f"{concept_id} es de tipo Class")
        narrative_parts.append(f"{concept_id} es de tipo Clinical_Finding")
        
        # Unir todo en una narrativa
        narrative = " ".join(narrative_parts)
        
        ground_truth_rows.append({
            'concepto': concept_id,
            'narrativa': narrative
        })
    
    ground_truth_df = pd.DataFrame(ground_truth_rows)
    
    print(f"   ✅ Creadas {len(ground_truth_df)} narrativas para conceptos ground truth")
    print(f"\n   Ejemplo de narrativa:")
    print(f"   {ground_truth_df.iloc[0]['concepto']}: {ground_truth_df.iloc[0]['narrativa'][:150]}...")
    
    return ground_truth_df


def load_noise_concepts(ground_truth_ids, target_noise_count):
    """
    Carga conceptos aleatorios de conceptos_con_narrativas.csv
    excluyendo los conceptos ground truth
    """
    print("\n" + "="*80)
    print("2. CARGANDO CONCEPTOS DE RUIDO")
    print("="*80)
    
    # Cargar ontología completa
    full_ontology_df = pd.read_csv(FULL_ONTOLOGY_CSV)
    print(f"   Ontología completa cargada: {len(full_ontology_df)} conceptos")
    
    # Convertir IDs a string para comparación consistente
    full_ontology_df['concepto'] = full_ontology_df['concepto'].astype(str)
    
    # Filtrar conceptos que NO están en ground truth
    available_concepts = full_ontology_df[~full_ontology_df['concepto'].isin(ground_truth_ids)]
    
    print(f"   Conceptos disponibles (excluyendo ground truth): {len(available_concepts)}")
    
    # Calcular cuántos necesitamos
    needed = min(target_noise_count, len(available_concepts))
    
    print(f"   Conceptos de ruido necesarios: {needed}")
    
    # Muestreo aleatorio
    noise_df = available_concepts.sample(n=needed, random_state=42)
    
    print(f"   ✅ Muestreados {len(noise_df)} conceptos de ruido")
    
    return noise_df


def build_hybrid_ontology():
    """
    Construye la ontología híbrida combinando ground truth y ruido
    """
    print("\n" + "="*80)
    print("CONSTRUCCIÓN DE ONTOLOGÍA HÍBRIDA")
    print("="*80)
    print(f"Objetivo: {TARGET_TOTAL_CONCEPTS} conceptos totales")
    print(f"  - ~32 conceptos ground truth (cobertura 100% requerida)")
    print(f"  - ~{TARGET_TOTAL_CONCEPTS - 32} conceptos de ruido")
    print()
    
    # 1. Cargar conceptos ground truth
    ground_truth_df = load_ground_truth_concepts()
    ground_truth_ids = set(ground_truth_df['concepto'].astype(str))
    
    # 2. Cargar conceptos de ruido
    target_noise = TARGET_TOTAL_CONCEPTS - len(ground_truth_df)
    noise_df = load_noise_concepts(ground_truth_ids, target_noise)
    
    # 3. Combinar
    print("\n" + "="*80)
    print("3. COMBINANDO CONCEPTOS")
    print("="*80)
    
    hybrid_df = pd.concat([ground_truth_df, noise_df], ignore_index=True)
    
    # Eliminar duplicados por si acaso
    hybrid_df = hybrid_df.drop_duplicates(subset=['concepto'])
    
    print(f"   Conceptos ground truth:     {len(ground_truth_df):>6}")
    print(f"   Conceptos de ruido:         {len(noise_df):>6}")
    print(f"   {'-'*80}")
    print(f"   TOTAL en ontología híbrida: {len(hybrid_df):>6}")
    
    # 4. Verificar cobertura 100% de ground truth
    print("\n" + "="*80)
    print("4. VERIFICACIÓN DE COBERTURA")
    print("="*80)
    
    hybrid_concept_ids = set(hybrid_df['concepto'].astype(str))
    coverage = len(ground_truth_ids & hybrid_concept_ids) / len(ground_truth_ids) * 100
    
    print(f"   Conceptos ground truth requeridos: {len(ground_truth_ids):>5}")
    print(f"   Conceptos ground truth en híbrida: {len(ground_truth_ids & hybrid_concept_ids):>5}")
    print(f"   Porcentaje de cobertura:            {coverage:>5.1f}%")
    
    if coverage < 100:
        missing = ground_truth_ids - hybrid_concept_ids
        print(f"\n   ❌ ERROR: Cobertura no es 100%! Conceptos faltantes:")
        for concept in sorted(missing):
            print(f"      - {concept}")
        raise ValueError("Cobertura incompleta de conceptos ground truth")
    else:
        print(f"\n   ✅ COBERTURA VERIFICADA: Todos los conceptos ground truth incluidos")
    
    # 5. Guardar ontología híbrida
    print("\n" + "="*80)
    print("5. GUARDANDO ONTOLOGÍA HÍBRIDA")
    print("="*80)
    
    hybrid_df.to_csv(OUTPUT_HYBRID_CSV, index=False, encoding='utf-8')
    
    print(f"   ✅ Ontología híbrida guardada en: {OUTPUT_HYBRID_CSV.name}")
    print(f"   Tamaño del archivo: {OUTPUT_HYBRID_CSV.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 6. Mostrar muestra de narrativas
    print("\n" + "="*80)
    print("6. MUESTRA DE NARRATIVAS")
    print("="*80)
    
    print("\n[CONCEPTOS GROUND TRUTH - Primeros 3]")
    for idx, row in ground_truth_df.head(3).iterrows():
        print(f"  {row['concepto']}: {row['narrativa'][:120]}...")
    
    print("\n[CONCEPTOS DE RUIDO - Primeros 3]")
    for idx, row in noise_df.head(3).iterrows():
        print(f"  {row['concepto']}: {row['narrativa'][:120]}...")
    
    print("\n" + "="*80)
    print("✅ ONTOLOGÍA HÍBRIDA CONSTRUIDA EXITOSAMENTE")
    print("="*80)
    print(f"\nArchivo de salida: {OUTPUT_HYBRID_CSV}")
    print(f"Siguiente paso: Reconstruir índice FAISS")
    print(f"  cd benchmark/strategies/04_utils && python ontology_preprocessor.py")
    
    return hybrid_df


if __name__ == "__main__":
    # Verificar que existen los archivos necesarios
    if not CONCEPTS_JSON.exists():
        raise FileNotFoundError(f"No se encontró {CONCEPTS_JSON}")
    
    if not FULL_ONTOLOGY_CSV.exists():
        raise FileNotFoundError(f"No se encontró {FULL_ONTOLOGY_CSV}")
    
    # Construir ontología híbrida
    hybrid_df = build_hybrid_ontology()
