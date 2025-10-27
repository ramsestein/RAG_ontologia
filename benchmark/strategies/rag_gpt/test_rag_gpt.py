#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test unitario para RAG+GPT Strategy
Procesa solo las notas de entrenamiento con RAG+GPT
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time

# Configurar path
SCRIPT_DIR = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.pipeline import RAGGPTPipeline
from evaluation.metrics_calculator import MetricsCalculator


def load_training_data():
    """Carga datos de entrenamiento"""
    data_dir = BENCHMARK_DIR / "data"
    
    # Cargar notas
    notes_df = pd.read_csv(data_dir / "mimic-iv_notes_training_set.csv")
    
    # Cargar ground truth
    annotations_df = pd.read_csv(data_dir / "train_annotations.csv")
    
    return notes_df, annotations_df


def main():
    print("="*80)
    print("RAG+GPT Strategy - Test Unitario")
    print("="*80)
    
    # 1. Cargar datos
    print("\n1. Cargando datos de entrenamiento...")
    notes_df, ground_truth_df = load_training_data()
    print(f"   [OK] {len(notes_df)} notas cargadas")
    print(f"   [OK] {len(ground_truth_df)} anotaciones de ground truth")
    
    # 2. Inicializar pipeline
    print("\n2. Inicializando pipeline RAG+GPT...")
    pipeline = RAGGPTPipeline(verbose=True)
    
    # 3. Ejecutar predicciones
    print("\n3. Ejecutando predicciones...")
    start_time = time.time()
    
    predictions_df = pipeline.predict(notes_df)
    
    execution_time = time.time() - start_time
    
    print(f"\n[OK] Predicciones completadas en {execution_time:.2f} segundos")
    print(f"   Total predicciones: {len(predictions_df)}")
    
    # 4. Evaluar métricas
    print("\n4. Calculando métricas...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(predictions_df, ground_truth_df)
    
    # 5. Mostrar resultados
    print("\n" + "="*80)
    print("RESULTADOS - RAG+GPT Strategy")
    print("="*80)
    
    print(f"\n[MÉTRICAS]")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")
    print(f"   F1-Score:   {metrics['f1']:.4f}")
    print(f"   Coverage:   {metrics.get('coverage', 1.0):.4f}")
    
    print(f"\n[CONTADORES]")
    print(f"   Predicciones:     {len(predictions_df)}")
    print(f"   Exact Matches:    {metrics.get('exact_matches', 0)}")
    print(f"   Partial Matches:  {metrics.get('partial_matches', 0)}")
    print(f"   Ground Truth:     {len(ground_truth_df)}")
    
    print(f"\n[TIEMPO]")
    print(f"   Tiempo de ejecución: {execution_time:.2f} segundos")
    print(f"   Tiempo por nota:     {execution_time/len(notes_df):.2f} segundos")
    
    print("\n" + "="*80)
    
    # 6. Guardar predicciones
    output_dir = BENCHMARK_DIR / "results" / "rag_gpt_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n[OK] Predicciones guardadas en: {predictions_path}")
    
    # 7. Análisis de códigos usados
    print("\n[ANÁLISIS DE CÓDIGOS]")
    code_counts = predictions_df['concept_id'].value_counts()
    print(f"   Códigos únicos usados: {len(code_counts)}")
    print(f"\n   Top 10 códigos más frecuentes:")
    for code, count in code_counts.head(10).items():
        print(f"      {code}: {count} veces")
    
    # Verificar códigos fallback
    fallback_codes = ['404684003', 'LINKING_FAILED']
    fallback_count = sum(predictions_df['concept_id'].isin(fallback_codes))
    if fallback_count > 0:
        print(f"\n   [WARNING] ADVERTENCIA: {fallback_count} predicciones con códigos fallback")
    else:
        print(f"\n   [OK] No se usaron códigos fallback")
    
    print("\n" + "="*80)
    print("Test completado")
    print("="*80)


if __name__ == "__main__":
    main()
