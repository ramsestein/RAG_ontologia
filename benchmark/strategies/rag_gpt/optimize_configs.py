#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de optimización automática - Prueba múltiples configuraciones
Usa variables de entorno (ENV) para configurar el pipeline sin parchear archivos.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time
import json
from datetime import datetime

# Configurar path
SCRIPT_DIR = Path(__file__).parent.resolve()
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.pipeline import RAGGPTPipeline
from evaluation.metrics_calculator import MetricsCalculator


# Configuraciones a probar (ahora via ENV)
CONFIGURATIONS = [
    {
        "name": "Config 1: Threshold Bajo + Más Contexto",
        "TOP_K": 30,
        "THRESHOLD": 0.50,
        "MAX_DISPLAY": 12,
        "query_clinical_suffix": "disorder finding",
        "use_llm_validation": False
    },
    {
        "name": "Config 2: Threshold Medio + Contexto Optimizado",
        "TOP_K": 25,
        "THRESHOLD": 0.55,
        "MAX_DISPLAY": 10,
        "query_clinical_suffix": "disorder finding",
        "use_llm_validation": False
    },
    {
        "name": "Config 3: Threshold Medio-Alto + Contexto Moderado",
        "TOP_K": 20,
        "THRESHOLD": 0.58,
        "MAX_DISPLAY": 8,
        "query_clinical_suffix": "disorder finding",
        "use_llm_validation": False
    },
    {
        "name": "Config 4: Threshold Alto + Contexto Selectivo",
        "TOP_K": 20,
        "THRESHOLD": 0.65,
        "MAX_DISPLAY": 6,
        "query_clinical_suffix": "disorder finding",
        "use_llm_validation": False
    },
    {
        "name": "Config 5: Threshold Muy Bajo + Máximo Contexto",
        "TOP_K": 40,
        "THRESHOLD": 0.45,
        "MAX_DISPLAY": 15,
        "query_clinical_suffix": "disorder finding symptom",
        "use_llm_validation": False
    },
    {
        "name": "Config 6: Threshold Bajo + Más query enhancement",
        "TOP_K": 30,
        "THRESHOLD": 0.52,
        "MAX_DISPLAY": 10,
        "query_clinical_suffix": "disease disorder condition",
        "use_llm_validation": False
    },
    {
        "name": "Config 7: Balance Óptimo",
        "TOP_K": 25,
        "THRESHOLD": 0.53,
        "MAX_DISPLAY": 9,
        "query_clinical_suffix": "disorder finding",
        "use_llm_validation": False
    }
]


def load_training_data():
    """Carga datos de entrenamiento"""
    data_dir = BENCHMARK_DIR / "data"
    notes_df = pd.read_csv(data_dir / "mimic-iv_notes_training_set.csv")
    annotations_df = pd.read_csv(data_dir / "train_annotations.csv")
    return notes_df, annotations_df


def _set_env_from_config(config):
    """Inyecta la configuración por variables de entorno para el pipeline."""
    os.environ["RAG_TOP_K"] = str(config["TOP_K"])
    os.environ["RAG_THRESHOLD"] = str(config["THRESHOLD"])
    os.environ["RAG_MAX_DISPLAY"] = str(config["MAX_DISPLAY"])
    os.environ["RAG_QUERY_SUFFIX"] = str(config["query_clinical_suffix"])
    os.environ["RAG_USE_LLM_VALIDATION"] = "true" if config.get("use_llm_validation", False) else "false"
    # Puedes fijar el modelo y temperatura si quieres variar:
    # os.environ["RAG_LLM_MODEL"] = "gpt-4o"
    # os.environ["RAG_LLM_TEMPERATURE"] = "0.0"


def _clear_env():
    """Limpia variables para evitar fugas entre runs."""
    for k in ["RAG_TOP_K", "RAG_THRESHOLD", "RAG_MAX_DISPLAY", "RAG_QUERY_SUFFIX", "RAG_USE_LLM_VALIDATION", "RAG_LLM_MODEL", "RAG_LLM_TEMPERATURE"]:
        if k in os.environ:
            del os.environ[k]


def test_configuration(config, notes_df, ground_truth_df):
    """Prueba una configuración específica"""
    print(f"\n{'='*80}")
    print(f"Probando: {config['name']}")
    print(f"{'='*80}")
    print(f"Parámetros:")
    print(f"  TOP_K: {config['TOP_K']}")
    print(f"  THRESHOLD: {config['THRESHOLD']}")
    print(f"  MAX_DISPLAY: {config['MAX_DISPLAY']}")
    print(f"  Query Enhancement: '{config['query_clinical_suffix']}'")
    print(f"  LLM Validation: {'ON' if config.get('use_llm_validation', False) else 'OFF'}")

    # Aplicar configuración vía ENV
    _clear_env()
    _set_env_from_config(config)

    try:
        # Ejecutar pipeline
        print("\nEjecutando pipeline...")
        start_time = time.time()
        pipeline = RAGGPTPipeline(verbose=False)
        predictions_df = pipeline.predict(notes_df)
        execution_time = time.time() - start_time

        # Calcular métricas
        calculator = MetricsCalculator()
        metrics = calculator.calculate_metrics(predictions_df, ground_truth_df, "RAG+GPT4o")

        # Análisis de fallbacks
        fallback_codes = ['404684003', 'LINKING_FAILED']
        fallback_count = sum(predictions_df['concept_id'].isin(fallback_codes))
        fallback_rate = fallback_count / len(predictions_df) if len(predictions_df) > 0 else 0

        results = {
            "config": config,
            "metrics": metrics,
            "execution_time": execution_time,
            "predictions_count": len(predictions_df),
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate
        }

        # Mostrar resultados
        print(f"\n[RESULTADOS]")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1']:.4f} {'⭐' if metrics['f1'] >= 0.6 else ''}")
        print(f"  Predictions: {len(predictions_df)}")
        print(f"  Fallbacks:  {fallback_count} ({fallback_rate*100:.1f}%)")
        print(f"  Tiempo:     {execution_time:.2f}s")

        return results

    except Exception as e:
        print(f"\n[ERROR] Error ejecutando configuración: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        _clear_env()


def main():
    print("="*80)
    print("OPTIMIZACIÓN AUTOMÁTICA DE CONFIGURACIONES RAG+GPT (ENV-DRIVEN)")
    print("="*80)

    # Cargar datos
    print("\nCargando datos de entrenamiento...")
    notes_df, ground_truth_df = load_training_data()
    print(f"[OK] {len(notes_df)} notas, {len(ground_truth_df)} anotaciones")

    # Probar cada configuración
    all_results = []

    for i, config in enumerate(CONFIGURATIONS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Configuración {i}/{len(CONFIGURATIONS)}")
        print(f"{'#'*80}")

        result = test_configuration(config, notes_df, ground_truth_df)

        if result:
            all_results.append(result)

    # Resumen final
    print(f"\n\n{'='*80}")
    print("RESUMEN DE TODAS LAS CONFIGURACIONES")
    print(f"{'='*80}\n")

    # Ordenar por F1
    all_results.sort(key=lambda x: x['metrics']['f1'], reverse=True)

    print(f"{'Rank':<6} {'Config':<45} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Fall%':<8}")
    print("-" * 95)

    for rank, result in enumerate(all_results, 1):
        config_name = result['config']['name'][:43]
        f1 = result['metrics']['f1']
        prec = result['metrics']['precision']
        rec = result['metrics']['recall']
        fall_pct = result['fallback_rate'] * 100

        star = "⭐" if f1 >= 0.6 else ""
        print(f"{rank:<6} {config_name:<45} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {fall_pct:<7.1f}% {star}")

    # Mejor configuración
    if all_results:
        best = all_results[0]
        print(f"\n{'='*80}")
        print(f"🏆 MEJOR CONFIGURACIÓN: {best['config']['name']}")
        print(f"{'='*80}")
        print(f"  F1-Score:   {best['metrics']['f1']:.4f}")
        print(f"  Precision:  {best['metrics']['precision']:.4f}")
        print(f"  Recall:     {best['metrics']['recall']:.4f}")
        print(f"  Fallbacks:  {best['fallback_count']} ({best['fallback_rate']*100:.1f}%)")
        print(f"\n  Parámetros:")
        print(f"    TOP_K = {best['config']['TOP_K']}")
        print(f"    THRESHOLD = {best['config']['THRESHOLD']}")
        print(f"    MAX_DISPLAY = {best['config']['MAX_DISPLAY']}")
        print(f"    Query enhancement = '{best['config']['query_clinical_suffix']}'")
        print(f"    LLM Validation = {'ON' if best['config'].get('use_llm_validation', False) else 'OFF'}")

        # Guardar resultados
        output_dir = BENCHMARK_DIR / "results" / "optimization"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"optimization_results_{timestamp}.json"

        # Preparar datos para JSON
        json_results = []
        for result in all_results:
            json_results.append({
                "config_name": result['config']['name'],
                "parameters": {
                    "TOP_K": result['config']['TOP_K'],
                    "THRESHOLD": result['config']['THRESHOLD'],
                    "MAX_DISPLAY": result['config']['MAX_DISPLAY'],
                    "query_enhancement": result['config']['query_clinical_suffix'],
                    "use_llm_validation": result['config'].get('use_llm_validation', False)
                },
                "metrics": {
                    "f1": result['metrics']['f1'],
                    "precision": result['metrics']['precision'],
                    "recall": result['metrics']['recall']
                },
                "fallback_rate": result['fallback_rate'],
                "execution_time": result['execution_time']
            })

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "configurations_tested": len(CONFIGURATIONS),
                "results": json_results,
                "best_config": json_results[0]
            }, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Resultados guardados en: {results_file}")

    print(f"\n{'='*80}")
    print("Optimización completada")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
