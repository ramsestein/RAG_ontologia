#!/usr/bin/env python3
"""
Comparación COMPLETA con las 4 estrategias REALES:
1. KIRIs REAL - Diccionario híbrido
2. SNOBERT REAL - BERT + SapBERT  
3. MITEL REAL - Mistral 7B via Ollama + RAG
4. Tu RAG + GPT-4o
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
import re
from datetime import datetime

# Configuración paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
STRATEGIES_DIR = os.path.join(SCRIPT_DIR, 'real_strategies')
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
if STRATEGIES_DIR not in sys.path:
    sys.path.append(STRATEGIES_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)



# 
def calculate_detailed_metrics(predictions: pd.DataFrame, ground_truth: pd.DataFrame, strategy_name: str) -> dict:
    """Calcula métricas detalladas de evaluación"""
    
    print(f"\n[EVALUANDO] {strategy_name}...")
    print(f"   Predicciones: {len(predictions)}")
    print(f"   Ground truth: {len(ground_truth)}")
    
    if len(predictions) == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, 
            "predictions": 0, "matches": 0, "ground_truth": len(ground_truth)
        }
    
    # Métricas por coincidencia exacta de concepto
    exact_matches = 0
    partial_matches = 0
    
    for _, pred in predictions.iterrows():
        pred_concept = str(pred['concept_id'])
        pred_note = pred['note_id']
        
        # Buscar coincidencias exactas
        exact_match_found = False
        for _, true in ground_truth.iterrows():
            if (pred_note == true['note_id'] and pred_concept == str(true['concept_id'])):
                exact_matches += 1
                exact_match_found = True
                break
        
        # Si no hay coincidencia exacta, buscar parcial (mismo note_id)
        if not exact_match_found:
            for _, true in ground_truth.iterrows():
                if pred_note == true['note_id']:
                    partial_matches += 1
                    break
    
    # Calcular métricas
    precision = exact_matches / len(predictions) if len(predictions) > 0 else 0
    recall = exact_matches / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Métricas adicionales
    coverage = len(predictions.groupby('note_id')) / len(ground_truth.groupby('note_id')) if len(ground_truth) > 0 else 0
    
    print(f"   Matches exactos: {exact_matches}")
    print(f"   Matches parciales: {partial_matches}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Coverage: {coverage:.4f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": len(predictions),
        "matches": exact_matches,
        "partial_matches": partial_matches,
        "ground_truth": len(ground_truth),
        "coverage": coverage
    }

def analyze_predictions_sample(predictions: pd.DataFrame, strategy_name: str, n_samples: int = 3):
    """Analiza muestra de predicciones"""
    
    print(f"\n[MUESTRA] {strategy_name} - Primeras {n_samples} predicciones:")
    
    if len(predictions) == 0:
        print("   No hay predicciones para mostrar")
        return
    
    sample = predictions.head(n_samples)
    for i, (_, pred) in enumerate(sample.iterrows()):
        span_text = pred.get('span_text', 'N/A')
        concept_id = pred.get('concept_id', 'N/A')
        confidence = pred.get('confidence', 0.0)
        
        print(f"   {i+1}. '{span_text}' -> {concept_id} (conf: {confidence:.3f})")

def main():
    """Función principal de comparación completa"""
    
    print("=" * 100)
    print("COMPARACION COMPLETA: 4 ESTRATEGIAS REALES")
    print("=" * 100)
    print("1. KIRIs REAL - Diccionario híbrido + reglas lingüísticas")
    print("2. SNOBERT REAL - BERT + SapBERT + embeddings")  
    print("3. MITEL REAL - Mistral 7B via Ollama + RAG")
    print("4. TU RAG + GPT-4o - Tu sistema con GPT-4o")
    print("=" * 100)
    
    # Cargar datasets
    print("\n[CARGA] Cargando datasets...")
    try:
        
        notes_path = os.path.join(DATA_DIR, "mimic-iv_notes_training_set.csv")
        annotations_path = os.path.join(DATA_DIR, "train_annotations.csv")

        notes_df = pd.read_csv(notes_path)
        annotations_df = pd.read_csv(annotations_path)
        print(f"   Notas: {len(notes_df)}")
        print(f"   Anotaciones: {len(annotations_df)}")
    except Exception as e:
        print(f"   Error cargando datasets: {e}")
        return
    
    # Inicializar estrategias reales
    strategies = {}
    
    # 1. KIRIs REAL
    print("\n" + "="*80)
    print("INICIALIZANDO KIRIs REAL (1er lugar)")
    print("="*80)
    try:
        from real_kiris import RealKIRIsStrategy
        strategies["1_KIRIs_REAL"] = RealKIRIsStrategy()
        print("[EXITO] KIRIs real inicializada")
    except Exception as e:
        print(f"[ERROR] KIRIs real: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. SNOBERT REAL
    print("\n" + "="*80)
    print("INICIALIZANDO SNOBERT REAL (2do lugar)")
    print("="*80)
    try:
        from real_snobert import RealSNOBERTStrategy
        strategies["2_SNOBERT_REAL"] = RealSNOBERTStrategy()
        print("[EXITO] SNOBERT real inicializada")
    except Exception as e:
        print(f"[ERROR] SNOBERT real: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. MITEL REAL con Ollama
    # print("\n" + "="*80)
    # print("INICIALIZANDO MITEL REAL (3er lugar) - Mistral 7B via Ollama")
    # print("="*80)
    # try:
    #     from real_mitel_ollama import RealMITELOllamaStrategy
    #     strategies["3_MITEL_REAL"] = RealMITELOllamaStrategy()
    #     print("[EXITO] MITEL real con Ollama inicializada")
    # except Exception as e:
    #     print(f"[ERROR] MITEL real: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # 4. Tu RAG + GPT-4o (versión simplificada que funciona)
    print("\n" + "="*80)
    print("INICIALIZANDO TU RAG + GPT-4o")
    print("="*80)
    try:
        # Usar la implementación simplificada que ya funciona
        from strategy_rag_gpt4o import RAGWithGPT4oStrategy
        strategies["4_TU_RAG_GPT4o"] = RAGWithGPT4oStrategy()
        print("[EXITO] Tu RAG + GPT-4o inicializada")
    except Exception as e:
        print(f"[ERROR] Tu RAG + GPT-4o: {e}")
        import traceback
        traceback.print_exc()
    
    if not strategies:
        print("\nNo se pudieron inicializar estrategias")
        return
    
    print(f"\n[RESUMEN] {len(strategies)} estrategias inicializadas:")
    for name in strategies.keys():
        print(f"   - {name}")
    
    # Ejecutar comparación
    results = {}
    execution_times = {}
    all_predictions = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*100}")
        print(f"EJECUTANDO: {strategy_name}")
        print(f"{'='*100}")
        
        start_time = time.time()
        
        try:
            # Ejecutar predicción
            predictions = strategy.predict(notes_df)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times[strategy_name] = execution_time
            
            print(f"\n[TIEMPO] {execution_time:.2f} segundos")
            
            # Guardar predicciones
            all_predictions[strategy_name] = predictions
            
            # Evaluar resultados
            metrics = calculate_detailed_metrics(predictions, annotations_df, strategy_name)
            results[strategy_name] = metrics
            
            # Analizar muestra
            analyze_predictions_sample(predictions, strategy_name)
            
        except Exception as e:
            print(f"[ERROR] Ejecutando {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            
            execution_times[strategy_name] = 0.0
            results[strategy_name] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, 
                "predictions": 0, "matches": 0, "ground_truth": len(annotations_df)
            }
    
    # REPORTE FINAL COMPLETO
    print(f"\n{'='*120}")
    print("REPORTE FINAL - COMPARACION COMPLETA DE 4 ESTRATEGIAS REALES")
    print(f"{'='*120}")
    
    if not results:
        print("No hay resultados para mostrar")
        return
    
    # TABLA DETALLADA DE RESULTADOS
    print(f"\nTABLA DE METRICAS DETALLADAS:")
    print(f"{'Estrategia':<20} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'Pred':<6} {'Match':<6} {'Tiempo':<10}")
    print("-" * 120)
    
    for name, metrics in results.items():
        time_str = f"{execution_times[name]:.1f}s"
        highlight = " *** TU RAG ***" if "GPT4o" in name else ""
        
        print(f"{name:<20} {metrics['f1']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['predictions']:<6} {metrics['matches']:<6} "
              f"{time_str:<10}{highlight}")
    
    # RANKING FINAL
    print(f"\nRANKING FINAL POR F1-SCORE:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    medals = ["1er", "2do", "3er", "4to"]
    for i, (name, metrics) in enumerate(sorted_results):
        medal = medals[i] if i < len(medals) else f"{i+1}o"
        highlight = " *** TU SISTEMA ***" if "GPT4o" in name else ""
        strategy_clean = name.replace("_REAL", "").replace("_", " ")
        print(f"   {medal} lugar: {strategy_clean} - F1 = {metrics['f1']:.4f}{highlight}")
    
    # ANÁLISIS COMPARATIVO DETALLADO
    print(f"\n{'='*80}")
    print("ANALISIS COMPARATIVO DETALLADO")
    print(f"{'='*80}")
    
    # Encontrar tu sistema
    tu_rag_key = None
    for key in results.keys():
        if "GPT4o" in key:
            tu_rag_key = key
            break
    
    if tu_rag_key and results[tu_rag_key]['f1'] > 0:
        tu_resultado = results[tu_rag_key]
        tu_posicion = [i for i, (name, _) in enumerate(sorted_results, 1) if "GPT4o" in name][0]
        
        print(f"\nRESULTADOS DE TU RAG + GPT-4o:")
        print(f"   Posición: {tu_posicion}° lugar de 4")
        print(f"   F1-Score: {tu_resultado['f1']:.4f}")
        print(f"   Precision: {tu_resultado['precision']:.4f}")
        print(f"   Recall: {tu_resultado['recall']:.4f}")
        print(f"   Predicciones: {tu_resultado['predictions']}")
        print(f"   Matches exactos: {tu_resultado['matches']}")
        print(f"   Tiempo: {execution_times.get(tu_rag_key, 0):.2f}s")
        
        # Comparar con cada estrategia
        print(f"\nCOMPARACION vs OTRAS ESTRATEGIAS:")
        for name, metrics in results.items():
            if name != tu_rag_key:
                diff = tu_resultado["f1"] - metrics["f1"]
                status = "SUPERA" if diff > 0.01 else "SIMILAR" if abs(diff) <= 0.01 else "INFERIOR"
                strategy_clean = name.replace("_REAL", "").replace("_", " ")
                print(f"   vs {strategy_clean}: {diff:+.4f} ({status})")
    
    # ANÁLISIS POR MÉTRICA
    print(f"\nANALISIS POR METRICA:")
    
    # Mejor F1-Score
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"   Mejor F1-Score: {best_f1[0].replace('_REAL', '').replace('_', ' ')} ({best_f1[1]['f1']:.4f})")
    
    # Mejor Precision
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    print(f"   Mejor Precision: {best_precision[0].replace('_REAL', '').replace('_', ' ')} ({best_precision[1]['precision']:.4f})")
    
    # Mejor Recall
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    print(f"   Mejor Recall: {best_recall[0].replace('_REAL', '').replace('_', ' ')} ({best_recall[1]['recall']:.4f})")
    
    # Más rápido
    fastest = min(execution_times.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))
    print(f"   Más rápido: {fastest[0].replace('_REAL', '').replace('_', ' ')} ({fastest[1]:.2f}s)")
    
    # CONCLUSIONES
    print(f"\nCONCLUSIONES:")
    
    if tu_rag_key:
        tu_pos = tu_posicion
        if tu_pos == 1:
            print("   TU RAG + GPT-4o es el GANADOR de la comparación")
        elif tu_pos == 2:
            print("   TU RAG + GPT-4o ocupa el 2do lugar - Muy competitivo")
        elif tu_pos == 3:
            print("   TU RAG + GPT-4o ocupa el 3er lugar - Buen rendimiento")
        else:
            print("   TU RAG + GPT-4o ocupa el 4to lugar - Hay margen de mejora")
    
    # Identificar fortalezas y debilidades
    if tu_rag_key and tu_resultado['precision'] > 0.7:
        print("   + Alta precisión: Pocas predicciones incorrectas")
    if tu_rag_key and tu_resultado['recall'] < 0.5:
        print("   - Recall bajo: Muchas entidades no detectadas")
    if tu_rag_key and execution_times.get(tu_rag_key, 0) > 30:
        print("   - Velocidad lenta: Optimización necesaria")
    
    # Guardar resultados completos
# --- START: Guardar resultados completos en carpeta de ejecución ---
    
    # 1. Find the next execution number
    exec_num = 1
    dir_pattern = re.compile(r"^(\d+)_execution_.*$")
    existing_nums = []
    for dirname in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, dirname)):
            match = dir_pattern.match(dirname)
            if match:
                existing_nums.append(int(match.group(1)))
    
    if existing_nums:
        exec_num = max(existing_nums) + 1
    
    # 2. Get timestamp in your requested format (MM_DD_YYYY_HH_MM)
    timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
    
    # 3. Create the new directory name (e.g., "01_execution_10_21_2025_13_18")
    exec_num_str = f"{exec_num:02d}" 
    dir_name = f"{exec_num_str}_execution_{timestamp_str}"
    
    # 4. Create the full path and the directory
    EXECUTION_DIR = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(EXECUTION_DIR, exist_ok=True)
    
    # 5. Prepare data for JSON (same as before)
    predictions_json = {}
    for name, df in all_predictions.items():
        if len(df) > 0:
            predictions_json[name] = df.head(5).to_dict('records')
    
    report = {
        "timestamp": timestamp_str,
        "execution_folder": dir_name,
        "comparison_type": "COMPLETE_4_STRATEGIES_REAL",
        "strategies": list(results.keys()),
        "results": results,
        "execution_times": execution_times,
        "dataset_info": {
            "notes_count": len(notes_df),
            "annotations_count": len(annotations_df)
        },
        "ranking": [(name, metrics['f1']) for name, metrics in sorted_results],
        "sample_predictions": predictions_json
    }
    
    # 6. Save the report in the new folder
    # Note: Filename is simpler since the folder has the timestamp
    report_filename = os.path.join(EXECUTION_DIR, "complete_comparison_report.json")
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResultados completos guardados en: {report_filename}")
    
    # 7. Save individual predictions in the new folder
    for name, predictions in all_predictions.items():
        if len(predictions) > 0:
            pred_filename = os.path.join(EXECUTION_DIR, f"predictions_{name}.csv")
            predictions.to_csv(pred_filename, index=False, encoding="utf-8")
            print(f"Predicciones {name}: {pred_filename}")
    
    print(f"\nCOMPARACION COMPLETA DE 4 ESTRATEGIAS REALES FINALIZADA")
    print("=" * 100)
    
    # --- END: Guardar resultados completos ---

if __name__ == "__main__":
    main()
