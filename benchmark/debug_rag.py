#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de debugging para RAG+GPT4o Strategy

Este script te permite probar la estrategia con UNA SOLA NOTA
para ver exactamente qué está pasando en cada paso.

Uso:
    python debug_rag.py
"""

import sys
import os
import pandas as pd

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGIES_DIR = os.path.join(SCRIPT_DIR, 'strategies')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, STRATEGIES_DIR)

# Import strategy - use importlib for numeric module names
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rag_gpt_module",
    os.path.join(STRATEGIES_DIR, "04_rag_gpt.py")
)
rag_gpt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_gpt_module)
RAGWithGPT4oStrategy = rag_gpt_module.RAGWithGPT4oStrategy


def main():
    print("="*80)
    print("DEBUG MODE - RAG+GPT4o Strategy")
    print("="*80)
    
    # Cargar datos
    data_dir = os.path.join(SCRIPT_DIR, 'data')
    notes_path = os.path.join(data_dir, 'mimic-iv_notes_training_set.csv')
    annotations_path = os.path.join(data_dir, 'train_annotations.csv')
    
    notes_df = pd.read_csv(notes_path)
    annotations_df = pd.read_csv(annotations_path)
    
    # Seleccionar UNA nota (la primera por defecto)
    test_note_id = notes_df.iloc[0]['note_id']
    test_note = notes_df[notes_df['note_id'] == test_note_id].iloc[0]
    
    print(f"\n[DEBUG] Nota de prueba: {test_note_id}")
    print(f"Texto (primeros 300 chars):")
    print(f"{test_note['text'][:300]}...")
    print()
    
    # Ground truth para esta nota
    gt = annotations_df[annotations_df['note_id'] == test_note_id]
    print(f"\n[DEBUG] Ground Truth: {len(gt)} anotaciones")
    for idx, row in gt.iterrows():
        print(f"  - {row['concept_id']}: '{row['span_text']}'")
    print()
    
    # Inicializar estrategia
    print("\n[DEBUG] Inicializando estrategia...")
    strategy = RAGWithGPT4oStrategy()
    
    # Ejecutar predicción
    print("\n[DEBUG] Ejecutando prediccion...")
    print("="*80)
    
    test_df = pd.DataFrame([test_note])
    predictions = strategy.predict(test_df)
    
    print("="*80)
    print(f"\n[DEBUG] Predicciones generadas: {len(predictions)}")
    
    # Mostrar predicciones
    print("\n[DEBUG] PREDICCIONES DETALLADAS:")
    print("-"*80)
    for idx, pred in predictions.iterrows():
        print(f"\nPrediccion {idx+1}:")
        print(f"  Span: '{pred['span_text']}'")
        print(f"  Concept ID: {pred['concept_id']}")
        print(f"  Start: {pred['start']}, End: {pred['end']}")
        print(f"  Entity Desc: {pred.get('entity_description', 'N/A')}")
        print(f"  Anatomy Code: {pred.get('anatomy_code', 'N/A')}")
        print(f"  Presence Code: {pred.get('presence_code', 'N/A')}")
        
        # Verificar si coincide con ground truth
        matches = gt[
            (gt['concept_id'] == pred['concept_id']) |
            (gt['span_text'].str.contains(pred['span_text'], case=False, na=False))
        ]
        
        if len(matches) > 0:
            print(f"  [OK] MATCH con ground truth!")
        else:
            print(f"  [ERROR] NO match con ground truth")
            # Buscar si el span coincide pero el código no
            span_matches = gt[gt['span_text'].str.contains(pred['span_text'], case=False, na=False)]
            if len(span_matches) > 0:
                print(f"     (Span correcto, pero codigo deberia ser: {span_matches.iloc[0]['concept_id']})")
    
    # Resumen
    print("\n" + "="*80)
    print("[DEBUG] RESUMEN:")
    print(f"  Ground Truth: {len(gt)} entidades")
    print(f"  Predicciones: {len(predictions)} entidades")
    
    # Calcular matches básicos
    exact_matches = 0
    for _, pred in predictions.iterrows():
        if len(gt[gt['concept_id'] == pred['concept_id']]) > 0:
            exact_matches += 1
    
    print(f"  Exact Matches: {exact_matches}")
    print(f"  Precision: {exact_matches / len(predictions) if len(predictions) > 0 else 0:.2%}")
    print(f"  Recall: {exact_matches / len(gt) if len(gt) > 0 else 0:.2%}")
    
    # Análisis de códigos usados
    print("\n[DEBUG] CODIGOS USADOS:")
    if len(predictions) > 0:
        code_counts = predictions['concept_id'].value_counts()
        for code, count in code_counts.items():
            if code in ['404684003', 'LINKING_FAILED']:
                print(f"  [WARNING] {code}: {count} veces (CODIGO GENERICO/FALLBACK)")
            else:
                print(f"  [OK] {code}: {count} veces")
    else:
        print("  (No hay predicciones para analizar)")
    
    print("\n" + "="*80)
    print("[DEBUG] Debug completado!")
    print("="*80)


if __name__ == "__main__":
    main()
