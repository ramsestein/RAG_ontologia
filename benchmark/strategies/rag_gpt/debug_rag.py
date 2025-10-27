#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script para RAG+GPT Strategy
Prueba el pipeline con una sola nota para debugging rápido
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Configurar path
SCRIPT_DIR = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(BENCHMARK_DIR))

from strategies.rag_gpt.pipeline import RAGGPTPipeline


def main():
    print("="*80)
    print("DEBUG MODE - RAG+GPT Strategy")
    print("="*80)
    
    # Cargar datos
    data_dir = BENCHMARK_DIR / "data"
    notes_df = pd.read_csv(data_dir / "mimic-iv_notes_training_set.csv")
    annotations_df = pd.read_csv(data_dir / "train_annotations.csv")
    
    # Seleccionar primera nota
    test_note = notes_df.iloc[0]
    note_id = test_note['note_id']
    text = test_note['text']
    
    print(f"\n[DEBUG] Nota de prueba: {note_id}")
    print(f"Texto (primeros 300 chars):\n{text[:300]}...")
    
    # Ground truth para esta nota
    gt_for_note = annotations_df[annotations_df['note_id'] == note_id]
    print(f"\n[DEBUG] Ground Truth: {len(gt_for_note)} anotaciones")
    for _, row in gt_for_note.iterrows():
        print(f"  - {row['concept_id']}: '{row['span_text']}'")
    
    # Inicializar pipeline
    print(f"\n[DEBUG] Inicializando pipeline...")
    pipeline = RAGGPTPipeline(verbose=True)
    
    # Ejecutar predicción
    print(f"\n[DEBUG] Ejecutando predicción...")
    entities = pipeline.process_note(text, note_id)
    
    print(f"\n[DEBUG] Entidades detectadas: {len(entities)}")
    
    # Mostrar resultados detallados
    print("\n" + "="*80)
    print("[DEBUG] PREDICCIONES DETALLADAS:")
    print("-"*80)
    
    for i, entity in enumerate(entities, 1):
        print(f"\nPrediccion {i}:")
        print(f"  Span: '{entity.get('span_text_real', entity['span_text'])}'")
        print(f"  Concept ID: {entity['entity_code']}")
        print(f"  Start: {entity['start']}, End: {entity['end']}")
        print(f"  Location: {entity['anatomical_location']}")
        print(f"  Presence: {entity['presence']}")
        
        # Verificar match con ground truth
        matches = gt_for_note[
            (gt_for_note['concept_id'] == entity['entity_code']) &
            (gt_for_note['start'] == entity['start']) &
            (gt_for_note['end'] == entity['end'])
        ]
        
        if len(matches) > 0:
            print(f"  [[OK]] MATCH EXACTO con ground truth!")
        else:
            # Verificar match parcial (mismo código)
            code_matches = gt_for_note[gt_for_note['concept_id'] == entity['entity_code']]
            if len(code_matches) > 0:
                print(f"  [[WARNING]] Match de código, pero diferente posición")
            else:
                print(f"  [[ERROR]] NO match con ground truth")
    
    # Resumen
    print("\n" + "="*80)
    print("[DEBUG] RESUMEN:")
    print(f"  Ground Truth: {len(gt_for_note)} entidades")
    print(f"  Predicciones: {len(entities)} entidades")
    
    # Calcular matches
    exact_matches = 0
    for entity in entities:
        matches = gt_for_note[
            (gt_for_note['concept_id'] == entity['entity_code']) &
            (gt_for_note['start'] == entity['start']) &
            (gt_for_note['end'] == entity['end'])
        ]
        if len(matches) > 0:
            exact_matches += 1
    
    print(f"  Exact Matches: {exact_matches}")
    
    if len(entities) > 0:
        precision = exact_matches / len(entities) * 100
        print(f"  Precision: {precision:.2f}%")
    
    if len(gt_for_note) > 0:
        recall = exact_matches / len(gt_for_note) * 100
        print(f"  Recall: {recall:.2f}%")
    
    # Mostrar códigos usados
    print(f"\n[DEBUG] CÓDIGOS USADOS:")
    from collections import Counter
    code_counts = Counter([e['entity_code'] for e in entities])
    for code, count in code_counts.most_common():
        status = "[OK]" if code not in ['404684003', 'LINKING_FAILED'] else "[WARNING]"
        print(f"  {status} {code}: {count} veces")
    
    print("\n" + "="*80)
    print("[DEBUG] Debug completado!")
    print("="*80)


if __name__ == "__main__":
    main()
