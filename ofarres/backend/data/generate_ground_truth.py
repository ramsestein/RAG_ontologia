#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_ground_truth.py

OBJETIVO: Genera ground_truth.json enriquecido con el campo 'text' extraído
          usando los índices start/end de ground_truth_no_concept.json y el
          texto crudo de notes.json.

INPUT:
  - ground_truth_no_concept.json: Contiene note_id y annotations con start, end, concept_id
  - notes.json: Contiene note_id y el texto crudo de cada nota

OUTPUT:
  - ground_truth.json: Mismo formato que ground_truth_no_concept.json pero cada
                       annotation incluye el campo "text" con el substring extraído.

USO:
  python data/generate_ground_truth.py
"""

import json
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
GT_NO_CONCEPT_PATH = SCRIPT_DIR / "ground_truth_no_concept.json"
NOTES_PATH = SCRIPT_DIR / "notes.json"
OUTPUT_PATH = SCRIPT_DIR / "ground_truth.json"


def load_json(path: Path) -> list:
    """Carga un archivo JSON y retorna su contenido."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: list, path: Path) -> None:
    """Guarda datos en formato JSON con indentación."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Guardado: {path}")


def generate_ground_truth():
    """
    Proceso principal:
    1. Cargar ground_truth_no_concept.json
    2. Cargar notes.json
    3. Para cada nota, extraer el texto de cada anotación usando start/end
    4. Guardar en ground_truth.json
    """
    print("[Cargando] ground_truth_no_concept.json...")
    gt_no_concept = load_json(GT_NO_CONCEPT_PATH)
    
    print("[Cargando] notes.json...")
    notes_raw = load_json(NOTES_PATH)
    
    # Crear diccionario de notas para búsqueda rápida por note_id
    notes_dict = {note['note_id']: note['text'] for note in notes_raw}
    
    # Procesar cada nota del ground truth
    enriched_gt = []
    total_annotations = 0
    annotations_with_text = 0
    
    for gt_note in gt_no_concept:
        note_id = gt_note['note_id']
        note_text = notes_dict.get(note_id)
        
        if note_text is None:
            print(f"[WARN] Nota {note_id} no encontrada en notes.json. Saltando...")
            continue
        
        enriched_annotations = []
        
        for ann in gt_note['annotations']:
            start = ann['start']
            end = ann['end']
            
            # Extraer el texto usando los índices
            extracted_text = note_text[start:end]
            
            # Crear la anotación enriquecida
            enriched_ann = {
                'start': start,
                'end': end,
                'concept_id': ann['concept_id'],
                'text': extracted_text
            }
            
            enriched_annotations.append(enriched_ann)
            total_annotations += 1
            annotations_with_text += 1
        
        enriched_gt.append({
            'note_id': note_id,
            'annotations': enriched_annotations
        })
    
    # Guardar resultado
    save_json(enriched_gt, OUTPUT_PATH)
    
    # Resumen
    print("\n" + "=" * 60)
    print(" RESUMEN DE GENERACIÓN")
    print("=" * 60)
    print(f"  Notas procesadas:        {len(enriched_gt)}")
    print(f"  Anotaciones totales:     {total_annotations}")
    print(f"  Anotaciones con texto:   {annotations_with_text}")
    print("=" * 60)
    
    # Mostrar ejemplos de las primeras anotaciones
    print("\n📋 EJEMPLOS (primeras 5 anotaciones de la primera nota):")
    if enriched_gt:
        first_note = enriched_gt[0]
        print(f"\n  Nota: {first_note['note_id']}")
        for i, ann in enumerate(first_note['annotations'][:5]):
            print(f"    [{i+1}] [{ann['start']}-{ann['end']}] \"{ann['text']}\"")


if __name__ == "__main__":
    generate_ground_truth()
