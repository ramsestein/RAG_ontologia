#!/usr/bin/env python3
"""
01_DFA.py - Extractor de entidades médicas basado en Aho-Corasick (DFA)

Este script carga la taxonomía médica y busca coincidencias en notas clínicas
usando el algoritmo Aho-Corasick para búsqueda eficiente de múltiples patrones.

Input:
    - data/processed/taxonomia.json: Taxonomía de términos médicos
    - test/samples/validation_test.json: Notas clínicas de validación

Output:
    - src/NER/output/stage1_candidates.json: Candidatos encontrados
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import ahocorasick


def load_json(filepath: str) -> Any:
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Guarda datos en formato JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_automaton(taxonomy: List[Dict]) -> ahocorasick.Automaton:
    """
    Construye el autómata Aho-Corasick con todos los términos de la taxonomía.
    
    Args:
        taxonomy: Lista de entidades con 'code', 'local_name' y 'aliases'
    
    Returns:
        Autómata configurado para búsqueda
    """
    automaton = ahocorasick.Automaton()
    
    for entity in taxonomy:
        code = entity.get('code', '')
        local_name = entity.get('local_name', '')
        aliases = entity.get('aliases', [])
        
        # Añadir todos los alias (incluyendo el local_name que suele estar en aliases)
        for alias in aliases:
            if alias:  # Evitar strings vacíos
                # Normalizar: minúsculas para búsqueda case-insensitive
                normalized_alias = alias.lower().strip()
                if normalized_alias:
                    # El valor asociado es (código, nombre local, alias original)
                    automaton.add_word(normalized_alias, (code, local_name, alias))
    
    automaton.make_automaton()
    return automaton


def extract_text_from_note(note: Dict) -> str:
    """
    Extrae todo el texto relevante de una nota clínica.
    
    Args:
        note: Diccionario con datos de la nota clínica
    
    Returns:
        Texto concatenado de todos los campos relevantes
    """
    clinical_data = note.get('clinical_data', {})
    
    text_parts = []
    
    # Extraer campos de clinical_data
    if 'history' in clinical_data:
        text_parts.append(clinical_data['history'])
    if 'findings' in clinical_data:
        text_parts.append(clinical_data['findings'])
    if 'impression' in clinical_data:
        text_parts.append(clinical_data['impression'])
    
    return '\n'.join(text_parts)


def find_candidates(automaton: ahocorasick.Automaton, text: str, note_id: str, 
                    original_id: str) -> List[Dict]:
    """
    Busca todas las coincidencias en el texto usando el autómata.
    
    Args:
        automaton: Autómata Aho-Corasick
        text: Texto donde buscar
        note_id: ID de la nota
        original_id: ID original del documento
    
    Returns:
        Lista de candidatos encontrados
    """
    candidates = []
    text_lower = text.lower()
    
    for end_index, (code, local_name, matched_term) in automaton.iter(text_lower):
        start_index = end_index - len(matched_term) + 1
        
        # Extraer el texto original (con capitalización original)
        matched_text_original = text[start_index:end_index + 1]
        
        # Extraer contexto (40 caracteres antes y después)
        context_start = max(0, start_index - 40)
        context_end = min(len(text), end_index + 41)
        context = text[context_start:context_end]
        
        candidate = {
            'id': original_id,
            'note_id': note_id,
            'code': code,
            'local_name': local_name,
            'matched_term': matched_term,
            'matched_text': matched_text_original,
            'start': start_index,
            'end': end_index + 1,
            'context': context
        }
        candidates.append(candidate)
    
    return candidates


def remove_duplicates(candidates: List[Dict]) -> List[Dict]:
    """
    Elimina candidatos duplicados (mismo término en la misma posición).
    """
    seen = set()
    unique_candidates = []
    
    for c in candidates:
        key = (c['id'], c['start'], c['end'], c['code'])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates


def main():
    """Función principal del matcher."""
    # Definir rutas
    base_path = Path(__file__).parent.parent.parent  # new_ofarres/
    taxonomy_path = base_path / 'data' / 'processed' / 'taxonomia.json'
    notes_path = base_path / 'test' / 'samples' / 'validation_test.json'
    output_path = base_path / 'src' / 'NER' / 'output' / 'stage1_candidates.json'
    
    print("=" * 60)
    print("PASO 1: MATCHER (Aho-Corasick DFA)")
    print("=" * 60)
    
    # Cargar taxonomía
    print(f"\n📂 Cargando taxonomía desde: {taxonomy_path}")
    taxonomy = load_json(taxonomy_path)
    print(f"   ✓ {len(taxonomy)} entidades cargadas")
    
    # Contar total de alias
    total_aliases = sum(len(e.get('aliases', [])) for e in taxonomy)
    print(f"   ✓ {total_aliases} alias totales")
    
    # Construir autómata
    print("\n🔧 Construyendo autómata Aho-Corasick...")
    automaton = build_automaton(taxonomy)
    print(f"   ✓ Autómata construido con {len(automaton)} patrones únicos")
    
    # Cargar notas de validación
    print(f"\n📂 Cargando notas desde: {notes_path}")
    notes = load_json(notes_path)
    print(f"   ✓ {len(notes)} notas cargadas")
    
    # Procesar cada nota
    print("\n🔍 Buscando candidatos en las notas...")
    all_candidates = []
    notes_with_matches = 0
    
    for note in notes:
        original_id = note.get('id', '')
        note_id = note.get('note_id', '')
        
        text = extract_text_from_note(note)
        candidates = find_candidates(automaton, text, note_id, original_id)
        
        if candidates:
            notes_with_matches += 1
        
        all_candidates.extend(candidates)
    
    # Eliminar duplicados
    unique_candidates = remove_duplicates(all_candidates)
    
    # Guardar resultados
    print(f"\n💾 Guardando resultados en: {output_path}")
    save_json(unique_candidates, output_path)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN PASO 1")
    print("=" * 60)
    print(f"📊 Total de notas procesadas: {len(notes)}")
    print(f"📊 Notas con coincidencias: {notes_with_matches}")
    print(f"📊 Total candidatos encontrados: {len(all_candidates)}")
    print(f"📊 Candidatos únicos (sin duplicados): {len(unique_candidates)}")
    print(f"✅ Resultados guardados en: {output_path}")
    
    # Mostrar estadísticas por entidad
    entity_counts = {}
    for c in unique_candidates:
        name = c['local_name']
        entity_counts[name] = entity_counts.get(name, 0) + 1
    
    print("\n📈 Top 10 entidades más frecuentes:")
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, count in sorted_entities:
        print(f"   • {name}: {count}")
    
    return unique_candidates


if __name__ == '__main__':
    main()
