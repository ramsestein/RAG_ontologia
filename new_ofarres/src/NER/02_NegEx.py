#!/usr/bin/env python3
"""
02_negex.py - Filtro de negación OPTIMIZADO para entidades médicas

VERSIÓN OPTIMIZADA con:
- Ventana basada en tokens (palabras) en lugar de caracteres
- Reset de negación en puntos y comas
- Patrones de normalidad radiológica
- Mejor detección contextual

Input:
    - src/NER/output/stage1_candidates.json: Candidatos del paso 1

Output:
    - src/NER/output/stage2_filtered.json: Candidatos con flag de negación
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set


# ============================================================================
# CONFIGURACIÓN DE NEGACIÓN
# ============================================================================

# Número de tokens (palabras) a revisar antes del hallazgo
NEGATION_WINDOW_TOKENS = 6

# Patrones de negación explícita (español y catalán)
NEGATION_TRIGGERS = [
    # Español - negación directa
    r'\bno\b',
    r'\bsin\b',
    r'\bnegativo\b',
    r'\bnegativa\b',
    r'\bnegativos\b',
    r'\bnegativas\b',
    r'\bdescartar\b',
    r'\bdescarta\b',
    r'\bdescartado\b',
    r'\bdescartada\b',
    r'\bse descarta\b',
    r'\bausencia\b',
    r'\bausente\b',
    r'\bniega\b',
    r'\bnega\b',
    r'\bno se observa\b',
    r'\bno se evidencia\b',
    r'\bno se identifica\b',
    r'\bno hay\b',
    r'\bno presenta\b',
    r'\bsin evidencia\b',
    r'\bsin signos\b',
    
    # Catalán
    r'\bsense\b',
    r"\bno s'observa\b",
    r"\bno s'evidencia\b",
    r'\bno hi ha\b',
    r'\babsència\b',
    r'\babsent\b',
    r'\bnegatiu\b',
    r'\bnegatius\b',
    r'\bdescartat\b',
]

# Patrones que indican NORMALIDAD (negación implícita de patología)
NORMALITY_PATTERNS = [
    r'\bpermeable[s]?\b',
    r'\bpermeables\b',
    r'\bsin ateromatosis\b',
    r'\bsin alteraciones\b',
    r'\bya conocid[oa]s?\b',
    r'\bdescrit[oa]s? en estudios previos\b',
    r'\bsin signos\b',
    r'\bconservad[oa]\b',
    r'\bnormal\b',
]

# Códigos de ARTERIAS que solo valen si hay patología (no solo mención)
ARTERY_CODES = {
    '86117002',   # arteria carótida interna
    '17232002',   # arteria cerebral media
    '60176003',   # arteria cerebral anterior
    '70382005',   # arteria cerebral posterior
    '369352006',  # arteria cerebral media derecha
    '369353001',  # arteria cerebral media izquierda
    '369298005',  # arteria cerebral anterior derecha
    '369299002',  # arteria cerebral anterior izquierda
    '369300005',  # arteria cerebral posterior derecha
    '369301009',  # arteria cerebral posterior izquierda
    'of6',        # carótida interna genérico
    'RID666',     # segmento m1
    'RID671',     # segmento m2
    'RID680',     # segmento m3
}

# Códigos que NO deben ser negados por patrones de normalidad
# (son hallazgos estructurales, no patológicos)
STRUCTURAL_CODES = {
    '24028007',   # derecho
    '7771000',    # izquierdo
    '303231004',  # intracraneal
    '303232006',  # extracraneal
    '1290003007', # ASPECTS (es una escala, no patología)
    '113305005',  # cerebelo
    '85637007',   # cápsula interna
    '11000004',   # caudado
    '41648007',   # lenticular
    '314190005',  # ribete insular
    '86117002',   # arteria carótida interna
    '17232002',   # arteria cerebral media
    '60176003',   # arteria cerebral anterior
    '70382005',   # arteria cerebral posterior
    '369352006',  # arteria cerebral media derecha
    '369353001',  # arteria cerebral media izquierda
    'RID666',     # segmento m1
    'RID671',     # segmento m2
    'RID4977',    # hipoperfusión
    'of21',       # territorio ACM
    'of20',       # territorio carótida
    '16218291000119100',  # signos lesión isquémica aguda
    '111298007',  # lesiones isquémicas crónicas
    'of3',        # trombo hiperdenso
    'of5',        # oclusión
    'of24',       # discordancia
}

# Códigos que SÍ pueden ser negados por normalidad (patologías)
PATHOLOGY_CODES = {
    '415582006',         # estenosis
    '1386000',           # hemorragia intracraneal
    '2929001',           # oclusión arterial
    '396339007',         # trombo
    'of5',               # oclusión arteria intracraneal
}

# Compilar patrones
NEGATION_REGEX = re.compile('|'.join(NEGATION_TRIGGERS), re.IGNORECASE)
NORMALITY_REGEX = re.compile('|'.join(NORMALITY_PATTERNS), re.IGNORECASE)


def load_json(filepath: str) -> Any:
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Guarda datos en formato JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_tokens_before(text: str, position: int, n_tokens: int) -> Tuple[str, int]:
    """
    Obtiene los n tokens (palabras) antes de una posición dada.
    Se detiene en puntos, dos puntos o saltos de línea.
    
    Returns:
        Tupla (texto_tokens, posición_inicio)
    """
    # Extraer texto antes de la posición
    text_before = text[:position]
    
    # Encontrar el último delimitador de oración
    delimiters = ['.', ':', '\n', ';']
    last_delimiter_pos = -1
    for delim in delimiters:
        pos = text_before.rfind(delim)
        if pos > last_delimiter_pos:
            last_delimiter_pos = pos
    
    # Extraer solo la parte desde el último delimitador
    if last_delimiter_pos >= 0:
        relevant_text = text_before[last_delimiter_pos + 1:]
        start_pos = last_delimiter_pos + 1
    else:
        relevant_text = text_before
        start_pos = 0
    
    # Tokenizar y tomar los últimos n tokens
    tokens = relevant_text.split()
    if len(tokens) > n_tokens:
        tokens = tokens[-n_tokens:]
    
    return ' '.join(tokens), start_pos


def check_negation(context: str, matched_text: str, code: str = '') -> Tuple[bool, str, str]:
    """
    Verifica negación usando ventana basada en tokens.
    
    Returns:
        Tupla (es_negado, tipo_negación, trigger)
    """
    # Encontrar posición del match en el contexto
    context_lower = context.lower()
    matched_lower = matched_text.lower()
    match_pos = context_lower.find(matched_lower)
    
    if match_pos == -1:
        match_pos = len(context) // 2  # Aproximación
    
    # Obtener tokens antes del hallazgo
    tokens_before, _ = get_tokens_before(context, match_pos, NEGATION_WINDOW_TOKENS)
    tokens_before_lower = tokens_before.lower()
    
    # 1. Buscar negación explícita
    negation_match = NEGATION_REGEX.search(tokens_before_lower)
    if negation_match:
        return True, 'explicit', negation_match.group()
    
    # 2. Buscar patrones de normalidad en TODO el contexto para arterias
    if code in ARTERY_CODES:
        normality_match = NORMALITY_REGEX.search(context_lower)
        if normality_match:
            trigger = normality_match.group()
            return True, 'normality_artery', trigger
    
    # 3. Buscar patrones de normalidad (solo para códigos de patología)
    if code not in STRUCTURAL_CODES and code in PATHOLOGY_CODES:
        normality_match = NORMALITY_REGEX.search(tokens_before_lower)
        if normality_match:
            trigger = normality_match.group()
            return True, 'normality', trigger
    
    return False, '', ''


def process_candidates(candidates: List[Dict]) -> List[Dict]:
    """
    Procesa los candidatos aplicando detección de negación optimizada.
    """
    processed = []
    
    for candidate in candidates:
        context = candidate.get('context', '')
        matched_text = candidate.get('matched_text', '')
        code = candidate.get('code', '')
        
        # Verificación principal basada en tokens
        is_negated, neg_type, trigger = check_negation(context, matched_text, code)
        
        # Crear candidato procesado
        processed_candidate = candidate.copy()
        processed_candidate['negated'] = is_negated
        processed_candidate['negation_type'] = neg_type if is_negated else None
        processed_candidate['negation_trigger'] = trigger if is_negated else None
        
        processed.append(processed_candidate)
    
    return processed


def print_comparison(candidates: List[Dict], num_notes: int = 2) -> None:
    """Imprime comparativa de resultados para varias notas."""
    notes = {}
    for c in candidates:
        note_id = c.get('note_id', 'unknown')
        if note_id not in notes:
            notes[note_id] = []
        notes[note_id].append(c)
    
    print("\n" + "=" * 80)
    print("COMPARATIVA DE RESULTADOS POR NOTA")
    print("=" * 80)
    
    note_ids = list(notes.keys())[:num_notes]
    
    for note_id in note_ids:
        note_candidates = notes[note_id]
        negated_count = sum(1 for c in note_candidates if c.get('negated', False))
        affirmed_count = len(note_candidates) - negated_count
        
        print(f"\n📋 NOTA ID: {note_id}")
        print(f"   Total hallazgos: {len(note_candidates)}")
        print(f"   ✅ Afirmados: {affirmed_count}")
        print(f"   ❌ Negados: {negated_count}")
        print("-" * 60)
        
        print("   Ejemplos AFIRMADOS:")
        affirmed = [c for c in note_candidates if not c.get('negated', False)][:3]
        for c in affirmed:
            print(f"      • [{c['code']}] {c['local_name']}: \"{c['matched_text']}\"")
        
        print("\n   Ejemplos NEGADOS:")
        negated = [c for c in note_candidates if c.get('negated', False)][:3]
        for c in negated:
            print(f"      • [{c['code']}] {c['local_name']}: \"{c['matched_text']}\"")
            print(f"        Trigger: \"{c['negation_trigger']}\" ({c.get('negation_type', 'unknown')})")


def main():
    """Función principal del filtro de negación."""
    # Definir rutas
    base_path = Path(__file__).parent.parent.parent  # new_ofarres/
    input_path = base_path / 'src' / 'NER' / 'output' / 'stage1_candidates.json'
    output_path = base_path / 'src' / 'NER' / 'output' / 'stage2_filtered.json'
    
    print("=" * 60)
    print("PASO 2: NEGEX OPTIMIZADO (Ventana por Tokens)")
    print("=" * 60)
    
    # Cargar candidatos del paso 1
    print(f"\n📂 Cargando candidatos desde: {input_path}")
    candidates = load_json(input_path)
    print(f"   ✓ {len(candidates)} candidatos cargados")
    
    # Mostrar configuración
    print(f"\n🔧 Configuración:")
    print(f"   • Ventana de negación: {NEGATION_WINDOW_TOKENS} tokens")
    print(f"   • Triggers de negación: {len(NEGATION_TRIGGERS)}")
    print(f"   • Patrones de normalidad: {len(NORMALITY_PATTERNS)}")
    
    # Procesar candidatos
    print("\n🔍 Aplicando detección de negación...")
    processed_candidates = process_candidates(candidates)
    
    # Calcular estadísticas
    total = len(processed_candidates)
    negated = sum(1 for c in processed_candidates if c.get('negated', False))
    affirmed = total - negated
    
    # Guardar resultados
    print(f"\n💾 Guardando resultados en: {output_path}")
    save_json(processed_candidates, output_path)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN PASO 2 (OPTIMIZADO)")
    print("=" * 60)
    print(f"📊 Total candidatos procesados: {total}")
    print(f"✅ Candidatos AFIRMADOS: {affirmed} ({affirmed/total*100:.1f}%)")
    print(f"❌ Candidatos NEGADOS: {negated} ({negated/total*100:.1f}%)")
    
    # Estadísticas de triggers de negación
    triggers = {}
    for c in processed_candidates:
        if c.get('negated') and c.get('negation_trigger'):
            trigger = c['negation_trigger'].lower()
            triggers[trigger] = triggers.get(trigger, 0) + 1
    
    if triggers:
        print("\n📈 Triggers de negación más frecuentes:")
        sorted_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:10]
        for trigger, count in sorted_triggers:
            print(f"   • \"{trigger}\": {count}")
    
    # Comparativa de notas
    print_comparison(processed_candidates, num_notes=2)
    
    return processed_candidates


if __name__ == '__main__':
    main()
