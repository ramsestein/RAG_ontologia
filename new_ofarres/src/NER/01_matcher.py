#!/usr/bin/env python3
"""
01_matcher.py - Extractor de entidades médicas basado en Aho-Corasick (DFA)

VERSIÓN OPTIMIZADA con:
- Normalización de acentos (español/catalán)
- Abreviaturas médicas comunes
- Expansión de términos multi-palabra
- Matching de lateralidad (D/I, dcho/izdo)

Input:
    - data/processed/taxonomia.json: Taxonomía de términos médicos
    - test/samples/validation_test.json: Notas clínicas de validación

Output:
    - src/NER/output/stage1_candidates.json: Candidatos encontrados
"""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import ahocorasick


# ============================================================================
# MAPEO DE ABREVIATURAS MÉDICAS COMUNES
# ============================================================================
ABBREVIATION_MAP = {
    # Arterias
    'ica': 'arteria carótida interna',
    'aci': 'arteria carótida interna',
    'ticas': 'arteria carótida interna',
    'tica': 'arteria carótida interna',
    'acm': 'arteria cerebral media',
    'acmd': 'arteria cerebral media derecha',
    'acmi': 'arteria cerebral media izquierda',
    'aca': 'arteria cerebral anterior',
    'acas': 'arteria cerebral anterior',
    'acp': 'arteria cerebral posterior',
    'acps': 'arteria cerebral posterior',
    'pica': 'arteria cerebelosa posteroinferior',
    'aica': 'arteria cerebelosa anteroinferior',
    'sca': 'arteria cerebelosa superior',
    # Segmentos
    'm1': 'segmento m1',
    'm2': 'segmento m2',
    'm3': 'segmento m3',
    'm4': 'segmento m4',
    'm5': 'segmento m5',
    'm6': 'segmento m6',
    # Lateralidad
    'd': 'derecho',
    'i': 'izquierdo',
    'dcha': 'derecho',
    'dcho': 'derecho',
    'izda': 'izquierdo',
    'izdo': 'izquierdo',
    'izqda': 'izquierdo',
    'der': 'derecho',
    'izq': 'izquierdo',
    # Otros
    'hsa': 'hemorragia subaracnoidea',
    'hsae': 'hemorragia subaracnoidea',
    'tsa': 'troncos supraórticos',
    'vl': 'ventrículo lateral',
    'cbf': 'flujo sanguíneo cerebral',
    'cbv': 'volumen sanguíneo cerebral',
    'ttp': 'tiempo al pico',
    'tmax': 'tiempo máximo',
}

# Patrones para detectar lateralidad contextual
LATERALITY_PATTERNS = [
    (r'\b(\w+)\s+derech[oa]s?\b', 'derecho'),
    (r'\b(\w+)\s+izquierd[oa]s?\b', 'izquierdo'),
    (r'\bderech[oa]s?\s+(\w+)\b', 'derecho'),
    (r'\bizquierd[oa]s?\s+(\w+)\b', 'izquierdo'),
]

# Términos adicionales a buscar (expansión de taxonomía)
EXTRA_TERMS = {
    # Lesiones isquémicas
    'lesión isquémica aguda': '16218291000119100',
    'lesion isquemica aguda': '16218291000119100',
    'isquemia aguda': '16218291000119100',
    'isquemia establecida': '16218291000119100',
    'lesión isquémica antigua': '111298007',
    'lesion isquemica antigua': '111298007',
    'lesión isquémica crónica': '111298007',
    'lesion isquemica cronica': '111298007',
    'isquemia crónica': '111298007',
    'isquemia cronica': '111298007',
    'infarto crónico': '111298007',
    'infarto cronico': '111298007',
    'infartos crónicos': '111298007',
    'infartos cronicos': '111298007',
    'infartos lacunares': '111298007',
    'infarto lacunar': '111298007',
    # Hemorragias
    'hemorragia subaracnoidea': '1386000',
    'hsa': '1386000',
    'sangrado agudo': '1386000',
    'contenido hemático': '1386000',
    'contenido hematico': '1386000',
    'hematoma': '1386000',
    # Oclusiones
    'oclusión total': '2929001',
    'oclusion total': '2929001',
    'oclusión completa': '2929001',
    'oclusion completa': '2929001',
    'oclusión trombótica': '2929001',
    'oclusion trombotica': '2929001',
    'oclusión focal': '2929001',
    'oclusion focal': '2929001',
    'oclusión arterial': '2929001',
    'oclusion arterial': '2929001',
    # Oclusión intracraneal específica
    'oclusión intracraneal': 'of5',
    'oclusion intracraneal': 'of5',
    # Estenosis
    'estenosis de la arteria carótida': '233964008',
    'estenosis de la arteria carotida': '233964008',
    'estenosis carotídea': '233964008',
    'estenosis carotidea': '233964008',
    'estenosis de la ica': '233964008',
    'estenosis ica': '233964008',
    'estenosis de arteria vertebral': '90520006',
    'estenosis vertebral': '90520006',
    # Territorios
    'territorio carotídeo': 'of20',
    'territorio carotideo': 'of20',
    'territorio de acm': 'of21',
    'territorio acm': 'of21',
    'territorio de aca': 'of22',
    'territorio aca': 'of22',
    # Perfusión
    'retraso en los mapas': 'RID4977',
    'retraso en mapas': 'RID4977',
    'retraso de tiempo': 'RID4977',
    'hipoperfusión': 'RID4977',
    'hipoperfusion': 'RID4977',
    'area de hipoperfusion': 'RID4977',
    'área de hipoperfusión': 'RID4977',
    # Trombo
    'trombo flotante': '396339007',
    'trombo hiperdenso': 'of3',
    'hiperdensidad basal': 'of3',
    # Discordancia/mismatch
    'mismatch': 'of24',
    'discordancia': 'of24',
    'discordancia significativa': 'of24',
    # Cerebelo
    'cerebeloso': '113305005',
    'cerebelosa': '113305005',
    'cerebelosas': '113305005',
    # Estructuras específicas
    'cabeza de caudado': '11000004',
    'núcleo caudado': '11000004',
    'nucleo caudado': '11000004',
    'n. lenticular': '41648007',
    'núcleo lenticular': '41648007',
    'nucleo lenticular': '41648007',
    # Arterias específicas
    'acm derecha': '369352006',
    'acmd': '369352006',
    'acm izquierda': '369353001',
    'acmi': '369353001',
    'aca izquierda': '369299002',
    'aca derecha': '369298005',
    # Segmentos
    'seg. m1': 'RID666',
    'segmento m1': 'RID666',
    'seg m1': 'RID666',
    'seg. m2': 'RID671',
    'segmento m2': 'RID671', 
    'seg m2': 'RID671',
    # Infarto cerebral
    'infarto cerebral': '432504007',
    'infarto': '432504007',
    'infartos': '432504007',
    # Oclusión intracraneal (of5) - más variantes
    'oclusión de': 'of5',
    'oclusion de': 'of5',
    'oclusión proximal': 'of5',
    'oclusion proximal': 'of5',
    'oclusión distal': 'of5',
    'oclusion distal': 'of5',
    'ocluida': 'of5',
    'ocluido': 'of5',
    'trombo': 'of5',
    'trombosis aguda': 'of5',
    'trombosis': 'of5',
    # Territorio ACM (of21) - más variantes
    'territorio de la acm': 'of21',
    'territorio de la arteria cerebral media': 'of21',
    'territorio m1': 'of21',
    'territorio m2': 'of21',
    # Lesiones crónicas
    'leucoaraiosis': '111298007',
    'lesiones crónicas': '111298007',
    'lesiones cronicas': '111298007',
    'cambios crónicos': '111298007',
    'cambios cronicos': '111298007',
    'gliosis': '111298007',
    # Estenosis vertebral
    'estenosis vertebrales': '90520006',
    # Ribete insular
    'ribete insular': '314190005',
    'surco insular': '314190005',
    'ínsula': '314190005',
    'insula': '314190005',
    # Segmento M2 específico
    'm2': 'RID671',
    'm1': 'RID666',
}


def normalize_text(text: str) -> str:
    """
    Normaliza el texto eliminando acentos y convirtiendo a minúsculas.
    """
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar acentos
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text


def load_json(filepath: str) -> Any:
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Guarda datos en formato JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_automaton(taxonomy: List[Dict]) -> Tuple[ahocorasick.Automaton, Dict[str, Tuple]]:
    """
    Construye el autómata Aho-Corasick con todos los términos de la taxonomía.
    Incluye normalización y términos expandidos.
    """
    automaton = ahocorasick.Automaton()
    term_to_info = {}  # Para mapeo inverso
    
    # Procesar taxonomía original
    for entity in taxonomy:
        code = entity.get('code', '')
        local_name = entity.get('local_name', '')
        aliases = entity.get('aliases', [])
        
        for alias in aliases:
            if alias:
                # Versión original normalizada
                normalized = normalize_text(alias.strip())
                if normalized and len(normalized) >= 2:
                    automaton.add_word(normalized, (code, local_name, alias))
                    term_to_info[normalized] = (code, local_name, alias)
    
    # Añadir términos extra
    for term, code in EXTRA_TERMS.items():
        normalized = normalize_text(term)
        if normalized not in term_to_info:
            automaton.add_word(normalized, (code, term, term))
            term_to_info[normalized] = (code, term, term)
    
    # Añadir abreviaturas que mapean a códigos conocidos
    abbrev_to_code = {
        'ica': ('86117002', 'arteria carótida interna', 'ICA'),
        'aci': ('86117002', 'arteria carótida interna', 'ACI'),
        'tica': ('86117002', 'arteria carótida interna', 'TICA'),
        'ticas': ('86117002', 'arteria carótida interna', 'TICAS'),
        'acm': ('17232002', 'arteria cerebral media', 'ACM'),
        'acmd': ('369352006', 'arteria cerebral media derecha', 'ACMd'),
        'acmi': ('369353001', 'arteria cerebral media izquierda', 'ACMi'),
        'aca': ('60176003', 'arteria cerebral anterior', 'ACA'),
        'acas': ('60176003', 'arteria cerebral anterior', 'ACAs'),
        'acp': ('70382005', 'arteria cerebral posterior', 'ACP'),
        'acps': ('70382005', 'arteria cerebral posterior', 'ACPs'),
        'pica': ('45242005', 'arteria cerebelosa posteroinferior', 'PICA'),
    }
    
    for abbrev, (code, name, original) in abbrev_to_code.items():
        if abbrev not in term_to_info:
            automaton.add_word(abbrev, (code, name, original))
            term_to_info[abbrev] = (code, name, original)
    
    automaton.make_automaton()
    return automaton, term_to_info


def extract_text_from_note(note: Dict) -> str:
    """Extrae todo el texto relevante de una nota clínica."""
    clinical_data = note.get('clinical_data', {})
    text_parts = []
    
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
    Usa texto normalizado para búsqueda pero preserva posiciones originales.
    """
    candidates = []
    
    # Normalizar texto para búsqueda
    text_normalized = normalize_text(text)
    
    # Crear mapeo de posiciones (normalizado -> original)
    # Esto es aproximado pero funciona para la mayoría de casos
    
    for end_index, (code, local_name, matched_term) in automaton.iter(text_normalized):
        start_index = end_index - len(matched_term) + 1
        
        # Verificar que no sea un match parcial de palabra
        # (debe estar rodeado por espacios, puntuación o inicio/fin)
        if start_index > 0 and text_normalized[start_index - 1].isalnum():
            continue
        if end_index < len(text_normalized) - 1 and text_normalized[end_index + 1].isalnum():
            continue
        
        # Extraer el texto original (aproximado a las mismas posiciones)
        matched_text_original = text[start_index:end_index + 1]
        
        # Extraer contexto (50 caracteres antes y después)
        context_start = max(0, start_index - 50)
        context_end = min(len(text), end_index + 51)
        context = text[context_start:context_end]
        
        candidate = {
            'id': original_id,
            'note_id': note_id,
            'code': code,
            'local_name': local_name,
            'matched_term': matched_term,
            'matched_text': matched_text_original.strip(),
            'start': start_index,
            'end': end_index + 1,
            'context': context
        }
        candidates.append(candidate)
    
    return candidates


def find_laterality_candidates(text: str, note_id: str, original_id: str) -> List[Dict]:
    """
    Busca específicamente términos de lateralidad (derecho/izquierdo).
    """
    candidates = []
    text_lower = text.lower()
    
    # Patrones para lateralidad
    patterns = [
        (r'\bderech[oa]s?\b', '24028007', 'derecho'),
        (r'\bizquierd[oa]s?\b', '7771000', 'izquierdo'),
        (r'\b[di]\b(?=\s*[>)])', None, None),  # D o I solos antes de > o )
    ]
    
    for pattern, code, name in patterns:
        if code is None:
            continue
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            start = match.start()
            end = match.end()
            
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            
            candidates.append({
                'id': original_id,
                'note_id': note_id,
                'code': code,
                'local_name': name,
                'matched_term': name,
                'matched_text': text[start:end],
                'start': start,
                'end': end,
                'context': text[context_start:context_end]
            })
    
    return candidates


def remove_duplicates(candidates: List[Dict]) -> List[Dict]:
    """Elimina candidatos duplicados (mismo código en posiciones muy cercanas)."""
    seen = set()
    unique_candidates = []
    
    # Ordenar por posición
    candidates_sorted = sorted(candidates, key=lambda x: (x['id'], x['start']))
    
    for c in candidates_sorted:
        # Crear clave basada en código y posición aproximada (ventana de 10 chars)
        key = (c['id'], c['code'], c['start'] // 10)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates


def filter_short_generic_terms(candidates: List[Dict]) -> List[Dict]:
    """
    Filtra términos genéricos que generan muchos FP.
    Aplica filtros contextuales más inteligentes.
    """
    # Códigos muy genéricos que solo deben contarse si hay patología
    generic_anatomical = {
        '60176003',   # arteria cerebral anterior
        '70382005',   # arteria cerebral posterior
        '17232002',   # arteria cerebral media
        '103421006',  # grado
        '303231004',  # intracraneal (muy genérico)
        '113305005',  # cerebeloso/a
    }
    
    # Códigos de lateralidad que solo valen con contexto específico
    laterality_codes = {
        '24028007',   # derecho
        '7771000',    # izquierdo
    }
    
    # Palabras que indican patología activa (justifican el hallazgo)
    pathology_keywords = [
        'estenosis', 'oclusión', 'oclusion', 'oclusivo', 'tromb', 
        'isquem', 'infarto', 'hemorrag', 'hipoperfus', 'hiperperfus',
        'defecto', 'lesión', 'lesion', 'aguda', 'subaguda', 'crónica', 'cronica',
        'disección', 'diseccion', 'aneurisma', 'malformación', 'malformacion',
        'ocluida', 'ocluido', 'obliteración', 'obliteracion'
    ]
    
    # Palabras que indican normalidad (NO justifican el hallazgo)
    normality_keywords = [
        'permeable', 'permeables', 'sin estenosis', 'sin signos', 
        'sin ateromatosis', 'conservad', 'normal', 'adecuad',
        'ya conocid', 'emboliza', 'material de embolización'
    ]
    
    # Palabras que justifican lateralidad (estructuras/hallazgos válidos)
    laterality_context_keywords = [
        'estenosis', 'oclusión', 'oclusion', 'isquem', 'infarto', 'lesión', 'lesion',
        'hemorrag', 'hematoma', 'trombo', 'defecto', 'hipoperfus', 'disección',
        'aneurisma', 'malformación', 'obliter'
    ]
    
    filtered = []
    for c in candidates:
        code = c['code']
        context_lower = c.get('context', '').lower()
        
        # Para códigos de lateralidad, ser más estricto
        if code in laterality_codes:
            has_valid_context = any(kw in context_lower for kw in laterality_context_keywords)
            has_normality = any(kw in context_lower for kw in normality_keywords)
            # Solo incluir si hay patología Y no hay normalidad
            if not has_valid_context or has_normality:
                continue
        
        # Para códigos anatómicos genéricos, verificar si hay patología
        if code in generic_anatomical:
            # Verificar si hay keywords de patología
            has_pathology = any(kw in context_lower for kw in pathology_keywords)
            # Verificar si predomina normalidad
            has_normality = any(kw in context_lower for kw in normality_keywords)
            
            # Solo incluir si hay patología y no predomina normalidad
            if not has_pathology or has_normality:
                continue
        
        filtered.append(c)
    
    return filtered


def main():
    """Función principal del matcher."""
    base_path = Path(__file__).parent.parent.parent
    taxonomy_path = base_path / 'data' / 'processed' / 'taxonomia.json'
    notes_path = base_path / 'test' / 'samples' / 'validation_test.json'
    output_path = base_path / 'src' / 'NER' / 'output' / 'stage1_candidates.json'
    
    print("=" * 60)
    print("PASO 1: MATCHER OPTIMIZADO (Aho-Corasick + Normalización)")
    print("=" * 60)
    
    # Cargar taxonomía
    print(f"\n📂 Cargando taxonomía desde: {taxonomy_path}")
    taxonomy = load_json(taxonomy_path)
    print(f"   ✓ {len(taxonomy)} entidades base cargadas")
    
    # Construir autómata con expansión
    print("\n🔧 Construyendo autómata expandido...")
    automaton, term_info = build_automaton(taxonomy)
    print(f"   ✓ Autómata construido con {len(automaton)} patrones")
    print(f"   ✓ +{len(EXTRA_TERMS)} términos extra añadidos")
    
    # Cargar notas
    print(f"\n📂 Cargando notas desde: {notes_path}")
    notes = load_json(notes_path)
    print(f"   ✓ {len(notes)} notas cargadas")
    
    # Procesar cada nota
    print("\n🔍 Buscando candidatos...")
    all_candidates = []
    
    for note in notes:
        original_id = note.get('id', '')
        note_id = note.get('note_id', '')
        text = extract_text_from_note(note)
        
        # Búsqueda principal con autómata
        candidates = find_candidates(automaton, text, note_id, original_id)
        all_candidates.extend(candidates)
        
        # Búsqueda adicional de lateralidad
        lat_candidates = find_laterality_candidates(text, note_id, original_id)
        all_candidates.extend(lat_candidates)
    
    # Post-procesamiento
    print("\n🔧 Post-procesando candidatos...")
    unique_candidates = remove_duplicates(all_candidates)
    filtered_candidates = filter_short_generic_terms(unique_candidates)
    
    # Guardar resultados
    print(f"\n💾 Guardando en: {output_path}")
    save_json(filtered_candidates, output_path)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN PASO 1 (OPTIMIZADO)")
    print("=" * 60)
    print(f"📊 Notas procesadas: {len(notes)}")
    print(f"📊 Candidatos brutos: {len(all_candidates)}")
    print(f"📊 Candidatos únicos: {len(unique_candidates)}")
    print(f"📊 Candidatos filtrados (final): {len(filtered_candidates)}")
    
    # Estadísticas por código
    code_counts = {}
    for c in filtered_candidates:
        code = c['code']
        code_counts[code] = code_counts.get(code, 0) + 1
    
    print(f"\n📈 Top 15 códigos encontrados:")
    for code, count in sorted(code_counts.items(), key=lambda x: -x[1])[:15]:
        name = next((c['local_name'] for c in filtered_candidates if c['code'] == code), code)
        print(f"   • {code} ({name}): {count}")
    
    return filtered_candidates


if __name__ == '__main__':
    main()
