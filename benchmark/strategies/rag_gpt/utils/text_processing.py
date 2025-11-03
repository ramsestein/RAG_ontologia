"""
Utilidades para procesamiento de texto y span matching
"""

import re
from typing import Tuple, Optional, List


def _make_flexible_regex(span_text: str) -> str:
    """
    Construye un patrón regex flexible que ignore múltiples espacios/saltos de línea entre palabras,
    escapando caracteres especiales del span.
    """
    palabras = re.split(r'\s+', span_text.strip())
    palabras_escapadas = [re.escape(palabra) for palabra in palabras if palabra]
    if not palabras_escapadas:
        return r''
    return r'\s+'.join(palabras_escapadas)


def find_span_in_text(span_text: str, text: str, start_from: int = 0) -> Optional[Tuple[int, int]]:
    """
    Encuentra la primera posición de un span en el texto usando regex flexible (case-insensitive).

    Args:
        span_text: Texto del span a buscar
        text: Texto completo donde buscar
        start_from: Índice desde donde comenzar la búsqueda

    Returns:
        Tupla (start, end) si se encuentra, None si no
    """
    pattern = _make_flexible_regex(span_text)
    if not pattern:
        return None

    match = re.search(pattern, text[start_from:], re.IGNORECASE)
    if match:
        start = match.start() + start_from
        end = match.end() + start_from
        return (start, end)
    return None


def find_all_spans_in_text(span_text: str, text: str) -> List[Tuple[int, int]]:
    """
    Encuentra TODAS las ocurrencias de un span en el texto con regex flexible (case-insensitive).

    Args:
        span_text: Texto del span a buscar
        text: Texto completo

    Returns:
        Lista de tuplas (start, end) para cada coincidencia encontrada
    """
    pattern = _make_flexible_regex(span_text)
    if not pattern:
        return []

    matches = []
    for m in re.finditer(pattern, text, re.IGNORECASE):
        matches.append((m.start(), m.end()))
    return matches


def clean_json_response(response: str) -> str:
    """
    Limpia una respuesta JSON de markdown y trailing commas

    Args:
        response: Respuesta cruda del LLM

    Returns:
        JSON limpio como string
    """
    response_clean = response.strip()

    # Remover markdown
    if '```json' in response_clean:
        json_start = response_clean.find('```json') + 7
        json_end = response_clean.find('```', json_start)
        response_clean = response_clean[json_start:json_end].strip()
    elif '```' in response_clean:
        json_start = response_clean.find('```') + 3
        json_end = response_clean.find('```', json_start)
        response_clean = response_clean[json_start:json_end].strip()

    # Limpiar trailing commas
    response_clean = re.sub(r',(\s*[}\]])', r'\1', response_clean)

    return response_clean


# =============================
# NUEVAS UTILIDADES (EXACT MATCH)
# =============================

def find_exact_span(span_text: str, text: str) -> Optional[Tuple[int, int]]:
    """
    Búsqueda EXACTA (case-sensitive) de un span en todo el texto.
    Devuelve solo la PRIMERA ocurrencia exacta.
    """
    if not span_text:
        return None
    idx = text.find(span_text)
    if idx == -1:
        return None
    return (idx, idx + len(span_text))


def find_first_case_insensitive(span_text: str, text: str) -> Optional[Tuple[int, int]]:
    """
    Búsqueda global de la PRIMERA ocurrencia en modo case-insensitive.
    Devuelve offsets en el texto original.
    """
    if not span_text:
        return None
    idx = text.lower().find(span_text.lower())
    if idx == -1:
        return None
    return (idx, idx + len(span_text))


def find_exact_span_near(span_text: str, text: str, approx_start: int, window: int = 50) -> Optional[Tuple[int, int]]:
    """
    Busca un span EXACTO alrededor de un offset aproximado (ventana local).
    Primero intenta case-sensitive, luego case-insensitive. Devuelve UNA coincidencia.
    """
    if not span_text:
        return None
    start = max(0, approx_start - window)
    end = min(len(text), approx_start + window + len(span_text))

    # Case-sensitive primero
    idx = text.find(span_text, start, end)
    if idx != -1:
        return (idx, idx + len(span_text))

    # Case-insensitive como último recurso local
    lower_text = text.lower()
    lower_span = span_text.lower()
    idx2 = lower_text.find(lower_span, start, end)
    if idx2 != -1:
        return (idx2, idx2 + len(span_text))

    return None
