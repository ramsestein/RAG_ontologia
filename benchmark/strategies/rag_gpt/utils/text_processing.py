"""
Utilidades para procesamiento de texto y span matching
"""

import re
from typing import Tuple, Optional


def find_span_in_text(span_text: str, text: str, start_from: int = 0) -> Optional[Tuple[int, int]]:
    """
    Encuentra la posición de un span en el texto usando regex flexible
    
    Args:
        span_text: Texto del span a buscar
        text: Texto completo donde buscar
        start_from: Índice desde donde comenzar la búsqueda
        
    Returns:
        Tupla (start, end) si se encuentra, None si no
    """
    # Crear patrón regex flexible que ignore múltiples espacios/saltos de línea
    palabras = re.split(r'\s+', span_text)
    palabras_escapadas = [re.escape(palabra) for palabra in palabras if palabra]
    regex_pattern = r'\s+'.join(palabras_escapadas)
    
    # Buscar desde start_from
    match = re.search(regex_pattern, text[start_from:], re.IGNORECASE)
    
    if match:
        start = match.start() + start_from
        end = match.end() + start_from
        return (start, end)
    
    return None


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
