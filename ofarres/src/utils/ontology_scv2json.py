import pandas as pd
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS ---
# El script está en src/utils, subimos 2 niveles para llegar a la raíz
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Rutas relativas desde la raíz
RAW_CSV_PATH = PROJECT_ROOT / "ontology" / "hybrid_ontology.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "ontology" / "ontology.json"

def clean_term(text: str) -> str:
    """
    Limpia artefactos de SNOMED-CT para dejar el término natural.
    Ej: "Estómago [como un todo] (estructura corporal)" -> "Estómago"
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Eliminar etiquetas de contexto entre corchetes: [como un todo], [dup]
    text = re.sub(r'\s*\[.*?\]', '', text)
    
    # 2. Eliminar etiquetas semánticas entre paréntesis al final: (estructura corporal), (hallazgo)
    text = re.sub(r'\s*\([^)]*\)$', '', text)
    
    # 3. Eliminar prefijos redundantes "estructura de" (opcional, pero recomendado para NER)
    # Ej: "estructura de la pared abdominal" -> "pared abdominal"
    text = re.sub(r'^estructura (de |del |de la |de los |de las )?', '', text, flags=re.IGNORECASE)
    
    # 4. Limpieza final de espacios y puntuación
    return text.strip().strip('.,;:')

def extract_data_from_narrative(narrative: str) -> dict:
    """
    Parsea el texto libre de la columna narrativa usando Regex.
    """
    result = {
        "preferred": None,
        "synonyms": set(),
        "definition": None,
        "types": []
    }
    
    if not isinstance(narrative, str):
        return result

    # --- REGEX PATTERNS ---
    
    # 1. Término Preferido
    pref_match = re.search(
        r"tiene término preferido\s+([^0-9]+?)(?=\s+\d{4,}|\s+tiene|\s+se define|\s+pertenece|\s+es de tipo|$)", 
        narrative, 
        re.IGNORECASE
    )
    if pref_match:
        result["preferred"] = clean_term(pref_match.group(1))

    # 2. Sinónimos
    syn_matches = re.findall(
        r"tiene sinónimo\s+([^0-9]+?)(?=\s+\d{4,}|\s+tiene|\s+se define|\s+pertenece|\s+es de tipo|$)", 
        narrative, 
        re.IGNORECASE
    )
    for s in syn_matches:
        clean = clean_term(s)
        if clean:
            result["synonyms"].add(clean)

    # 3. Definición
    def_match = re.search(
        r"se define como:\s*(.*?)(?=\s+\d{4,}|\s+pertenece|\s+es de tipo|$)", 
        narrative, 
        re.IGNORECASE
    )
    if def_match:
        result["definition"] = def_match.group(1).strip()

    # 4. Tipo Semántico (capturamos todos para elegir el mejor luego)
    type_matches = re.findall(r"es de tipo\s+([a-zA-Z_]+)", narrative)
    if type_matches:
        result["types"] = type_matches

    return result

def determine_semantic_type(types_list):
    """Elige el tipo más específico de la lista encontrada."""
    if not types_list:
        return "Unknown"
    
    # Prioridad: Si hay algo que NO sea 'Class', úsalo.
    # 'Class' es el genérico de SNOMED.
    specific_types = [t for t in types_list if t.lower() != 'class']
    
    if specific_types:
        return specific_types[0] # Devolver el primero específico encontrado
    return types_list[0] # Si solo hay Class, devolvemos Class

def main():
    print("="*60)
    print(" 🏗️  ONTOLOGY BUILDER: CSV -> JSON MULTILINGÜE")
    print("="*60)
    
    if not RAW_CSV_PATH.exists():
        print(f"[ERROR] No se encuentra el archivo: {RAW_CSV_PATH}")
        sys.exit(1)

    print(f"[INFO] Leyendo: {RAW_CSV_PATH}")
    try:
        df = pd.read_csv(RAW_CSV_PATH)
    except Exception as e:
        print(f"[ERROR] Fallo al leer CSV: {e}")
        sys.exit(1)

    # Normalizar nombres de columnas
    col_narrativa = 'narrativa' if 'narrativa' in df.columns else 'term'
    col_concepto = 'concepto'

    ontology_list = []
    
    print(f"[INFO] Procesando {len(df)} filas...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        cid = str(row.get(col_concepto, ''))
        raw_text = str(row.get(col_narrativa, ''))
        
        # 1. Extraer datos crudos
        extracted = extract_data_from_narrative(raw_text)
        
        # 2. Consolidar lista de términos en Español
        # (Preferido + Sinónimos, eliminando duplicados y el propio preferido de la lista de sinónimos)
        terms_es = []
        if extracted["preferred"]:
            terms_es.append(extracted["preferred"])
        
        # Añadir sinónimos que no sean iguales al preferido
        for syn in extracted["synonyms"]:
            if extracted["preferred"] and syn.lower() != extracted["preferred"].lower():
                terms_es.append(syn)
            elif not extracted["preferred"]:
                 terms_es.append(syn)

        # Si no hay términos, saltamos este concepto (datos corruptos)
        if not terms_es:
            continue

        # 3. Construir el objeto JSON final
        entry = {
            "concept_id": cid,
            "semantic_type": determine_semantic_type(extracted["types"]),
            "languages": {
                "es": {
                    "terms": terms_es,
                    "definition": extracted["definition"]
                },
                "en": {
                    "terms": [],       # Placeholder para LLM
                    "definition": None # Placeholder para LLM
                },
                "ca": {
                    "terms": [],       # Placeholder para LLM
                    "definition": None # Placeholder para LLM
                }
            }
        }
        
        ontology_list.append(entry)

    # --- GUARDAR ---
    print(f"\n[INFO] Guardando JSON en: {OUTPUT_JSON_PATH}")
    
    # Asegurar que el directorio existe
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(ontology_list, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Completado. {len(ontology_list)} conceptos exportados.")
    
    # Preview
    if ontology_list:
        print("\n🔍 Preview del primer elemento:")
        print(json.dumps(ontology_list[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()