import json
from pathlib import Path
from flashtext import KeywordProcessor

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
JSON_PATH = PROJECT_ROOT / "ontology" / "multilingual_ontology.json"

# Términos que vimos fallar en tus screenshots
MISSING_TERMS = [
    "Thrombectomy", "thrombectomy",
    "Angiography", "angiography",
    "Stroke", "stroke",
    "Infarct", "Infarcts",
    "Weakness", "left-sided weakness"
]

def main():
    print(f"🔍 AUTOPSIA DE ONTOLOGÍA: {JSON_PATH}")
    
    if not JSON_PATH.exists():
        print("❌ No se encuentra el archivo JSON. ¿Ejecutaste el enricher?")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    print(f"📚 Conceptos cargados: {len(ontology)}")
    
    processor = KeywordProcessor(case_sensitive=False)
    term_to_id = {}

    # Cargar en FlashText (simulando OntologyNER)
    for item in ontology:
        cid = item['concept_id']
        langs = item.get('languages', {})
        
        # Chequear Español
        for t in langs.get('es', {}).get('terms', []):
            processor.add_keyword(t, cid)
            term_to_id[t.lower()] = cid
            
        # Chequear Inglés
        for t in langs.get('en', {}).get('terms', []):
            processor.add_keyword(t, cid)
            term_to_id[t.lower()] = cid

    print("\n--- DIAGNÓSTICO DE TÉRMINOS PERDIDOS ---")
    
    for term in MISSING_TERMS:
        print(f"\n🔎 Buscando: '{term}'")
        
        # 1. ¿Está en el diccionario exacto?
        if term.lower() in term_to_id:
            print(f"   ✅ ESTÁ EN LA ONTOLOGÍA (ID: {term_to_id[term.lower()]})")
        else:
            print(f"   ❌ NO ESTÁ en la lista de términos cargada.")
            # Búsqueda parcial para ver si es un problema de string
            found_partial = [k for k in term_to_id.keys() if term.lower() in k]
            if found_partial:
                print(f"      ...pero encontré variantes similares: {found_partial[:3]}")

        # 2. ¿Lo extrae FlashText de una frase?
        text_test = f"The patient underwent {term} yesterday."
        extracted = processor.extract_keywords(text_test)
        if extracted:
            print(f"   ✅ FLASHTEXT LO DETECTA: {extracted}")
        else:
            print(f"   ❌ FLASHTEXT FALLA en la extracción.")

if __name__ == "__main__":
    main()