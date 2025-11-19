import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv 

# --- 1. CARGAR VARIABLES ---
load_dotenv() 

# --- CONFIGURACIÓN ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
INPUT_FILE = PROJECT_ROOT / "ontology" / "ontology.json"
OUTPUT_FILE = PROJECT_ROOT / "ontology" / "multilingual_ontology.json"

# --- 2. CONFIGURAR GEMINI ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ ERROR: No se encontró GEMINI_API_KEY en el archivo .env")
    exit(1)

genai.configure(api_key=API_KEY)

# --- 3. SELECCIÓN DE MODELO ---
# Intentamos usar el más rápido y moderno disponible
MODEL_NAME = "gemini-2.5-flash" # Default

# Configuración de generación para JSON estricto
generation_config = {
    "temperature": 0.0,  # Determinista
    "response_mime_type": "application/json"
}

model = genai.GenerativeModel(
    model_name=MODEL_NAME, 
    generation_config=generation_config
)

BATCH_SIZE = 15
SAVE_INTERVAL = 5  # Guardar cada 5 lotes

def generate_batch_prompt(batch_concepts):
    """
    Prompt Few-Shot optimizado para terminología SNOMED-CT.
    """
    concepts_str = json.dumps(batch_concepts, ensure_ascii=False)
    
    return f"""
    You are a Senior Clinical Terminologist specializing in SNOMED-CT localization.
    
    **OBJECTIVE:**
    Map the provided Spanish SNOMED-CT concepts to their standard English (International Edition) and Catalan equivalents.
    Focus on clinical accuracy and retrieval efficiency.
    
    **INSTRUCTIONS:**
    1. **Terms:** Provide the "Preferred Term" first, followed by valid clinical "Synonyms" and common acronyms (e.g., ACV -> CVA, Stroke).
    2. **Definition:** Translate the definition precisely. If the input definition is missing, generate a concise, clinically accurate one based on the term.
    3. **Format:** Output strictly valid JSON indexed by `concept_id`.
    
    **FEW-SHOT EXAMPLES (Learn from this pattern):**
    
    --- EXAMPLE 1 ---
    Input: [{{ "concept_id": "230690007", "term": "Accidente cerebrovascular", "def": "Déficit neurológico agudo..." }}]
    Output: 
    {{
      "230690007": {{
        "en": {{ 
            "terms": ["Cerebrovascular accident", "Stroke", "CVA", "Brain attack"], 
            "definition": "Acute neurological deficit resulting from interruption of blood supply to the brain." 
        }},
        "ca": {{ 
            "terms": ["Accident vascular cerebral", "Ictus", "AVC", "Feridura"], 
            "definition": "Dèficit neurològic agut resultat de la interrupció del subministrament sanguini al cervell." 
        }}
      }}
    }}
    
    --- EXAMPLE 2 ---
    Input: [{{ "concept_id": "38341003", "term": "Hipertensión arterial", "def": "Presión arterial elevada..." }}]
    Output: 
    {{
      "38341003": {{
        "en": {{ 
            "terms": ["Hypertensive disorder", "Hypertension", "HTN", "High blood pressure"], 
            "definition": "Condition characterized by persistently high pressure in the arteries." 
        }},
        "ca": {{ 
            "terms": ["Hipertensió arterial", "Hipertensió", "HTA"], 
            "definition": "Condició caracteritzada per una pressió persistentment alta a les artèries." 
        }}
      }}
    }}
    
    **YOUR TASK (Process this batch):**
    Input: {concepts_str}
    Output:
    """

def process_batch_with_gemini(batch):
    """Envía el lote a Gemini."""
    mini_batch = []
    for item in batch:
        # Solo enviamos lo necesario para ahorrar tokens
        term_es = item["languages"]["es"]["terms"][0] if item["languages"]["es"]["terms"] else "Unknown"
        mini_batch.append({
            "concept_id": item["concept_id"],
            "term": term_es,
            "def": item["languages"]["es"]["definition"]
        })
        
    prompt = generate_batch_prompt(mini_batch)
    
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"\n[ERROR] Gemini Batch Error: {e}")
        time.sleep(2) 
        return {}

def main():
    print("="*60)
    print(f" ♊  SNOMED-CT ENRICHER (Gemini 2.5 Flash | Resumable)")
    print("="*60)
    
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input not found: {INPUT_FILE}")
        return

    # 1. Cargar Ontología Base
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        base_ontology = json.load(f)
    
    # 2. Cargar Progreso Existente (Si existe)
    processed_ids = set()
    enriched_ontology = []
    
    if OUTPUT_FILE.exists():
        print(f"[INFO] Encontrado archivo existente. Intentando reanudar...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                enriched_ontology = existing_data
                for item in existing_data:
                    # Consideramos procesado si tiene al menos un término en inglés
                    if item["languages"]["en"]["terms"]: 
                        processed_ids.add(item['concept_id'])
                        
            print(f"[RESUME] {len(processed_ids)} conceptos ya procesados. Saltando...")
        except json.JSONDecodeError:
            print("[WARN] Archivo de salida corrupto. Empezando de cero.")
            enriched_ontology = []

    # 3. Identificar pendientes
    # Mapeamos la base por ID para acceso rápido
    base_map = {item['concept_id']: item for item in base_ontology}
    
    # Lista de IDs que faltan
    ids_to_process = [item['concept_id'] for item in base_ontology if item['concept_id'] not in processed_ids]
    
    if not ids_to_process:
        print("[SUCCESS] ¡Todo está procesado! No hay nada que hacer.")
        return

    print(f"[INFO] Procesando {len(ids_to_process)} conceptos restantes...")

    # 4. Bucle de Proceso
    for i in tqdm(range(0, len(ids_to_process), BATCH_SIZE)):
        batch_ids = ids_to_process[i:i + BATCH_SIZE]
        batch_items = [base_map[bid] for bid in batch_ids]
        
        # Llamada a Gemini
        translations_map = process_batch_with_gemini(batch_items)
        
        # Fusionar resultados
        for item in batch_items:
            cid = item["concept_id"]
            
            if translations_map and cid in translations_map:
                trans = translations_map[cid]
                
                # Inyectar EN
                if "en" in trans:
                    terms = trans["en"].get("terms", [])
                    if isinstance(terms, str): terms = [terms]
                    item["languages"]["en"]["terms"] = terms
                    item["languages"]["en"]["definition"] = trans["en"].get("definition")
                
                # Inyectar CA
                if "ca" in trans:
                    terms = trans["ca"].get("terms", [])
                    if isinstance(terms, str): terms = [terms]
                    item["languages"]["ca"]["terms"] = terms
                    item["languages"]["ca"]["definition"] = trans["ca"].get("definition")
            
            enriched_ontology.append(item)

        # --- CHECKPOINT: GUARDAR CADA X LOTES ---
        if (i // BATCH_SIZE) % SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(enriched_ontology, f, indent=2, ensure_ascii=False)
        
        # Rate limit courtesy
        time.sleep(0.5) 

    # Guardado Final
    print(f"\n[INFO] Guardado final en: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched_ontology, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Proceso completado.")

if __name__ == "__main__":
    main()