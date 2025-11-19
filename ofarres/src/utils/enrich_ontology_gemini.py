import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv 

# --- 1. CARGAR VARIABLES ---
load_dotenv() 

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# El archivo maestro (contiene TODOS los conceptos, pero solo en ES)
SOURCE_FILE = PROJECT_ROOT / "ontology" / "ontology.json"

# Tu archivo de trabajo (contiene algunos ya traducidos y será donde guardemos todo)
WORK_FILE = PROJECT_ROOT / "ontology" / "multilingual_ontology.json"

# --- 2. CONFIGURAR GEMINI ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ ERROR: No se encontró GEMINI_API_KEY en el archivo .env")
    exit(1)

genai.configure(api_key=API_KEY)

# --- 3. SELECCIÓN DE MODELO ---
MODEL_NAME = "gemini-2.5-flash" 

generation_config = {
    "temperature": 0.0,  
    "response_mime_type": "application/json"
}

model = genai.GenerativeModel(
    model_name=MODEL_NAME, 
    generation_config=generation_config
)

BATCH_SIZE = 15
SAVE_INTERVAL = 5 

def generate_batch_prompt(batch_concepts):
    """ Prompt Few-Shot optimizado (Sin cambios) """
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
    
    **FEW-SHOT EXAMPLES:**
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
    
    **YOUR TASK (Process this batch):**
    Input: {concepts_str}
    Output:
    """

def process_batch_with_gemini(batch):
    """Envía el lote a Gemini."""
    mini_batch = []
    for item in batch:
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

def sync_missing_items(source_data, work_data):
    """
    Compara el archivo maestro con el de trabajo y añade los que faltan.
    Devuelve la lista fusionada.
    """
    # Crear conjunto de IDs existentes en el archivo de trabajo
    existing_ids = set(item["concept_id"] for item in work_data)
    
    added_count = 0
    for item in source_data:
        if item["concept_id"] not in existing_ids:
            # Clonamos el item del maestro (solo tiene ES) y lo añadimos al trabajo
            work_data.append(item)
            added_count += 1
            
    if added_count > 0:
        print(f"[SYNC] 📥 Se han importado {added_count} conceptos nuevos desde ontology.json")
    else:
        print(f"[SYNC] ✅ El archivo de trabajo ya contiene todos los conceptos del maestro.")
        
    return work_data

def main():
    print("="*60)
    print(f" ♊  SNOMED-CT ENRICHER (Fusión + Proceso)")
    print("="*60)
    
    # 1. Cargar Maestro
    if not SOURCE_FILE.exists():
        print(f"[ERROR] No existe el archivo maestro: {SOURCE_FILE}")
        return
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_ontology = json.load(f)

    # 2. Cargar Archivo de Trabajo (si existe, sino lista vacía)
    work_ontology = []
    if WORK_FILE.exists():
        print(f"[INFO] Cargando trabajo previo de: {WORK_FILE}")
        try:
            with open(WORK_FILE, 'r', encoding='utf-8') as f:
                work_ontology = json.load(f)
        except json.JSONDecodeError:
            print("[WARN] Archivo de trabajo corrupto. Se regenerará desde cero.")
            work_ontology = []
    else:
        print("[INFO] No existe archivo de trabajo. Se creará nuevo.")

    # 3. SINCRONIZACIÓN: Añadir lo que falta del maestro al trabajo
    work_ontology = sync_missing_items(source_ontology, work_ontology)
    
    # Guardamos inmediatamente tras la sincronización para asegurar integridad
    with open(WORK_FILE, 'w', encoding='utf-8') as f:
        json.dump(work_ontology, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------
    # A PARTIR DE AQUÍ: LÓGICA DE RELLENADO (Igual que antes)
    # ---------------------------------------------------------
    
    total_concepts = len(work_ontology)
    pending_indices = []
    
    # Detectar cuáles de la lista fusionada les falta el inglés
    for idx, item in enumerate(work_ontology):
        has_english = bool(item["languages"]["en"]["terms"])
        if not has_english:
            pending_indices.append(idx)
            
    processed_count = total_concepts - len(pending_indices)
    
    print(f"\n📊 ESTADO DE TRABAJO:")
    print(f"   - Total conceptos: {total_concepts}")
    print(f"   - Completados:     {processed_count}")
    print(f"   - Pendientes:      {len(pending_indices)} (🚀 A procesar ahora)")
    
    if not pending_indices:
        print("\n[SUCCESS] ¡Todo completo! No hay nada que procesar.")
        return

    time.sleep(1)

    # 4. Bucle de Proceso
    for i in tqdm(range(0, len(pending_indices), BATCH_SIZE)):
        
        current_batch_indices = pending_indices[i : i + BATCH_SIZE]
        batch_items = [work_ontology[idx] for idx in current_batch_indices]
        
        translations_map = process_batch_with_gemini(batch_items)
        
        # Actualizar en memoria
        for idx, item in zip(current_batch_indices, batch_items):
            cid = item["concept_id"]
            if translations_map and cid in translations_map:
                trans = translations_map[cid]
                
                if "en" in trans:
                    terms = trans["en"].get("terms", [])
                    if isinstance(terms, str): terms = [terms]
                    work_ontology[idx]["languages"]["en"]["terms"] = terms
                    work_ontology[idx]["languages"]["en"]["definition"] = trans["en"].get("definition")
                
                if "ca" in trans:
                    terms = trans["ca"].get("terms", [])
                    if isinstance(terms, str): terms = [terms]
                    work_ontology[idx]["languages"]["ca"]["terms"] = terms
                    work_ontology[idx]["languages"]["ca"]["definition"] = trans["ca"].get("definition")
            
        # Checkpoint
        if (i // BATCH_SIZE) % SAVE_INTERVAL == 0:
            with open(WORK_FILE, 'w', encoding='utf-8') as f:
                json.dump(work_ontology, f, indent=2, ensure_ascii=False)
        
        time.sleep(0.5) 

    # Guardado Final
    print(f"\n[INFO] Guardado final en: {WORK_FILE}")
    with open(WORK_FILE, 'w', encoding='utf-8') as f:
        json.dump(work_ontology, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Proceso finalizado.")

if __name__ == "__main__":
    main()