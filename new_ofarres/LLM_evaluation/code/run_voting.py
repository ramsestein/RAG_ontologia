import json
import os
import re
import time
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT_FILE = os.path.join(BASE_DIR, "outputs/gpt_full.json")
DEEPSEEK_FILE = os.path.join(BASE_DIR, "outputs/deepseek_full.json")
TAXONOMY_FILE = os.path.join(BASE_DIR, "../data/processed/taxonomia.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "outputs/ground_truth.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Codes that should NEVER appear alone (Directions, Degrees, Generic terms)
# 7771000=Left, 24028007=Right, 103421006=Grade, of15=Stenotomy/Generic
BLACKLIST_CODES = ["7771000", "24028007", "103421006", "of15"]

# --- LOAD KEYS ---
load_dotenv(ENV_FILE)
api_key_openai = os.getenv("OPENAI_API_KEY")

# --- SSL BYPASS (Corporate Proxy Fix) ---
insecure_transport = httpx.Client(verify=False, timeout=60.0)
client_judge = OpenAI(api_key=api_key_openai, http_client=insecure_transport)

# --- HELPER 1: PARSE ENTITIES ---
def extract_entities(tagged_text):
    """
    Parses [text](code). Returns dict { 'text_lower': {'original': 'Text', 'code': 'Code'} }
    """
    if not tagged_text: return {}
    matches = re.findall(r"\[(.*?)\]\((.*?)\)", tagged_text)
    entity_map = {}
    for text, code in matches:
        key = text.strip().lower()
        entity_map[key] = {
            "original_text": text.strip(),
            "code": code.strip()
        }
    return entity_map

# --- HELPER 2: THE JUDGE (GPT-4o-mini) ---
def call_gpt_judge(note_text, conflicts, taxonomy_map):
    """
    Asks GPT-4o-mini to resolve disagreements based on taxonomy context.
    """
    # Token Optimization: Only send relevant taxonomy definitions
    relevant_codes = set()
    for c in conflicts:
        if c['gpt_code'] != "MISSING": relevant_codes.add(c['gpt_code'])
        if c['deepseek_code'] != "MISSING": relevant_codes.add(c['deepseek_code'])
    
    relevant_tax_str = "\n".join([f"Code {c}: {taxonomy_map.get(c, 'Unknown')}" for c in relevant_codes if c not in ["NULL", "MISSING"]])

    prompt = f"""
    ### ROLE
    You are an expert Medical Coding Judge.
    
    ### CONTEXT
    - **Original Text:** "{note_text}"
    - **Taxonomy Definitions:**
    {relevant_tax_str}

    ### DISAGREEMENTS
    {json.dumps(conflicts, indent=2)}

    ### TASK
    Decide the "final_code" for each conflict.
    1. If the text matches the Taxonomy Definition, use that code.
    2. If the text describes a DIFFERENT body part/condition than the Code (e.g. "Kidney" vs "Brain"), reject it ("NULL").
    3. If not in taxonomy, return "NULL".

    ### OUTPUT FORMAT
    Return ONLY a JSON list:
    [ {{"entity_text": "...", "final_code": "..."}} ]
    """

    try:
        response = client_judge.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ Judge Error: {e}")
        return []

# --- HELPER 3: INTELLIGENT REDUNDANCY FILTER ---
def filter_redundancies(entities):
    """
    1. Subsumption: Removes entities contained inside larger entities.
    2. Noise Reduction: Removes isolated BLACKLIST codes (Left/Right) if they stand alone.
    """
    if not entities: return []

    # 1. Sort by length DESCENDING (Longest first)
    sorted_ents = sorted(entities, key=lambda x: len(x['entity_text']), reverse=True)
    
    kept_entities = []
    
    for candidate in sorted_ents:
        cand_text = candidate['entity_text'].strip().lower()
        cand_code = candidate['code'].strip()
        is_subsumed = False
        
        # A. Check against already accepted (longer) entities
        for accepted in kept_entities:
            accepted_text = accepted['entity_text'].strip().lower()
            
            # IF candidate is inside accepted AND they are not the exact same string
            if cand_text in accepted_text and cand_text != accepted_text:
                is_subsumed = True
                break
        
        # B. Decision Logic
        if not is_subsumed:
            # If it's NOT subsumed, we usually keep it.
            # UNLESS it is a Blacklisted code (Left/Right) standing alone.
            if cand_code in BLACKLIST_CODES:
                continue # Skip isolated directions (Noise reduction)
                
            kept_entities.append(candidate)
            
    return kept_entities

# --- MAIN PIPELINE ---
def run_voting():
    print(f"🚀 Starting Integrated Pipeline (Extraction -> Judge -> Cleaning)...")
    
    # 1. Load Data
    try:
        with open(GPT_FILE, 'r', encoding='utf-8') as f: gpt_data = {str(n.get('note_id', n.get('id'))): n for n in json.load(f)}
        with open(DEEPSEEK_FILE, 'r', encoding='utf-8') as f: ds_data = {str(n.get('note_id', n.get('id'))): n for n in json.load(f)}
        with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f: taxonomy = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Critical Error: Missing file {e}")
        return

    tax_map = {t['code']: t['local_name'] for t in taxonomy}
    final_ground_truth = []
    
    # 2. Iterate Notes
    all_ids = sorted(list(gpt_data.keys()))
    
    for i, note_id in enumerate(all_ids):
        print(f"⚖️  Processing Note {note_id} ({i+1}/{len(all_ids)})...")
        
        gpt_note = gpt_data[note_id]
        ds_note = ds_data.get(note_id)
        
        if not ds_note:
            print("   -> Skipping (Missing DeepSeek data)")
            continue
            
        ents_gpt = extract_entities(gpt_note.get('tagged_text', ''))
        ents_ds = extract_entities(ds_note.get('tagged_text', ''))
        
        all_terms = set(ents_gpt.keys()).union(set(ents_ds.keys()))
        
        consensus = []
        conflicts = []
        
        # 3. Matching Phase
        for term in all_terms:
            val_gpt = ents_gpt.get(term)
            val_ds = ents_ds.get(term)
            
            code_gpt = val_gpt['code'] if val_gpt else "MISSING"
            code_ds = val_ds['code'] if val_ds else "MISSING"
            original_text = val_gpt['original_text'] if val_gpt else val_ds['original_text']

            # A. Consensus
            if code_gpt == code_ds:
                if code_gpt not in ["NULL", "MISSING"]:
                    consensus.append({
                        "entity_text": original_text,
                        "code": code_gpt,
                        "source": "CONSENSUS",
                        "confidence": "HIGH"
                    })
            # B. Conflict
            else:
                conflicts.append({
                    "entity_text": original_text,
                    "gpt_code": code_gpt,
                    "deepseek_code": code_ds
                })
        
        # 4. Arbitration Phase
        arbitrated_results = []
        if conflicts:
            print(f"   -> Found {len(conflicts)} conflicts. Calling Judge...")
            note_text = f"{gpt_note['clinical_data']['history']}\n{gpt_note['clinical_data']['findings']}"
            judgments = call_gpt_judge(note_text, conflicts, tax_map)
            
            for j in judgments:
                final_code = j.get('final_code', 'NULL')
                if final_code not in ["NULL", "MISSING"]:
                    arbitrated_results.append({
                        "entity_text": j['entity_text'],
                        "code": final_code,
                        "source": "ARBITER",
                        "confidence": "MEDIUM"
                    })
        
        # 5. Consolidation & Cleaning Phase (Subsumption)
        raw_entities = consensus + arbitrated_results
        
        # Apply the Intelligent Filter
        clean_entities = filter_redundancies(raw_entities)
        
        removed_count = len(raw_entities) - len(clean_entities)
        if removed_count > 0:
            print(f"   -> 🧹 Cleaned {removed_count} redundant/noise entities.")

        final_ground_truth.append({
            "note_id": note_id,
            "original_note": gpt_note.get('clinical_data'),
            "ground_truth_entities": clean_entities
        })

    # 6. Save Final Result
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_ground_truth, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Clean Ground Truth Generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_voting()