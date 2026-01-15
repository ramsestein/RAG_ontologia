import json
import os
import argparse
import re
import time  # NEW: For timing metrics
import httpx 
from dotenv import load_dotenv
from openai import OpenAI
import chunking

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "../data/medical_notes.json")
TAXONOMY_FILE = os.path.join(BASE_DIR, "../data/processed/taxonomia.json")
PROMPT_FILE = os.path.join(BASE_DIR, "prompts/tagging_prompt.txt")
ENV_FILE = os.path.join(BASE_DIR, ".env")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- LOAD KEYS ---
load_dotenv(ENV_FILE)
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")

# --- SSL BYPASS CLIENTS (CORPORATE PROXY FIX) ---
# verify=False bypasses the self-signed cert error. timeout=60 prevents infinite hangs.
insecure_transport = httpx.Client(verify=False, timeout=60.0)

client_openai = OpenAI(
    api_key=api_key_openai, 
    http_client=insecure_transport
) if api_key_openai else None

client_deepseek = OpenAI(
    api_key=api_key_deepseek, 
    base_url="https://api.deepseek.com",
    http_client=insecure_transport
) if api_key_deepseek else None

# --- HELPER FUNCTIONS ---

def flatten_taxonomy(taxonomy_json):
    """Formats taxonomy for the prompt text."""
    lines = []
    for item in taxonomy_json:
        terms = list(set([item['local_name']] + item.get('aliases', [])))
        lines.append(f"- Code: {item['code']} | Terms: {', '.join(terms)}")
    return "\n".join(lines)

# --- LLM CALL HANDLER ---

def call_llm(prompt, mode):
    # Split Prompt for System/User separation
    if "### REAL TASK STARTS HERE" in prompt:
        parts = prompt.split("### REAL TASK STARTS HERE")
        system_content = parts[0].strip()
        user_content = "### REAL TASK STARTS HERE" + parts[1]
    else:
        system_content = "You are a strict clinical entity tagger. Output format: [text](code)."
        user_content = prompt

    start_time = time.time() # Start Clock

    try:
        # GPT-4o-mini (Fast & Cheap)
        if mode == "GPT":
            if not client_openai: return "[ERROR: No OpenAI Key]"
            response = client_openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            duration = time.time() - start_time
            print(f"    > GPT-Mini finished in {duration:.2f}s")
            return response.choices[0].message.content.strip()

        # DeepSeek V3 (Fast Chat) vs R1 (Slow Reasoner)
        elif mode == "DeepSeek":
            if not client_deepseek: return "[ERROR: No DeepSeek Key]"
            
            # Use "deepseek-chat" (V3) for SPEED. Use "deepseek-reasoner" (R1) for LOGIC.
            # Currently set to V3 to fix your 10-minute hang issue.
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            duration = time.time() - start_time
            print(f"    > DeepSeek finished in {duration:.2f}s")
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Error calling {mode}: {e}")
        return "" 

# --- MAIN PIPELINE ---

def run_pipeline(mode):
    print(f"\n🚀 STARTING PIPELINE | Mode: {mode}")
    print(f"📂 Output Directory: {OUTPUT_DIR}\n")

    # 1. Load Resources
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f: notes = json.load(f)
        with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f: raw_tax = json.load(f)
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f: template = f.read()
    except FileNotFoundError as e:
        print(f"⛔ CRITICAL ERROR: File not found ({e})")
        return

    # 2. Flatten Taxonomy
    flat_tax_str = flatten_taxonomy(raw_tax)

    results_gpt = []
    results_ds = []

    # 3. Processing Loop
    for i, note in enumerate(notes):
        note_id = note.get('note_id') or note.get('id')
        
        full_text = f"{note['clinical_data']['history']}\n{note['clinical_data']['findings']}"
        chunks = chunking.create_smart_chunks(full_text)
        
        note_chunks_gpt = []
        note_chunks_ds = []

        print(f"📄 Note {note_id} ({i+1}/{len(notes)}) | {len(chunks)} Chunks processing...")
        note_start = time.time()

        for c_idx, chunk in enumerate(chunks):
            # Safe replacement
            final_prompt = template.replace("{taxonomy}", flat_tax_str).replace("{text}", chunk)

            # --- GPT ---
            if mode in ["GPT", "All"]:
                tagged = call_llm(final_prompt, "GPT")
                tagged = re.sub(r"^```(text|json)?\s*", "", tagged, flags=re.MULTILINE)
                tagged = re.sub(r"\s*```$", "", tagged, flags=re.MULTILINE)
                note_chunks_gpt.append(tagged)

            # --- DeepSeek ---
            if mode in ["DeepSeek", "All"]:
                tagged = call_llm(final_prompt, "DeepSeek")
                tagged = re.sub(r"^```(text|json)?\s*", "", tagged, flags=re.MULTILINE)
                tagged = re.sub(r"\s*```$", "", tagged, flags=re.MULTILINE)
                tagged = re.sub(r"<think>.*?</think>", "", tagged, flags=re.DOTALL)
                note_chunks_ds.append(tagged)
        
        note_end = time.time()
        print(f"⏱️  Note {note_id} processed in {note_end - note_start:.2f}s")

        # 4. Save to Memory
        if mode in ["GPT", "All"]:
            note_copy = note.copy()
            note_copy['tagged_text'] = " ".join(note_chunks_gpt)
            note_copy['model_used'] = "gpt-4o-mini"
            results_gpt.append(note_copy)

        if mode in ["DeepSeek", "All"]:
            note_copy = note.copy()
            note_copy['tagged_text'] = " ".join(note_chunks_ds)
            note_copy['model_used'] = "deepseek-chat (V3)" # Updated Label
            results_ds.append(note_copy)

    # 5. Final Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if mode in ["GPT", "All"]:
        path = os.path.join(OUTPUT_DIR, "gpt_full.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_gpt, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved GPT Results: {path}")

    if mode in ["DeepSeek", "All"]:
        path = os.path.join(OUTPUT_DIR, "deepseek_full.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_ds, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved DeepSeek Results: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["GPT", "DeepSeek", "All"], required=True)
    args = parser.parse_args()
    
    run_pipeline(args.mode)