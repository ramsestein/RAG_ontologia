import json
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import chunking  # Ensure chunking.py is in the same folder

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "../data/medical_notes.json")
TAXONOMY_FILE = os.path.join(BASE_DIR, "../data/processed/taxonomia.json")
PROMPT_FILE = os.path.join(BASE_DIR, "prompts/tagging_prompt.txt")
ENV_FILE = os.path.join(BASE_DIR, ".env")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Load Keys
load_dotenv(ENV_FILE)
# Handle cases where keys might be missing to avoid immediate crash
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")

client_openai = OpenAI(api_key=api_key_openai) if api_key_openai else None
client_deepseek = OpenAI(api_key=api_key_deepseek, base_url="https://api.deepseek.com") if api_key_deepseek else None

# --- FUNCTIONS ---

def flatten_taxonomy(taxonomy_json):
    """Formats taxonomy for the prompt."""
    lines = []
    for item in taxonomy_json:
        terms = list(set([item['local_name']] + item.get('aliases', [])))
        lines.append(f"- Code: {item['code']} | Terms: {', '.join(terms)}")
    return "\n".join(lines)

 # Inside run_tagging.py

def call_llm(prompt, mode):
    # We split the prompt into System (Instructions) and User (Data)
    # This is a heuristic split based on the marker we added in the text file
    parts = prompt.split("### REAL DATA START")
    
    if len(parts) == 2:
        system_content = parts[0].strip()
        user_content = parts[1].strip()
    else:
        # Fallback if split fails
        system_content = "You are an expert Medical Coder. output format: [text](code)."
        user_content = prompt

    try:
        if mode == "GPT":
            response = client_openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0
            )
            return response.choices[0].message.content.strip()
            
        elif mode == "DeepSeek":
            response = client_deepseek.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1 # Low temp for precision
            )
            return response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"❌ Error calling {mode}: {e}")
        return "" # Return empty to avoid crashing pipeline

def run_pipeline(mode):
    print(f"🚀 Starting Tagging Pipeline. Mode: {mode}")

    # Load Resources
    with open(DATA_FILE, 'r', encoding='utf-8') as f: notes = json.load(f)
    with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f: tax = json.load(f)
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f: template = f.read()

    flat_tax = flatten_taxonomy(tax)

    # We prepare lists to hold the FULL note objects with the new tags
    results_gpt = []
    results_ds = []

    for i, note in enumerate(notes):
        note_id = note.get('note_id') or note.get('id')
        
        # Merge History and Findings for context
        full_original_text = f"{note['clinical_data']['history']}\n{note['clinical_data']['findings']}"
        
        # 1. Chunking
        chunks = chunking.create_smart_chunks(full_original_text)
        
        # Buffers for chunks
        note_chunks_gpt = []
        note_chunks_ds = []

        print(f"Processing Note {note_id} ({i+1}/{len(notes)}) - {len(chunks)} chunks")

        for chunk in chunks:
            final_prompt = template.format(taxonomy=flat_tax, text=chunk)

            # 2. Inference
            if mode in ["GPT", "All"]:
                tagged = call_llm(final_prompt, "GPT")
                # Clean markdown if present
                tagged = tagged.replace("```text", "").replace("```", "")
                note_chunks_gpt.append(tagged)

            if mode in ["DeepSeek", "All"]:
                tagged = call_llm(final_prompt, "DeepSeek")
                tagged = tagged.replace("```text", "").replace("```", "")
                note_chunks_ds.append(tagged)

        # 3. Construct the output objects (PRESERVING METADATA)
        
        if mode in ["GPT", "All"]:
            # Deep copy the original note to avoid modifying it in place for the next loop
            note_copy_gpt = note.copy()
            # Inject the new data
            note_copy_gpt['tagged_text'] = " ".join(note_chunks_gpt)
            note_copy_gpt['processing_model'] = "gpt-4o"
            results_gpt.append(note_copy_gpt)

        if mode in ["DeepSeek", "All"]:
            note_copy_ds = note.copy()
            note_copy_ds['tagged_text'] = " ".join(note_chunks_ds)
            note_copy_ds['processing_model'] = "deepseek-v3"
            results_ds.append(note_copy_ds)

    # 4. Save Final JSON Files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if mode in ["GPT", "All"]:
        path = os.path.join(OUTPUT_DIR, "gpt_full.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_gpt, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved GPT results (with metadata) to {path}")

    if mode in ["DeepSeek", "All"]:
        path = os.path.join(OUTPUT_DIR, "deepseek_full.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_ds, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved DeepSeek results (with metadata) to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["GPT", "DeepSeek", "All"], required=True)
    args = parser.parse_args()
    run_pipeline(args.mode)