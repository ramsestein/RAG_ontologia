import json
import os
from pathlib import Path
from typing import List, Dict

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ground_truth_cleaned.json"

def is_valid_match(text: str, start: int, end: int) -> bool:
    """
    Determines if a Ground Truth annotation is valid based on boundary rules.
    
    Rules:
    1. LEFT BOUNDARY: Must strictly be the start of a word. 
       (Reject 'tia' inside 'initiated').
    
    2. RIGHT BOUNDARY:
       a) If end is punctuation/space -> Valid.
       b) If end is alphanumeric (part of a longer word):
          - If term is SHORT (<= 3 chars, e.g. 'CT', 'TIA'), REJECT. 
            (Prevents 'CT' inside 'Action').
          - If term is LONG (> 3 chars, e.g. 'Infarct'), ACCEPT.
            (Allows 'Infarct' inside 'Infarction' or 'Stroke' inside 'Strokes').
    """
    span_text = text[start:end]
    
    # --- RULE 1: STRICT LEFT BOUNDARY ---
    # The character before the match must NOT be a letter/number.
    if start > 0:
        char_before = text[start - 1]
        if char_before.isalnum():
            return False # It is a substring starting in the middle (e.g., ini*tia*ted)
            
    # --- RULE 2: CONDITIONAL RIGHT BOUNDARY ---
    # Check what comes immediately after the match
    if end < len(text):
        char_after = text[end]
        
        # If the word continues (alphanumeric)...
        if char_after.isalnum():
            # SUB-RULE A: Protect Acronyms (Length <= 3)
            # Acronyms must be whole words. We don't want "at" matching "attack".
            if len(span_text) <= 3:
                return False
            
            # SUB-RULE B: Allow Morphology for Clinical Terms (Length > 3)
            # Allow "Infarct" matching "Infarction", "Stroke" matching "Strokes"
            else:
                return True
                
    return True

def custom_json_dump(data: List[Dict], filepath: Path):
    """
    Saves JSON in the requested compact format:
    - One line per annotation object.
    - Sorted by char offset.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('[\n')
        
        for i, entry in enumerate(data):
            f.write('  {\n')
            f.write(f'    "note_id": "{entry["note_id"]}",\n')
            f.write('    "annotations": [\n')
            
            anns = entry['annotations']
            for j, ann in enumerate(anns):
                # Manual formatting to ensure single line: { "key": "val", ... }
                json_str = json.dumps(ann) 
                json_str = json_str.replace('{"', '{ "').replace('"}', '" }').replace('":', '": ').replace(',"', ', "')
                
                comma = "," if j < len(anns) - 1 else ""
                f.write(f'      {json_str}{comma}\n')
                
            f.write('    ]\n')
            
            comma = "," if i < len(data) - 1 else ""
            f.write(f'  }}{comma}\n')
            
        f.write(']\n')

def main():
    print(f"📂 Project Root: {PROJECT_ROOT}")
    
    if not NOTES_PATH.exists() or not GT_PATH.exists():
        print(f"[ERROR] Files not found.")
        return

    print(f"[1/3] Loading data...")
    with open(NOTES_PATH, 'r', encoding='utf-8') as f:
        notes = {n['note_id']: n['text'] for n in json.load(f)}
        
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
        
    cleaned_gt = []
    total_removed = 0
    
    print(f"[2/3] Cleaning and Sorting...")
    
    for entry in gt_data:
        nid = entry['note_id']
        text = notes.get(nid, "")
        valid_anns = []
        
        # 1. Filter Bad Annotations
        for ann in entry['annotations']:
            start, end = ann['start'], ann['end']
            
            if is_valid_match(text, start, end):
                valid_anns.append(ann)
            else:
                span_text = text[start:end]
                # Context for log
                s_ctx = max(0, start-5)
                e_ctx = min(len(text), end+5)
                context = text[s_ctx:e_ctx].replace('\n', ' ')
                print(f"   🗑️ Removed invalid match: '{span_text}' inside '...{context}...' (Note {nid})")
                total_removed += 1
        
        # 2. Sort by 'start' offset
        valid_anns.sort(key=lambda x: x['start'])
        
        cleaned_gt.append({
            "note_id": nid,
            "annotations": valid_anns
        })
        
    print(f"[3/3] Saving to {OUTPUT_PATH}...")
    custom_json_dump(cleaned_gt, OUTPUT_PATH)
    
    print(f"\n✅ DONE. Removed {total_removed} invalid matches.")
    print(f"✅ Kept valid morphological matches (e.g., 'Infarct' inside 'Infarction').")

if __name__ == "__main__":
    main()