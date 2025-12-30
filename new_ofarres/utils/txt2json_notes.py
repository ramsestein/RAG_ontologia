#!/usr/bin/env python3
"""
Convert medical notes from .txt files to a CLEAN and STRUCTURED JSON format.

This script reads medical notes, parses the raw headers, and transforms them
into a canonical schema (clinical_data, study_metadata, admin_metadata).
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def parse_medical_note(text: str) -> Dict[str, str]:
    """
    Parse a medical note text into a dictionary of raw fields.
    """
    header_pattern = re.compile(
        r'^([A-Z0-9_]+)\s*(?:\([^)]*\))?\s*:\s*(.*)$',
        re.MULTILINE
    )
    
    headers = list(header_pattern.finditer(text))
    
    if not headers:
        return {}
    
    result = {}
    
    for i, match in enumerate(headers):
        code = match.group(1)
        first_line = match.group(2).strip()
        
        content_start = match.end()
        
        if i + 1 < len(headers):
            content_end = headers[i + 1].start()
        else:
            content_end = len(text)
        
        multi_line_content = text[content_start:content_end].strip()
        
        if first_line and multi_line_content:
            full_content = f"{first_line}\n{multi_line_content}"
        elif first_line:
            full_content = first_line
        else:
            full_content = multi_line_content
        
        result[code] = full_content.strip()
    
    return result


def transform_to_canonical(raw_content: Dict[str, str], file_name: str, note_id: int) -> Dict:
    """
    Transforms the raw content dictionary into the clean, nested structure.
    """
    # Clean the filename to create a clean ID (remove _cleaned.txt extension)
    clean_id = file_name.replace("_cleaned.txt", "").replace(".txt", "")

    return {
        "id": clean_id,
        "note_id": note_id,  # Sequential ID as requested
        
        # 1. CLINICAL DATA (The signal for NLP)
        "clinical_data": {
            "history": raw_content.get("CLIN", "").strip(),
            "findings": raw_content.get("RESL", "").strip(),
            "impression": raw_content.get("CONCL", "").strip()
        },
        
        # 2. STUDY METADATA (Context)
        "study_metadata": {
            "procedure": raw_content.get("PRUEB", "").strip(),
            "protocol": raw_content.get("OBSR", "").strip(),
            "description": raw_content.get("EST_T_DES", "").strip()
        },
        
        # 3. ADMIN METADATA (Administrative info)
        "admin_metadata": {
            "unit": raw_content.get("UO_PACN", "").strip(),
            "tech_code": raw_content.get("COD_TECN", "").strip(),
            "additional_activity": raw_content.get("ACTI_ADICI", "").strip(),
            "diagnosis_text": raw_content.get("X00DIATXT", "").strip(),
            "comments": raw_content.get("DKTXT", "").strip(),
            "icon_code": raw_content.get("ICON", "").strip(),
            "vessels_right": raw_content.get("ZVASOS_D", "").strip()
        }
    }


def process_notes_folder(
    input_folder: Path,
    output_file: Path,
    pattern: str = "*.txt"
) -> int:
    """
    Process all text files in a folder, parse, transform and save to JSON.
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    print(f"Scanning folder: {input_folder}")
    print(f"File pattern: {pattern}")
    
    txt_files = sorted(input_folder.glob(pattern))
    
    if not txt_files:
        print(f"Warning: No files matching '{pattern}' found in {input_folder}")
        return 0
    
    print(f"Found {len(txt_files)} file(s) to process\n")
    
    dataset = []
    processed = 0
    errors = 0
    note_id = 1  # Initialize sequential ID
    
    for txt_file in txt_files:
        try:
            # Read the file
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 1. Parse raw text
            raw_content = parse_medical_note(text)
            
            # 2. Transform to Canonical Structure
            clean_entry = transform_to_canonical(raw_content, txt_file.name, note_id)
            
            dataset.append(clean_entry)
            processed += 1
            note_id += 1
            
        except Exception as e:
            print(f"✗ Error processing {txt_file.name}: {e}")
            errors += 1
    
    # Write to JSON
    print(f"\nWriting output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"{'='*60}")
    print(f"Files processed:     {processed}")
    print(f"Files with errors:   {errors}")
    print(f"Output file:         {output_file}")
    print(f"{'='*60}")
    
    # Show sample
    if dataset:
        print(f"\nSample entry (first file):")
        sample = dataset[0]
        print(f"ID: {sample['id']}")
        print(f"Note ID: {sample['note_id']}")
        print(f"Findings start: {sample['clinical_data']['findings'][:50]}...")
    
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Convert medical notes .txt -> structured clean JSON"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw_notes',
        help='Input folder containing .txt files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/medical_notes.json',
        help='Output JSON file'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.txt',
        help='File pattern to match'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir.parent / input_path
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir.parent / output_path
    
    try:
        process_notes_folder(input_path, output_path, args.pattern)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())