#!/usr/bin/env python3
"""
Convert Taxonomia.csv to a clean JSON format for NLP pipeline.

This script reads a medical taxonomy CSV file and:
- Cleans terminology codes (handles floats, scientific notation, NaN)
- Aggregates all synonym columns into a single aliases list
- Normalizes aliases (trim, lowercase, deduplicate)
- Fixes specific spelling errors (e.g., intracreaneal -> intracraneal)
- Generates unique internal codes (of1, of2...) for items without standard codes
- Outputs to taxonomia.json
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def clean_terminology_code(code):
    """
    Clean terminology code column values.
    
    Handles:
    - Floats/Scientific notation (e.g., 1.62183E+16) -> convert to full integer string
    - NaN/Empty values -> return "uncoded"
    - Standard integers -> convert to string
    
    Args:
        code: Raw code value from CSV
        
    Returns:
        Cleaned code as string or "uncoded"
    """
    # Check if value is NaN or empty
    if pd.isna(code) or code == '':
        return "uncoded"
    
    # If it's already a string, check if it's numeric
    if isinstance(code, str):
        code = code.strip()
        if not code:
            return "uncoded"
        try:
            # Try to convert to float first to handle scientific notation strings
            code = float(code)
        except ValueError:
            # If it's not numeric, return as-is (e.g. RID666)
            return code
    
    # Handle numeric values (float or int)
    if isinstance(code, (int, float, np.integer, np.floating)):
        # Convert to integer if it's a valid number
        try:
            # Use format to avoid scientific notation
            return f"{int(code)}"
        except (ValueError, OverflowError):
            return "uncoded"
    
    return str(code)


def fix_spelling_errors(text):
    """
    Fix known spelling errors in medical terminology.
    
    Args:
        text: Text to correct
        
    Returns:
        Corrected text
    """
    if not isinstance(text, str):
        return text
    
    # Dictionary of known spelling errors and their corrections
    corrections = {
        'intracreaneal': 'intracraneal',
    }
    
    # Apply corrections
    corrected = text
    for wrong, correct in corrections.items():
        corrected = corrected.replace(wrong, correct)
    
    return corrected


def aggregate_aliases(row, synonym_columns):
    """
    Aggregate all synonym columns into a single list of aliases.
    
    Args:
        row: DataFrame row
        synonym_columns: List of column names containing synonyms
        
    Returns:
        List of cleaned, deduplicated aliases
    """
    aliases = []
    
    for col in synonym_columns:
        if col in row.index:
            value = row[col]
            # Skip NaN or empty values
            if pd.notna(value) and str(value).strip():
                # Fix spelling errors first
                value = fix_spelling_errors(str(value))
                # Clean: strip whitespace and convert to lowercase
                cleaned = value.strip().lower()
                if cleaned:
                    aliases.append(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            unique_aliases.append(alias)
    
    return unique_aliases


def convert_csv_to_json(csv_path, json_path):
    """
    Convert taxonomia.csv to taxonomia.json.
    
    Args:
        csv_path: Path to input CSV file
        json_path: Path to output JSON file
    """
    print(f"Reading CSV from: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Define synonym columns (all columns that might contain text aliases)
    synonym_columns = [
        'nombre_local_hallazgo',
        'preferido',
        'sinonimo',
        'sinonimo_1',
        'sinonimo_2',
        'sinonimo_3',
        'sinonimo_4',
        'sinonimo_5'
    ]
    
    # Filter to only existing columns
    existing_synonym_columns = [col for col in synonym_columns if col in df.columns]
    print(f"Synonym columns found: {existing_synonym_columns}")
    
    # Process each row
    results = []
    skipped = 0
    internal_code_counter = 1  # Counter for generating 'of1', 'of2', etc.
    
    for idx, row in df.iterrows():
        # 1. Clean terminology code
        raw_code = clean_terminology_code(row.get('terminology_code'))
        
        # 2. Logic for Internal Codes
        # If the code is "uncoded", we assign a unique internal ID
        if raw_code == "uncoded":
            code = f"of{internal_code_counter}"
            internal_code_counter += 1
        else:
            code = raw_code
        
        # 3. Get local name and fix spelling
        local_name = row.get('nombre_local_hallazgo', '')
        if pd.isna(local_name):
            local_name = ''
        local_name = fix_spelling_errors(str(local_name)).strip().lower()
        
        # 4. Aggregate aliases (spelling fixes applied inside)
        aliases = aggregate_aliases(row, existing_synonym_columns)
        
        # Skip rows with no meaningful data
        if not local_name and not aliases:
            skipped += 1
            continue
        
        # Create entry
        entry = {
            "code": code,
            "local_name": local_name,
            "aliases": aliases
        }
        
        results.append(entry)
    
    # Write to JSON
    print(f"\nWriting JSON to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"{'='*60}")
    print(f"Total rows read:        {len(df)}")
    print(f"Rows converted to JSON: {len(results)}")
    print(f"Rows skipped (empty):   {skipped}")
    print(f"Internal codes generated: {internal_code_counter - 1} (of1 - of{internal_code_counter - 1})")
    print(f"Output file:            {json_path}")
    print(f"{'='*60}")
    
    # Show sample entries
    print(f"\nSample entries (first 3):")
    for i, entry in enumerate(results[:3], 1):
        print(f"\n{i}. Code: {entry['code']}")
        print(f"   Local name: {entry['local_name']}")
        print(f"   Aliases ({len(entry['aliases'])}): {entry['aliases'][:3]}...")


def main():
    """Main function to run the conversion."""
    # Define paths
    script_dir = Path(__file__).parent
    # Assuming standard structure: project/utils/this_script.py -> project/data
    data_dir = script_dir.parent / 'data' / 'processed'
    # Adjust if your raw csv is in /data/raw_taxonomy and output in /data/processed
    raw_data_dir = script_dir.parent / 'data' / 'raw_taxonomy'

    # Ensure the raw taxonomy path is respected even if 'processed' is missing.
    # Try to create the processed directory if it doesn't exist so output can be written.
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created missing processed data directory: {data_dir}")
        except Exception:
            # If creation fails, fall back to the project data directory
            data_dir = script_dir.parent / 'data'

    csv_path = raw_data_dir / 'Taxonomia.csv'
    # If not in raw, check processed or root
    if not csv_path.exists():
         csv_path = data_dir / 'Taxonomia.csv'
    
    json_path = data_dir / 'taxonomia.json'
    
    # Check if CSV exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please check your directory structure.")
        return 1
    
    try:
        convert_csv_to_json(csv_path, json_path)
        print("\n✓ Conversion completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())