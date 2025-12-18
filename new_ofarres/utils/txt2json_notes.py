#!/usr/bin/env python3
"""
Convert medical notes from .txt files to structured JSON format.

This script reads medical notes that follow a semi-structured format with headers
and converts them into a clean JSON dataset for NLP processing.

Header format examples:
- CODE (Description): content
- CODE: content

The content for each key spans multiple lines until the next header is found.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def parse_medical_note(text: str) -> Dict[str, str]:
    """
    Parse a medical note text into a dictionary of fields.
    
    The text contains headers in the format:
    - CODE (Description): content
    - CODE: content
    
    Content spans multiple lines until the next header.
    
    Args:
        text: Raw text content of the medical note
        
    Returns:
        Dictionary mapping field codes to their content
    """
    # Regular expression to match headers:
    # - Starts at beginning of line (^)
    # - Captures the CODE (uppercase letters, numbers, underscores)
    # - Optionally matches (Description) in parentheses
    # - Followed by a colon
    header_pattern = re.compile(
        r'^([A-Z0-9_]+)\s*(?:\([^)]*\))?\s*:\s*(.*)$',
        re.MULTILINE
    )
    
    # Find all headers and their positions
    headers = list(header_pattern.finditer(text))
    
    if not headers:
        return {}
    
    result = {}
    
    for i, match in enumerate(headers):
        code = match.group(1)  # The field code (e.g., UO_PACN, RESL)
        first_line = match.group(2).strip()  # Content on the same line as header
        
        # Determine the start and end positions for this field's content
        content_start = match.end()
        
        # If there's another header after this one, content ends there
        # Otherwise, content goes to the end of the text
        if i + 1 < len(headers):
            content_end = headers[i + 1].start()
        else:
            content_end = len(text)
        
        # Extract the multi-line content
        multi_line_content = text[content_start:content_end].strip()
        
        # Combine first line with multi-line content
        if first_line and multi_line_content:
            full_content = f"{first_line}\n{multi_line_content}"
        elif first_line:
            full_content = first_line
        else:
            full_content = multi_line_content
        
        # Clean up: strip extra whitespace but preserve internal line breaks
        full_content = full_content.strip()
        
        result[code] = full_content
    
    return result


def process_notes_folder(
    input_folder: Path,
    output_file: Path,
    pattern: str = "*.txt"
) -> int:
    """
    Process all text files in a folder and convert to JSON.
    
    Args:
        input_folder: Path to folder containing .txt files
        output_file: Path to output JSON file
        pattern: Glob pattern for files to process (default: "*.txt")
        
    Returns:
        Number of files processed
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    if not input_folder.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")
    
    print(f"Scanning folder: {input_folder}")
    print(f"File pattern: {pattern}")
    
    # Find all matching files
    txt_files = sorted(input_folder.glob(pattern))
    
    if not txt_files:
        print(f"Warning: No files matching '{pattern}' found in {input_folder}")
        return 0
    
    print(f"Found {len(txt_files)} file(s) to process\n")
    
    # Process each file
    dataset = []
    processed = 0
    errors = 0
    
    for txt_file in txt_files:
        try:
            print(f"Processing: {txt_file.name}...", end=" ")
            
            # Read the file
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Parse the content
            content = parse_medical_note(text)
            
            # Create entry
            entry = {
                "file_id": txt_file.name,
                "content": content
            }
            
            dataset.append(entry)
            processed += 1
            print(f"✓ ({len(content)} fields)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
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
    print(f"Total entries:       {len(dataset)}")
    print(f"Output file:         {output_file}")
    print(f"{'='*60}")
    
    # Show sample
    if dataset:
        print(f"\nSample entry (first file):")
        sample = dataset[0]
        print(f"File ID: {sample['file_id']}")
        print(f"Fields found: {list(sample['content'].keys())}")
        if sample['content']:
            first_key = list(sample['content'].keys())[0]
            first_value = sample['content'][first_key]
            preview = first_value[:100] + "..." if len(first_value) > 100 else first_value
            print(f"\nExample field '{first_key}':")
            print(f"  {preview}")
    
    return processed


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert medical notes from .txt files to structured JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files in default 'raw_notes' folder
  python txt2json_notes.py
  
  # Process files in a specific folder
  python txt2json_notes.py --input ../data/notes
  
  # Specify custom output file
  python txt2json_notes.py --output my_dataset.json
  
  # Process files with custom pattern
  python txt2json_notes.py --pattern "*_cleaned.txt"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='raw_notes',
        help='Input folder containing .txt files (default: raw_notes)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='medical_notes_dataset.json',
        help='Output JSON file (default: medical_notes_dataset.json)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.txt',
        help='File pattern to match (default: *.txt)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    
    # Input folder: if relative, make it relative to script directory
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir.parent / input_path
    
    # Output file: if relative, make it relative to script directory
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir.parent / output_path
    
    try:
        processed = process_notes_folder(
            input_folder=input_path,
            output_file=output_path,
            pattern=args.pattern
        )
        
        if processed > 0:
            print("\n✓ Conversion completed successfully!")
            return 0
        else:
            print("\n⚠ No files were processed.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
