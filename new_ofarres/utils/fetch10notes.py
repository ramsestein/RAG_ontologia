#!/usr/bin/env python3
"""
Select 16 non-consecutive random notes from medical_notes.json for validation testing.

This script:
- Uses a fixed random seed (RND=16) for reproducibility
- Selects 16 notes ensuring they are not consecutive
- Saves the selected notes to test/samples/validation_test.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def select_non_consecutive_notes(notes: List[Dict], num_samples: int = 16, min_gap: int = 2, seed: int = 16) -> List[Dict]:
    """
    Select non-consecutive notes from the dataset.
    
    Args:
        notes: List of all notes
        num_samples: Number of notes to select (default: 16)
        min_gap: Minimum gap between consecutive selected indices (default: 2)
        seed: Random seed for reproducibility (default: 16)
        
    Returns:
        List of selected notes
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    total_notes = len(notes)
    print(f"Total notes available: {total_notes}")
    
    if num_samples > total_notes:
        raise ValueError(f"Cannot select {num_samples} notes from {total_notes} available notes")
    
    # Generate all possible indices
    all_indices = list(range(total_notes))
    
    # Shuffle indices
    random.shuffle(all_indices)
    
    # Select indices ensuring they are not consecutive
    selected_indices = []
    
    for idx in all_indices:
        # Check if this index is far enough from all previously selected indices
        is_valid = True
        for selected_idx in selected_indices:
            if abs(idx - selected_idx) < min_gap:
                is_valid = False
                break
        
        if is_valid:
            selected_indices.append(idx)
        
        # Stop when we have enough samples
        if len(selected_indices) == num_samples:
            break
    
    # If we couldn't find enough non-consecutive indices, raise an error
    if len(selected_indices) < num_samples:
        raise ValueError(
            f"Could not find {num_samples} non-consecutive notes with gap >= {min_gap}. "
            f"Only found {len(selected_indices)}. Try reducing min_gap or num_samples."
        )
    
    # Sort indices for easier verification
    selected_indices.sort()
    
    print(f"\nSelected {len(selected_indices)} note indices:")
    print(f"  Indices: {selected_indices}")
    print(f"  Note IDs: {[notes[i].get('note_id', i+1) for i in selected_indices]}")
    
    # Extract the selected notes
    selected_notes = [notes[i] for i in selected_indices]
    
    # Verify non-consecutiveness
    gaps = [selected_indices[i+1] - selected_indices[i] for i in range(len(selected_indices)-1)]
    min_actual_gap = min(gaps) if gaps else float('inf')
    print(f"  Minimum gap between indices: {min_actual_gap}")
    
    return selected_notes


def main():
    """Main function to select and save validation notes."""
    # Configuration
    RND = 16  # Fixed random seed
    NUM_SAMPLES = 12  # Number of notes to select
    MIN_GAP = 2  # Minimum gap between consecutive indices
    
    # Define paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    input_file = project_dir / 'data' / 'medical_notes.json'
    output_dir = project_dir / 'test' / 'samples'
    output_file = output_dir / 'validation_test.json'
    
    print("="*60)
    print("Medical Notes Validation Set Generator")
    print("="*60)
    print(f"Configuration:")
    print(f"  Random seed (RND): {RND}")
    print(f"  Number of samples: {NUM_SAMPLES}")
    print(f"  Minimum gap: {MIN_GAP}")
    print(f"  Input file: {input_file}")
    print(f"  Output file: {output_file}")
    print("="*60)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\n✗ Error: Input file not found: {input_file}")
        return 1
    
    # Load notes
    print(f"\nLoading notes from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        notes = json.load(f)
    
    print(f"Loaded {len(notes)} notes")
    
    try:
        # Select non-consecutive notes
        selected_notes = select_non_consecutive_notes(
            notes=notes,
            num_samples=NUM_SAMPLES,
            min_gap=MIN_GAP,
            seed=RND
        )
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save selected notes
        print(f"\nSaving validation set to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_notes, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        print(f"Total notes in dataset: {len(notes)}")
        print(f"Notes selected: {len(selected_notes)}")
        print(f"Output saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        print(f"{'='*60}")
        
        # Show sample
        if selected_notes:
            print(f"\nFirst selected note sample:")
            sample = selected_notes[0]
            print(f"  Note ID: {sample.get('note_id', 'N/A')}")
            print(f"  File ID: {sample.get('id', sample.get('file_id', 'N/A'))}")
            if 'clinical_data' in sample and 'history' in sample['clinical_data']:
                history_preview = sample['clinical_data']['history'][:80]
                print(f"  History: {history_preview}...")
        
        print("\n✓ Validation set created successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
