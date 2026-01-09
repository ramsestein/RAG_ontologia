#!/usr/bin/env python3
"""
run_stage3.py - Run SemanticFisher (Stage 3) on test notes

This script processes all test notes using the SemanticFisher pipeline
and generates stage3 predictions for evaluation.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Import SemanticFisher from the same directory
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location("llm_module", Path(__file__).parent / "03_LLM.py")
llm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_module)
SemanticFisher = llm_module.SemanticFisher


def load_text_file(filepath: str) -> str:
    """Load text from a cleaned note file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def save_json(data, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_stage3_pipeline():
    """Run SemanticFisher on all test notes."""
    print("=" * 80)
    print("🎣 Running Stage 3: SemanticFisher (High-Recall NER)")
    print("=" * 80)
    print()
    
    # Setup paths
    base_path = Path(__file__).parent.parent.parent  # new_ofarres/
    notes_dir = base_path / 'data' / 'raw_notes'
    gt_path = base_path / 'test' / 'llm' / 'ground_truth.json'
    ontology_path = base_path.parent / 'ofarres' / 'backend' / 'ontology' / 'ontology.json'
    output_path = base_path / 'src' / 'NER' / 'output' / 'stage3_llm.json'
    
    # Verify paths
    if not ontology_path.exists():
        print(f"❌ Error: Ontology not found at {ontology_path}")
        return
    
    if not gt_path.exists():
        print(f"❌ Error: Ground truth not found at {gt_path}")
        return
    
    print(f"📂 Notes directory: {notes_dir}")
    print(f"📂 Ontology: {ontology_path}")
    print(f"📂 Ground truth: {gt_path}")
    print(f"📂 Output will be saved to: {output_path}")
    print()
    
    # Load ground truth to get the list of notes to process
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    note_ids = [item['id'] for item in ground_truth]
    print(f"📋 Found {len(note_ids)} notes in ground truth")
    print()
    
    # Initialize SemanticFisher
    print("🔧 Initializing SemanticFisher...")
    fisher = SemanticFisher(str(ontology_path))
    print()
    
    # Process each note
    all_results = []
    similarity_threshold = 0.5  # Lower threshold for higher recall
    gliner_threshold = 0.01     # Very low threshold to catch everything
    
    print("=" * 80)
    print("🔍 Processing notes...")
    print(f"    GLiNER threshold: {gliner_threshold} (max recall)")
    print(f"    Similarity threshold: {similarity_threshold}")
    print("=" * 80)
    print()
    
    for i, note_id in enumerate(note_ids, 1):
        note_file = notes_dir / f"{note_id}_cleaned.txt"
        
        if not note_file.exists():
            print(f"⚠️  [{i}/{len(note_ids)}] Note {note_id} not found, skipping...")
            continue
        
        # Load note text
        text = load_text_file(note_file)
        
        # Run SemanticFisher with tuned parameters
        entities = fisher.run_pipeline(
            text, 
            threshold=similarity_threshold,
            gliner_threshold=gliner_threshold,
            top_k=3
        )
        
        print(f"✓ [{i}/{len(note_ids)}] {note_id}: Found {len(entities)} entities")
        
        # Format results for this note
        for entity in entities:
            result = {
                "id": note_id,
                "text": entity["text"],
                "label": entity["label"],
                "code": entity["code"],
                "confidence": entity["confidence"],
                "matched_term": entity["matched_term"],
                "top_matches": entity.get("top_matches", []),
                "negated": False,  # SemanticFisher doesn't handle negation (can be added later)
                "method": "stage3_llm"
            }
            all_results.append(result)
    
    print()
    print("=" * 80)
    print("💾 Saving results...")
    print("=" * 80)
    
    # Save results
    output_data = {
        "metadata": {
            "stage": "stage3_llm",
            "method": "SemanticFisher",
            "timestamp": datetime.now().isoformat(),
            "total_notes": len(note_ids),
            "total_entities": len(all_results),
            "similarity_threshold": similarity_threshold,
            "gliner_threshold": gliner_threshold
        },
        "results": all_results
    }
    
    save_json(output_data, output_path)
    
    print(f"✅ Stage 3 results saved to: {output_path}")
    print(f"📊 Total entities extracted: {len(all_results)}")
    print()
    
    # Summary statistics
    codes_with_mapping = sum(1 for r in all_results if r['code'] is not None)
    codes_without_mapping = len(all_results) - codes_with_mapping
    
    print("=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    print(f"Total entities extracted: {len(all_results)}")
    print(f"  - Mapped to ontology: {codes_with_mapping} ({codes_with_mapping/len(all_results)*100:.1f}%)")
    print(f"  - Unmapped: {codes_without_mapping} ({codes_without_mapping/len(all_results)*100:.1f}%)")
    print()
    
    # Average confidence
    confidences = [r['confidence'] for r in all_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"Average confidence score: {avg_confidence:.3f}")
    print()
    
    return output_path


if __name__ == "__main__":
    try:
        output_path = run_stage3_pipeline()
        print("✅ Stage 3 pipeline completed successfully!")
        print()
        print("Next step: Run the evaluator with the new stage3 output:")
        print("  python src/evaluation/evaluator.py")
    except Exception as e:
        print(f"❌ Error running Stage 3 pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
