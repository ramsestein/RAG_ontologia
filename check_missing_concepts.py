"""
Script to check which concepts from the training data are missing in conceptos_con_narrativas.csv
Author: Oriol Farrés
Date: October 27, 2025
"""

import pandas as pd
import os
from collections import Counter

def load_ontology(filepath):
    """Load the ontology CSV and return set of concept IDs"""
    print(f"Loading ontology from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Check column names
    print(f"Ontology columns: {df.columns.tolist()}")
    
    # Assuming first column is the concept ID
    concept_col = df.columns[0]
    concepts = set(df[concept_col].astype(str).unique())
    print(f"Total unique concepts in ontology: {len(concepts)}")
    
    return concepts, df

def load_training_annotations(filepath):
    """Load training annotations and extract all concept codes"""
    print(f"\nLoading training annotations from: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"Annotation columns: {df.columns.tolist()}")
    
    # Extract all concept_id values
    all_codes = df['concept_id'].astype(str).tolist()
    
    print(f"Total annotations found: {len(all_codes)}")
    
    # Get unique codes and their frequencies
    unique_codes = set(all_codes)
    code_counts = Counter(all_codes)
    
    print(f"Total unique concept codes in training data: {len(unique_codes)}")
    
    return unique_codes, code_counts

def find_missing_concepts(ontology_concepts, training_concepts, code_counts):
    """Find which concepts are missing from the ontology"""
    missing = training_concepts - ontology_concepts
    
    print(f"\n{'='*80}")
    print(f"MISSING CONCEPTS ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal concepts in training data: {len(training_concepts)}")
    print(f"Total concepts in ontology: {len(ontology_concepts)}")
    print(f"Missing concepts: {len(missing)}")
    print(f"Coverage: {((len(training_concepts) - len(missing)) / len(training_concepts) * 100):.2f}%")
    
    if missing:
        print(f"\n{'='*80}")
        print(f"MISSING CONCEPT DETAILS (sorted by frequency)")
        print(f"{'='*80}")
        
        # Sort missing concepts by frequency
        missing_with_counts = [(code, code_counts[code]) for code in missing]
        missing_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Concept Code':<20} {'Frequency':<10} {'% of Total Annotations'}")
        print(f"{'-'*60}")
        
        total_annotations = sum(code_counts.values())
        
        for code, count in missing_with_counts:
            percentage = (count / total_annotations) * 100
            print(f"{code:<20} {count:<10} {percentage:.2f}%")
        
        # Calculate impact
        missing_annotations = sum(count for code, count in missing_with_counts)
        impact = (missing_annotations / total_annotations) * 100
        
        print(f"\n{'='*80}")
        print(f"IMPACT ANALYSIS")
        print(f"{'='*80}")
        print(f"Total annotations affected by missing concepts: {missing_annotations} / {total_annotations}")
        print(f"Percentage of annotations that cannot be matched: {impact:.2f}%")
    else:
        print("\n✅ All concepts from training data are present in the ontology!")
    
    return missing

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ontology_path = os.path.join(base_dir, 'conceptos_con_narrativas.csv')
    training_path = os.path.join(base_dir, 'benchmark', 'data', 'train_annotations.csv')
    
    # Check if files exist
    if not os.path.exists(ontology_path):
        print(f"❌ Ontology file not found: {ontology_path}")
        return
    
    if not os.path.exists(training_path):
        print(f"❌ Training file not found: {training_path}")
        return
    
    # Load data
    ontology_concepts, ontology_df = load_ontology(ontology_path)
    training_concepts, code_counts = load_training_annotations(training_path)
    
    # Find missing concepts
    missing = find_missing_concepts(ontology_concepts, training_concepts, code_counts)
    
    # Save missing concepts to file for further analysis
    if missing:
        output_path = os.path.join(base_dir, 'missing_concepts.txt')
        with open(output_path, 'w') as f:
            f.write("Missing Concepts from conceptos_con_narrativas.csv\n")
            f.write("="*80 + "\n\n")
            
            missing_with_counts = [(code, code_counts[code]) for code in missing]
            missing_with_counts.sort(key=lambda x: x[1], reverse=True)
            
            for code, count in missing_with_counts:
                f.write(f"{code}: {count} occurrences\n")
        
        print(f"\n📝 Missing concepts saved to: {output_path}")

if __name__ == "__main__":
    main()
