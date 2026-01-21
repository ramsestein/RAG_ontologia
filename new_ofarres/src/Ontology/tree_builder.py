# -*- coding: utf-8 -*-
"""
Ontology Tree Builder for Spanish Radiology NER.

Uses owlready2 to load a SNOMED-based OWL ontology and constructs
hierarchical trees from extracted entities.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

try:
    from owlready2 import get_ontology, Thing
except ImportError:
    print("[ERROR] owlready2 not installed. Install with: pip install owlready2")
    raise


class OntologyTree:
    """
    Builds hierarchical trees from NER entities using an OWL ontology.
    
    Maps extracted entity codes to ontology classes and constructs
    lineages (paths to root) for tree visualization.
    """
    
    def __init__(self, owl_path: str):
        """
        Initialize the ontology tree builder.
        
        Args:
            owl_path: Path to the OWL ontology file
        """
        print(f"[OntologyTree] Loading ontology from: {owl_path}")
        
        if not os.path.exists(owl_path):
            raise FileNotFoundError(f"OWL file not found: {owl_path}")
        
        # Load ontology
        self.onto = get_ontology(f"file://{owl_path}").load()
        print(f"[OK] Ontology loaded: {self.onto.base_iri}")
        
        # Build code-to-class lookup
        self._build_code_lookup()
    
    def _build_code_lookup(self):
        """Build a dictionary mapping codes to OWL classes."""
        self.code_to_class = {}
        self.class_to_info = {}
        
        count = 0
        for cls in self.onto.classes():
            # Get the class name (IRI fragment)
            class_name = cls.name if hasattr(cls, 'name') else str(cls).split('#')[-1]
            
            # Try to get codigoTerminologia property
            codigo = None
            nombre = None
            
            # Check for codigoTerminologia annotation - SAFE access
            try:
                codigos_prop = getattr(cls, 'codigoTerminologia', None)
                if codigos_prop is not None:
                    codigos = list(codigos_prop)
                    if codigos:
                        codigo = str(codigos[0])
            except (TypeError, AttributeError):
                pass
            
            # Check for nombreLocal annotation - SAFE access
            try:
                nombres_prop = getattr(cls, 'nombreLocal', None)
                if nombres_prop is not None:
                    nombres = list(nombres_prop)
                    if nombres:
                        nombre = str(nombres[0])
            except (TypeError, AttributeError):
                pass
            
            # Store class info
            self.class_to_info[cls] = {
                'name': class_name,
                'code': codigo,
                'local_name': nombre or class_name
            }
            
            # Map code to class
            if codigo:
                self.code_to_class[codigo] = cls
                count += 1
            
            # Also map by class name (IRI fragment) for RID-style codes
            self.code_to_class[class_name] = cls
        
        print(f"[OK] Built lookup for {count} coded classes, {len(self.code_to_class)} total mappings")
    
    def _get_class_label(self, cls) -> str:
        """Get a readable label for a class."""
        info = self.class_to_info.get(cls, {})
        return info.get('local_name', cls.name if hasattr(cls, 'name') else str(cls).split('#')[-1])
    
    def get_lineage(self, code: str) -> List[Tuple[str, str]]:
        """
        Get the lineage (path to root) for a given code.
        
        Args:
            code: Entity code (e.g., "415582006" or "RID666")
            
        Returns:
            List of (class_name, label) tuples from root to leaf
        """
        # Find the class
        cls = self.code_to_class.get(code)
        
        if cls is None:
            # Try case-insensitive match
            for key, val in self.code_to_class.items():
                if str(key).lower() == code.lower():
                    cls = val
                    break
        
        if cls is None:
            return [("Unknown", f"[{code}] (not in ontology)")]
        
        # Build lineage by walking up is_a relationships
        lineage = []
        visited = set()
        current = cls
        
        while current is not None and current not in visited:
            visited.add(current)
            
            class_name = current.name if hasattr(current, 'name') else str(current).split('#')[-1]
            label = self._get_class_label(current)
            lineage.append((class_name, label))
            
            # Get parent(s) - filter out Thing and non-class parents
            try:
                parents = [p for p in current.is_a if p != Thing and hasattr(p, 'name')]
                current = parents[0] if parents else None
            except:
                current = None
        
        # Reverse to get root -> leaf order
        lineage.reverse()
        return lineage
    
    def build_tree_for_note(self, note: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a hierarchical tree for a medical note.
        
        Args:
            note: Note dict with 'id' and 'extracted_entities'
            
        Returns:
            Hierarchical tree structure
        """
        note_id = note.get('id', 'unknown')
        entities = note.get('extracted_entities', [])
        
        # Group by field_location first
        by_field = defaultdict(list)
        for ent in entities:
            field = ent.get('field_location', 'unknown')
            by_field[field].append(ent)
        
        # Build tree for each field
        tree = {
            'note_id': note_id,
            'fields': {}
        }
        
        for field, field_entities in by_field.items():
            field_tree = self._build_field_tree(field_entities)
            tree['fields'][field] = field_tree
        
        return tree
    
    def _build_field_tree(self, entities: List[Dict]) -> Dict:
        """Build a tree structure for entities in a single field."""
        # Build a nested dict representing the tree
        root = {}
        
        for ent in entities:
            code = ent.get('code', 'unknown')
            text = ent.get('text', '')
            category = ent.get('category', '')
            source = ent.get('source', '')
            
            # Get lineage
            lineage = self.get_lineage(code)
            
            # Insert into tree
            current = root
            for i, (class_name, label) in enumerate(lineage):
                if class_name not in current:
                    current[class_name] = {
                        '_label': label,
                        '_entities': [],
                        '_children': {}
                    }
                
                # If this is the leaf, add the entity
                if i == len(lineage) - 1:
                    current[class_name]['_entities'].append({
                        'code': code,
                        'text': text,
                        'category': category,
                        'source': source
                    })
                
                current = current[class_name]['_children']
        
        return root
    
    def print_ascii_tree(self, tree: Dict[str, Any], max_depth: int = 10):
        """
        Print a visual ASCII tree representation.
        
        Args:
            tree: Tree structure from build_tree_for_note()
            max_depth: Maximum depth to display
        """
        note_id = tree.get('note_id', 'unknown')
        fields = tree.get('fields', {})
        
        print(f"\n{'='*60}")
        print(f"Note: {note_id}")
        print('='*60)
        
        for field_name, field_tree in fields.items():
            print(f"\n[{field_name.upper()}]")
            self._print_node(field_tree, prefix="", depth=0, max_depth=max_depth)
    
    def _print_node(self, node: Dict, prefix: str, depth: int, max_depth: int):
        """Recursively print tree nodes."""
        if depth > max_depth:
            return
        
        items = list(node.items())
        for i, (class_name, data) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "|__ " if is_last else "|-- "
            
            label = data.get('_label', class_name)
            entities = data.get('_entities', [])
            children = data.get('_children', {})
            
            # Print this node
            if entities:
                for ent in entities:
                    text = ent.get('text', '')
                    code = ent.get('code', '')
                    source = ent.get('source', '')
                    print(f"{prefix}{connector}{label} -> [{code}] \"{text}\" ({source})")
            else:
                print(f"{prefix}{connector}{label}")
            
            # Print children
            new_prefix = prefix + ("    " if is_last else "|   ")
            if children:
                self._print_node(children, new_prefix, depth + 1, max_depth)
    
    def process_all_notes(self, results_path: str, output_path: str = None) -> List[Dict]:
        """
        Process all notes from ensemble results.
        
        Args:
            results_path: Path to ensemble_results.json
            output_path: Optional path to save knowledge graph JSON
            
        Returns:
            List of tree structures for all notes
        """
        print(f"\n[Processing] Loading entities from: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        
        print(f"[OK] Loaded {len(notes)} notes")
        
        trees = []
        for i, note in enumerate(notes):
            tree = self.build_tree_for_note(note)
            trees.append(tree)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(notes)} notes...")
        
        print(f"[OK] Built trees for {len(trees)} notes")
        
        # Save to JSON if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(trees, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] Knowledge graph saved to: {output_path}")
        
        return trees


# --- Main Entry Point ---
if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OWL_FILE = os.path.join(BASE_DIR, "data", "llm_snomed_final4.owl")
    RESULTS_FILE = os.path.join(BASE_DIR, "data", "processed", "ensemble_results.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "knowledge_graph.json")

    try:
        # Initialize tree builder
        tree_builder = OntologyTree(OWL_FILE)
        
        # Process all notes
        trees = tree_builder.process_all_notes(RESULTS_FILE, OUTPUT_FILE)
        
        # Print sample trees
        print("\n" + "="*60)
        print("SAMPLE OUTPUT: First 2 Notes")
        print("="*60)
        
        for tree in trees[:2]:
            tree_builder.print_ascii_tree(tree)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
