"""
Fishing Net Strategy for High-Recall NER
=========================================

This module implements a semantic entity extraction pipeline that combines:
1. GLiNER: For broad entity extraction (maximizing recall)
2. Sentence Transformers: For semantic embeddings
3. FAISS: For fast similarity search
4. Owlready2: For ontology integration

The "fishing net" casts wide to catch fuzzy terms, synonyms, and complex phrasings
that dictionary-based approaches might miss.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

from gliner import GLiNER
from sentence_transformers import SentenceTransformer
import faiss


class SemanticFisher:
    """
    A high-recall Named Entity Recognition pipeline that uses:
    - GLiNER for entity span extraction (the "net")
    - Embeddings + FAISS for semantic matching (the "mapper")
    """
    
    def __init__(self, ontology_path: str):
        """
        Initialize the Semantic Fisher pipeline.
        
        Args:
            ontology_path: Path to the ontology JSON file
        """
        print("🎣 Initializing SemanticFisher...")
        
        # Step 1: Load GLiNER model (for entity extraction)
        print("  📡 Loading GLiNER model...")
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        
        # Step 2: Load Embedding model (Spanish biomedical)
        print("  🧬 Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        )
        
        # Step 3: Load Ontology
        print(f"  🗂️  Loading ontology from {ontology_path}...")
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)
        
        # Step 4: Extract ontology labels and build FAISS index
        print("  🏗️  Building FAISS index...")
        self.labels, self.codes, self.index, self.id_to_code = self._build_faiss_index()
        
        print(f"✅ SemanticFisher ready! Indexed {len(self.codes)} ontology terms.\n")
    
    def _extract_ontology_labels(self) -> List[Tuple[str, str]]:
        """
        Extract all terms and their codes from the ontology JSON.
        Includes all synonyms and variations for each concept.
        
        Returns:
            List of (label_text, concept_id) tuples
        """
        label_code_pairs = []
        
        for concept in self.ontology:
            concept_id = concept.get("concept_id")
            
            if not concept_id:
                continue
            
            # Extract Spanish terms (primary language)
            languages = concept.get("languages", {})
            es_data = languages.get("es", {})
            terms = es_data.get("terms", [])
            
            # Add all Spanish terms
            for term in terms:
                if term and term.strip():
                    label_code_pairs.append((term.lower().strip(), concept_id))
            
            # Optionally, add English and Catalan terms if available
            for lang in ["en", "ca"]:
                lang_data = languages.get(lang, {})
                lang_terms = lang_data.get("terms", [])
                for term in lang_terms:
                    if term and term.strip():
                        label_code_pairs.append((term.lower().strip(), concept_id))
        
        return label_code_pairs
    
    def _build_faiss_index(self) -> Tuple[List[str], List[str], faiss.Index, Dict[int, str]]:
        """
        Build FAISS index from ontology labels.
        
        Returns:
            Tuple of (labels, codes, faiss_index, id_to_code_mapping)
        """
        # Extract labels and codes
        label_code_pairs = self._extract_ontology_labels()
        
        if not label_code_pairs:
            raise ValueError("No labels found in ontology!")
        
        labels = [pair[0] for pair in label_code_pairs]
        codes = [pair[1] for pair in label_code_pairs]
        
        # Generate embeddings for all labels
        print(f"    Embedding {len(labels)} ontology terms...")
        embeddings = self.embedding_model.encode(
            labels,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build FAISS index (using Inner Product for normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        
        # Create mapping from FAISS index ID to ontology code
        id_to_code = {i: code for i, code in enumerate(codes)}
        
        return labels, codes, index, id_to_code
    
    def run_pipeline(
        self,
        text: str,
        threshold: float = 0.5,
        gliner_threshold: float = 0.01,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Run the complete Fishing Net pipeline on input text.
        
        Args:
            text: Input medical text
            threshold: Minimum similarity score for mapping (default: 0.5)
            gliner_threshold: GLiNER extraction threshold (default: 0.01 for max recall)
            top_k: Number of top matches to consider from FAISS (default: 3)
        
        Returns:
            List of extracted entities with mappings:
            [
                {
                    "text": "entity span",
                    "label": "predicted label",
                    "code": "ONTOLOGY_CODE or None",
                    "confidence": float
                }
            ]
        """
        # Step 1: Cast the net - GLiNER extraction with broad labels
        gliner_labels = [
            "síntoma",
            "enfermedad",
            "procedimiento",
            "anatomía",
            "hallazgo clínico",
            "medicamento",
            "signo",
            "diagnóstico",
            "tratamiento",
            "examen",
            "prueba",
            "condición médica",
            "órgano",
            "estructura corporal"
        ]
        
        # Use very low threshold for maximum recall
        entities = self.gliner_model.predict_entities(
            text,
            gliner_labels,
            threshold=gliner_threshold
        )
        
        if not entities:
            return []
        
        # Step 2-4: Map each entity to ontology
        results = []
        
        for entity in entities:
            entity_text = entity["text"]
            entity_label = entity["label"]
            
            # Generate embedding for the extracted span
            entity_embedding = self.embedding_model.encode(
                [entity_text.lower()],
                convert_to_numpy=True
            )
            
            # Normalize
            entity_embedding = entity_embedding / np.linalg.norm(entity_embedding, keepdims=True)
            
            # Query FAISS index for top-k nearest neighbors
            similarities, indices = self.index.search(
                entity_embedding.astype('float32'),
                k=min(top_k, len(self.codes))
            )
            
            # Get the best match
            similarity_score = float(similarities[0][0])
            best_match_idx = int(indices[0][0])
            
            # Always assign the best match (for maximum recall)
            # Even if below threshold, assign it but mark low confidence
            ontology_code = self.id_to_code[best_match_idx]
            matched_label = self.labels[best_match_idx]
            
            # Store all top-k matches for potential disambiguation
            top_matches = []
            for i in range(len(similarities[0])):
                if similarities[0][i] >= threshold:
                    top_matches.append({
                        "code": self.id_to_code[int(indices[0][i])],
                        "term": self.labels[int(indices[0][i])],
                        "score": float(similarities[0][i])
                    })
            
            results.append({
                "text": entity_text,
                "label": entity_label,
                "code": ontology_code,
                "confidence": similarity_score,
                "matched_term": matched_label,
                "top_matches": top_matches if top_matches else [{"code": ontology_code, "term": matched_label, "score": similarity_score}]
            })
        
        return results
    
    def batch_process(
        self,
        texts: List[str],
        threshold: float = 0.5,
        gliner_threshold: float = 0.01,
        top_k: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            threshold: Minimum similarity score for mapping
            gliner_threshold: GLiNER extraction threshold
            top_k: Number of top matches to consider
        
        Returns:
            List of results for each input text
        """
        return [self.run_pipeline(text, threshold, gliner_threshold, top_k) for text in texts]


def main():
    """
    Test the SemanticFisher pipeline.
    """
    print("=" * 70)
    print("Testing SemanticFisher - High-Recall NER Pipeline")
    print("=" * 70)
    print()
    
    # Path to ontology (adjust as needed)
    # Try multiple possible paths
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "../../../ofarres/backend/ontology/ontology.json"),
        os.path.join(script_dir, "../../ofarres/backend/ontology/ontology.json"),
        os.path.join(script_dir, "../../data/processed/ontology.json"),
        os.path.join(script_dir, "../ontology.json"),
        "/mnt/c/Users/sterr/Desktop/RAG_ontologia/ofarres/backend/ontology/ontology.json",  # Absolute fallback
    ]
    
    ontology_path = None
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            ontology_path = normalized_path
            print(f"✅ Found ontology at: {ontology_path}")
            break
    
    if ontology_path is None:
        print("❌ Error: Could not find ontology.json file.")
        print("   Searched in:")
        for path in possible_paths:
            normalized = os.path.normpath(path)
            exists = "✓" if os.path.exists(normalized) else "✗"
            print(f"     [{exists}] {normalized}")
        print("\n   Please provide the correct path to your ontology file.")
        return
    
    # Initialize the fisher
    try:
        fisher = SemanticFisher(ontology_path)
    except Exception as e:
        print(f"❌ Error initializing SemanticFisher: {e}")
        return
    
    # Test sentences
    test_cases = [
        "Paciente con molestias inespecíficas en hipocondrio.",
        "Presenta dolor abdominal difuso y náuseas matutinas.",
        "Antecedentes de hipertensión arterial y diabetes mellitus tipo 2.",
        "Se observa hepatomegalia en ecografía.",
        "Tratamiento con paracetamol y omeprazol."
    ]
    
    print("\n" + "=" * 70)
    print("🧪 Running Test Cases")
    print("=" * 70)
    print()
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"   Text: {text}")
        print()
        
        results = fisher.run_pipeline(text, threshold=0.5, gliner_threshold=0.01)
        
        if results:
            print(f"   ✨ Found {len(results)} entities:")
            for j, entity in enumerate(results, 1):
                print(f"      {j}. '{entity['text']}'")
                print(f"         Label: {entity['label']}")
                print(f"         Code: {entity['code'] or 'UNMAPPED'}")
                print(f"         Confidence: {entity['confidence']:.3f}")
                if entity['matched_term']:
                    print(f"         Matched: {entity['matched_term']}")
        else:
            print("   ⚠️  No entities found.")
        
        print()
    
    print("=" * 70)
    print("✅ Testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
