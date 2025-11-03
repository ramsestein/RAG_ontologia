"""
Módulo RAG (Retrieval-Augmented Generation) usando FAISS
Robusto y escalable:
- Carga hints EN→ES opcionales desde JSON (assets/bilingual_hints.json) con fallback embebido.
- Expansión de queries multivariantes (EN/ES, lowercasing).
- Búsqueda multi-query con fusión por máximo de similitud.
- Embeddings de query con ventana mayor (max_length=64, padding=True) y mean pooling.
"""

import os
import json
import pickle
import faiss
from typing import List, Tuple, Dict, Iterable
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# --- MODELO CORRECTO (del README) ---
MODEL_NAME = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'


class RAGRetriever:
    """Sistema de recuperación semántica usando FAISS (SapBERT mean-pooling)"""
    
    def __init__(self, assets_dir: str):
        """
        Args:
            assets_dir: Directorio con índice FAISS y archivos pickle
        """
        self.assets_dir = assets_dir
        self.faiss_index = None
        self.conceptos = []
        self.narrativas = []
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        self.hints: Dict[str, Iterable[str]] = {}
        
        self._load_model_and_tokenizer()
        self._load_index()
        self._load_ontology()
        self._load_bilingual_hints()
        # Validación de coherencia
        try:
            self._validate_index_coherence()
            print("[RAG] [OK] Validación de índice superada")
        except Exception as e:
            print(f"[RAG] [ERROR] {e}")
            raise

    # ---------------------------
    # Carga de recursos
    # ---------------------------
    def _load_model_and_tokenizer(self):
        """Carga el modelo y tokenizador de HuggingFace (SapBERT-style)"""
        try:
            print(f"[RAG] Cargando modelo y tokenizador: {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            self.model.eval()
            print(f"[RAG] [OK] Modelo cargado en {self.device}")
        except Exception as e:
            print(f"[RAG] [ERROR] No se pudo cargar el modelo de HuggingFace: {e}")

    def _load_index(self):
        """Carga el índice FAISS pre-construido"""
        index_path = os.path.join(self.assets_dir, 'ontology.index')
        if not os.path.exists(index_path):
            print(f"[RAG] [WARNING] Índice FAISS no encontrado en {index_path}")
            return
        
        try:
            print(f"[RAG] Cargando índice FAISS...")
            self.faiss_index = faiss.read_index(index_path)
            
            metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RAG] [OK] Índice cargado: {metadata['n_concepts']} conceptos (Modelo: {metadata.get('model_name', '??')})")
                
                # Validar que el índice se hizo con el mismo modelo
                if metadata.get('model_name') != MODEL_NAME:
                    print(f"[RAG] [WARNING] ¡El índice fue construido con un modelo diferente ({metadata.get('model_name')})!")
                    print(f"[RAG] [WARNING] Por favor, borra los assets y re-ejecuta ontology_preprocessor.py")

        except Exception as e:
            print(f"[RAG] [ERROR] Error cargando índice: {e}")
    
    def _load_ontology(self):
        """Carga conceptos y narrativas desde pickle"""
        concepts_path = os.path.join(self.assets_dir, 'ontology_concepts.pkl')
        narratives_path = os.path.join(self.assets_dir, 'ontology_narratives.pkl')
        
        try:
            with open(concepts_path, 'rb') as f:
                self.conceptos = pickle.load(f)
            with open(narratives_path, 'rb') as f:
                self.narrativas = pickle.load(f)
            print(f"[RAG] [OK] Ontología cargada: {len(self.conceptos)} conceptos")
        except Exception as e:
            print(f"[RAG] [ERROR] Error cargando ontología: {e}")

    def _load_bilingual_hints(self):
        """
        Carga mapeos EN→ES desde JSON opcional (assets/bilingual_hints.json).
        Si no existe, usa un conjunto por defecto.
        """
        default_hints = {
            "stroke": ["ictus", "accidente cerebrovascular"],
            "headache": ["cefalea", "dolor de cabeza"],
            "nausea": ["náusea"],
            "vomiting": ["vómito", "emesis"],
            "weakness": ["debilidad", "astenia"],
            "hemiparesis": ["hemiparesia"],
            "hemiplegia": ["hemiplejia"],
            "aphasia": ["afasia"],
            "dysarthria": ["disartria"],
            "atrial fibrillation": ["fibrilación auricular", "FA"],
            "thrombectomy": ["trombectomía", "terapia endovascular"],
            "tpa": ["alteplasa", "tPA", "rtPA"],
            "middle cerebral artery": ["arteria cerebral media", "ACM"],
            "mca": ["arteria cerebral media"],
            "occlusion": ["oclusión"],
            "recanalization": ["recanalización"],
            "ischemic penumbra": ["penumbra isquémica", "penumbra"],
            "hemorrhage": ["hemorragia", "hemorragia intracerebral", "hemorragia intracraneal"],
            "infarct": ["infarto cerebral", "infarto isquémico"],
        }
        path = os.path.join(self.assets_dir, "bilingual_hints.json")
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # normalizar a listas
                for k, v in data.items():
                    if isinstance(v, str):
                        data[k.lower()] = [v]
                    elif isinstance(v, list):
                        data[k.lower()] = list({str(x) for x in v if str(x).strip()})
                self.hints = {**default_hints, **data}
                print(f"[RAG] [OK] Hints bilingües cargados desde {path} ({len(self.hints)} entradas)")
            else:
                self.hints = default_hints
                print(f"[RAG] [OK] Hints bilingües por defecto ({len(self.hints)} entradas)")
        except Exception as e:
            print(f"[RAG] [WARNING] No se pudieron cargar hints desde JSON: {e}")
            self.hints = default_hints

    # ---------------------------
    # Embeddings y búsqueda
    # ---------------------------
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Genera embedding normalizado (mean pooling con máscara)."""
        text = query if isinstance(query, str) else str(query)
        with torch.no_grad():
            toks = self.tokenizer.encode_plus(
                text,
                padding=True,
                max_length=64,   # ventana mayor para términos compuestos
                truncation=True,
                return_tensors="pt"
            )
            toks_on_device = {k: v.to(self.device) for k, v in toks.items()}

            outputs = self.model(**toks_on_device)
            last_hidden = outputs.last_hidden_state            # (B, T, H)
            mask = toks_on_device["attention_mask"].unsqueeze(-1)  # (B, T, 1)

            # mean pooling sobre tokens válidos
            sum_vec = (last_hidden * mask).sum(dim=1)          # (B, H)
            len_vec = mask.sum(dim=1).clamp(min=1)             # (B, 1)
            mean_vec = sum_vec / len_vec

            emb = mean_vec.cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            normalized_emb = emb / np.clip(norm, 1e-12, None)

            return normalized_emb.astype("float32")

    def _search_single(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        """Búsqueda FAISS para una sola query ya expandida/normalizada."""
        if self.faiss_index is None or self.model is None:
            print("[RAG] [WARNING] Sistema RAG no disponible. Faltan índice o modelo.")
            return []

        try:
            q_emb = self._get_query_embedding(query)
            k = int(k)
            distances, indices = self.faiss_index.search(q_emb, k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0:
                    continue
                if idx < len(self.conceptos):
                    concepto = self.conceptos[idx]
                    narrativa = self.narrativas[idx]
                    similarity = float(distances[0][i])
                    if str(concepto).isdigit():
                        results.append((concepto, narrativa, similarity))
            return results
        except Exception as e:
            print(f"[RAG] [ERROR] Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _expand_query_variants(self, q: str) -> List[str]:
        """
        Genera variantes de búsqueda:
        - original
        - lower
        - hints EN→ES si existen
        """
        variants = []
        if not isinstance(q, str):
            q = str(q)
        base = q.strip()
        if not base:
            return variants
        candidates = [base, base.lower()]
        key = base.lower()
        if key in self.hints:
            # puede ser lista
            for es in self.hints[key]:
                candidates.append(es)
        # deduplicación conservando orden
        seen = set()
        for c in candidates:
            c2 = c.strip()
            if c2 and c2 not in seen:
                seen.add(c2)
                variants.append(c2)
        return variants

    def retrieve_multi(self, queries: List[str], k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Ejecuta varias queries y fusiona resultados por máximo de similitud por concepto.
        """
        if not queries:
            return []
        pool: Dict[str, Tuple[str, float]] = {}
        for q in queries:
            res = self._search_single(q, k)
            for concepto, narrativa, sim in res:
                # conservar la mejor similitud para cada concepto
                prev = pool.get(concepto)
                if (prev is None) or (sim > prev[1]):
                    pool[concepto] = (narrativa, sim)
        fused = [(c, n, s) for c, (n, s) in pool.items()]
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[:k]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Recupera conceptos similares usando búsqueda semántica con expansión bilingüe.
        """
        variants = self._expand_query_variants(query)
        return self.retrieve_multi(variants, k=k)

    # ---------------------------
    # FallBack y validación
    # ---------------------------
    def _simple_search(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        """Búsqueda simple de texto como fallback"""
        results = []
        query_lower = query.lower()
        for concepto, narrativa in zip(self.conceptos, self.narrativas):
            if not str(concepto).isdigit():
                continue
            score = sum(1 for palabra in query_lower.split() if palabra in narrativa.lower())
            if score > 0:
                results.append((concepto, narrativa, 1.0 / (1.0 + score)))
        results.sort(key=lambda x: x[2])
        return results[:k]

    def _validate_index_coherence(self):
        """Valida que el índice sea IP (coseno), tamaños coincidan y el modelo sea el esperado."""
        if self.faiss_index is None:
            raise RuntimeError("Índice FAISS no cargado.")

        # (a) métrica = inner product (cosine con embeddings normalizados)
        try:
            metric = self.faiss_index.metric_type
        except Exception:
            metric = None

        if metric != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError(
                f"Índice FAISS con métrica no soportada ({metric}). "
                f"Reconstruye el índice con IndexFlatIP y embeddings L2-normalizados."
            )

        # (b) tamaños deben coincidir
        if self.faiss_index.ntotal != len(self.conceptos) or len(self.conceptos) != len(self.narrativas):
            raise RuntimeError(
                f"Inconsistencia: index.ntotal={self.faiss_index.ntotal}, "
                f"conceptos={len(self.conceptos)}, narrativas={len(self.narrativas)}."
            )

        # (c) modelo del índice debe coincidir
        metadata_path = os.path.join(self.assets_dir, 'ontology_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            idx_model = metadata.get('model_name')
            if idx_model != MODEL_NAME:
                raise RuntimeError(
                    f"El índice se construyó con '{idx_model}' y el runtime usa '{MODEL_NAME}'. "
                    f"Reconstruye el índice con el mismo modelo."
                )
