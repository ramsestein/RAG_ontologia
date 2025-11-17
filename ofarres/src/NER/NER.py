"""
Consolidated NER Module - RAG+GPT Pipeline in a Single File
Processes one clinical note at a time and returns SNOMED-CT coded entities

This module consolidates the entire RAG+GPT pipeline logic:
- NER: Entity extraction using GPT-4o with character offsets
- RAG: SNOMED-CT concept retrieval using FAISS/SapBERT
- Coding: SNOMED-CT code assignment with deterministic selection
- Span Matching: Exact text location with offset correction

Adapted from: original/benchmark/strategies/rag_gpt/
"""

import os
import sys
import json
import re
import pickle
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Third-party imports
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the NER pipeline"""
    
    # Paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # RAG_ontologia/
    ASSETS_DIR = PROJECT_ROOT / "original" / "benchmark" / "strategies" / "rag_gpt" / "assets" / "ontology"
    PROMPTS_DIR = PROJECT_ROOT / "original" / "benchmark" / "strategies" / "rag_gpt" / "src" / "prompts"
    
    # Model settings
    SAPBERT_MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
    
    # OpenAI settings
    OPENAI_MODEL = os.getenv("NER_LLM_MODEL", "gpt-4o")
    OPENAI_TEMPERATURE = float(os.getenv("NER_TEMPERATURE", "0.1"))
    OPENAI_MAX_TOKENS = int(os.getenv("NER_MAX_TOKENS", "4000"))
    OPENAI_TOP_P = float(os.getenv("NER_TOP_P", "0.9"))
    
    # RAG settings
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "50"))
    RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", "0.40"))
    RAG_MAX_DISPLAY = int(os.getenv("RAG_MAX_DISPLAY", "15"))
    RAG_QUERY_SUFFIX = os.getenv("RAG_QUERY_SUFFIX", "disorder finding")
    RAG_USE_LLM_VALIDATION = os.getenv("RAG_USE_LLM_VALIDATION", "true").lower() == "true"
    RAG_ALLOW_FALLBACK = os.getenv("RAG_ALLOW_FALLBACK", "false").lower() == "true"
    RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o")
    RAG_LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.0"))
    
    # Span processing
    SPAN_TIGHTEN = os.getenv("RAG_SPAN_TIGHTEN", "true").lower() == "true"
    
    # Evaluation offsets
    EVAL_OFFSET_BASE = int(os.getenv("EVAL_OFFSET_BASE", "0"))
    EVAL_END_INCLUSIVE = os.getenv("EVAL_END_INCLUSIVE", "false").lower() == "true"
    
    # SNOMED codes
    FALLBACK_CODE = "404684003"  # Clinical finding (generic)
    DEFAULT_ANATOMY = "12738006"  # Brain structure
    PRESENCE_MAP = {
        "presente": "52101004",   # Present
        "ausente": "272519000",   # Absent
        "incierto": "261665006"   # Unknown
    }
    
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment or api_keys file"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_file = cls.PROJECT_ROOT / "original" / "benchmark" / "strategies" / "rag_gpt" / "api_keys"
            try:
                with open(api_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("chatGPT="):
                            api_key = line.split("=", 1)[1].strip()
                            break
            except Exception as e:
                print(f"[CONFIG] Could not read {api_file}: {e}")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it as environment variable or in api_keys file.")
        return api_key


# ============================================================================
# TEXT UTILITIES
# ============================================================================

class TextUtils:
    """Text processing and span matching utilities"""
    
    PUNCT_TO_TRIM = set(list(' \t\r\n.,;:!?"\'()[]{}／/\\'))
    
    @staticmethod
    def clean_json_response(response: str) -> str:
        """Clean JSON response from markdown and trailing commas"""
        response_clean = response.strip()
        
        # Remove markdown
        if '```json' in response_clean:
            json_start = response_clean.find('```json') + 7
            json_end = response_clean.find('```', json_start)
            response_clean = response_clean[json_start:json_end].strip()
        elif '```' in response_clean:
            json_start = response_clean.find('```') + 3
            json_end = response_clean.find('```', json_start)
            response_clean = response_clean[json_start:json_end].strip()
        
        # Clean trailing commas
        response_clean = re.sub(r',(\s*[}\]])', r'\1', response_clean)
        
        return response_clean
    
    @staticmethod
    def tighten_span_boundaries(text: str, start: int, end: int) -> Tuple[int, int]:
        """Trim whitespace and punctuation from span boundaries"""
        if not isinstance(start, int) or not isinstance(end, int):
            return start, end
        s, e = max(0, start), min(len(text), end)
        
        # Trim left
        while s < e and (text[s].isspace() or text[s] in TextUtils.PUNCT_TO_TRIM):
            s += 1
        # Trim right
        while e > s and (text[e - 1].isspace() or text[e - 1] in TextUtils.PUNCT_TO_TRIM):
            e -= 1
        
        return s, e
    
    @staticmethod
    def find_exact_span(span_text: str, text: str) -> Optional[Tuple[int, int]]:
        """Find exact case-sensitive match"""
        if not span_text:
            return None
        idx = text.find(span_text)
        if idx == -1:
            return None
        return (idx, idx + len(span_text))
    
    @staticmethod
    def find_first_case_insensitive(span_text: str, text: str) -> Optional[Tuple[int, int]]:
        """Find first case-insensitive match"""
        if not span_text:
            return None
        idx = text.lower().find(span_text.lower())
        if idx == -1:
            return None
        return (idx, idx + len(span_text))
    
    @staticmethod
    def find_exact_span_near(span_text: str, text: str, approx_start: int, window: int = 50) -> Optional[Tuple[int, int]]:
        """Find exact match near an approximate offset"""
        if not span_text:
            return None
        start = max(0, approx_start - window)
        end = min(len(text), approx_start + window + len(span_text))
        
        # Case-sensitive first
        idx = text.find(span_text, start, end)
        if idx != -1:
            return (idx, idx + len(span_text))
        
        # Case-insensitive fallback
        lower_text = text.lower()
        lower_span = span_text.lower()
        idx2 = lower_text.find(lower_span, start, end)
        if idx2 != -1:
            return (idx2, idx2 + len(span_text))
        
        return None


# ============================================================================
# PROMPT LOADER
# ============================================================================

class PromptLoader:
    """Load prompts from JSON files"""
    
    @staticmethod
    def load_prompt(prompt_name: str) -> Dict:
        """Load a prompt JSON by name"""
        prompt_path = Config.PROMPTS_DIR / f"{prompt_name}.json"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @staticmethod
    def render_template(template: str, variables: Dict[str, Any]) -> str:
        """Safe template rendering without interpreting JSON braces"""
        s = template
        for k, v in variables.items():
            s = s.replace("{" + k + "}", str(v))
        return s


# ============================================================================
# NER EXTRACTOR
# ============================================================================

class NERExtractor:
    """Named Entity Recognition using GPT-4o"""
    
    def __init__(self, client: OpenAI, prompt_config: Dict, model_config: Dict, system_prompt: str):
        self.client = client
        self.prompt_template = prompt_config['template']
        self.model_config = model_config
        self.system_prompt = system_prompt
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities from text using GPT-4o"""
        print("[NER] Extracting entities with GPT-4o...")
        
        prompt = PromptLoader.render_template(self.prompt_template, {"informe": text})
        response = self._call_gpt4o(prompt)
        entities = self._parse_ner_response(response)
        
        print(f"[NER] Detected {len(entities)} entities")
        for i, ent in enumerate(entities[:3]):
            print(f"  - Span: \"{ent['span_text']}\"")
        if len(entities) > 3:
            print(f"  ... and {len(entities) - 3} more")
        
        return entities
    
    def _call_gpt4o(self, prompt: str, max_retries: int = 3) -> str:
        """Call GPT-4o with error handling"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_config["model"],
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config.get("temperature", 0.1),
                    max_tokens=self.model_config.get("max_tokens", 4000),
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[NER] Error calling GPT-4o (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return '{"entities": []}'
        
        return '{"entities": []}'
    
    def _strip_control_chars(self, s: str) -> str:
        """Remove non-printable control characters"""
        return ''.join(ch for ch in s if ch in '\t\n\r' or ord(ch) >= 32)
    
    def _extract_top_level_json(self, s: str) -> Optional[str]:
        """Extract first balanced JSON object, ignoring extra text"""
        s = TextUtils.clean_json_response(self._strip_control_chars(s))
        s = s.rstrip('.… \n\r\t')
        
        start = s.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
        
        if depth > 0:
            return s[start:] + ('}' * depth)
        return None
    
    def _repair_entities_array(self, text: str) -> str:
        """Repair separators and close entities array"""
        text = re.sub(r'}\s*{', '}, {', text)
        
        ent_open = text.find('"entities"')
        if ent_open != -1:
            arr_open = text.find('[', ent_open)
            if arr_open != -1:
                arr_close = text.find(']', arr_open)
                if arr_close == -1:
                    last_brace = text.rfind('}')
                    if last_brace != -1 and last_brace > arr_open:
                        text = text[:last_brace] + ']' + text[last_brace:]
        return text
    
    def _safe_int(self, x):
        """Safely convert to int"""
        if x is None:
            return None
        try:
            if isinstance(x, str):
                x = x.strip()
                if not re.match(r'^-?\d+(\.0+)?$', x):
                    return None
            return int(float(x))
        except Exception:
            return None
    
    def _normalize_entity(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize entity fields"""
        if not isinstance(finding, dict):
            return None
        core_entity = (finding.get("core_entity") or finding.get("entity") or "").strip()
        full_span = (finding.get("full_span") or core_entity or "").strip()
        if not full_span:
            return None
        anatomical_location = (finding.get("anatomical_location") or "No especificado") or "No especificado"
        presence = (finding.get("presence") or "presente") or "presente"
        value = finding.get("value", None)
        start_int = self._safe_int(finding.get("start", None))
        end_int = self._safe_int(finding.get("end", None))
        return {
            "span_text": core_entity if core_entity else full_span,
            "full_span": full_span,
            "anatomical_location": anatomical_location,
            "presence": presence,
            "value": value,
            "start": start_int,
            "end": end_int
        }
    
    def _parse_ner_response(self, response: str) -> List[Dict]:
        """Parse NER JSON response with error tolerance"""
        try:
            candidate = self._extract_top_level_json(response)
            if not candidate:
                raise ValueError("No JSON object found in response")
            
            candidate = self._repair_entities_array(candidate)
            
            try:
                data = json.loads(candidate)
                entities_raw = data.get("entities", [])
                if not isinstance(entities_raw, list):
                    for v in data.values():
                        if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                            entities_raw = v
                            break
                entities = []
                for finding in entities_raw:
                    ent = self._normalize_entity(finding)
                    if ent:
                        entities.append(ent)
                return entities
            except json.JSONDecodeError:
                # Retry with simple repairs
                candidate2 = re.sub(r',(\s*[}\]])', r'\1', candidate)
                candidate2 = re.sub(r'}\s*{', '}, {', candidate2)
                try:
                    data = json.loads(candidate2)
                    entities_raw = data.get("entities", [])
                    if not isinstance(entities_raw, list):
                        for v in data.values():
                            if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                                entities_raw = v
                                break
                    entities = []
                    for finding in entities_raw:
                        ent = self._normalize_entity(finding)
                        if ent:
                            entities.append(ent)
                    return entities
                except json.JSONDecodeError:
                    print(f"[NER] Failed to parse JSON, returning empty list")
                    return []
        except Exception as e:
            print(f"[NER] Error parsing response: {e}")
            print(f"[NER] Response: {response[:500]}...")
            return []


# ============================================================================
# RAG RETRIEVER
# ============================================================================

class RAGRetriever:
    """FAISS-based semantic retrieval using SapBERT embeddings"""
    
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.faiss_index = None
        self.conceptos = []
        self.narrativas = []
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        self._load_model_and_tokenizer()
        self._load_index()
        self._load_ontology()
        
        print("[RAG] Retrieval system initialized")
    
    def _load_model_and_tokenizer(self):
        """Load SapBERT model and tokenizer"""
        try:
            print(f"[RAG] Loading model: {Config.SAPBERT_MODEL}...")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.SAPBERT_MODEL)
            self.model = AutoModel.from_pretrained(Config.SAPBERT_MODEL).to(self.device)
            self.model.eval()
            print(f"[RAG] Model loaded on {self.device}")
        except Exception as e:
            print(f"[RAG] Error loading model: {e}")
    
    def _load_index(self):
        """Load pre-built FAISS index"""
        index_path = self.assets_dir / 'ontology.index'
        if not index_path.exists():
            print(f"[RAG] WARNING: FAISS index not found at {index_path}")
            return
        
        try:
            print(f"[RAG] Loading FAISS index...")
            self.faiss_index = faiss.read_index(str(index_path))
            
            metadata_path = self.assets_dir / 'ontology_metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"[RAG] Index loaded: {metadata['n_concepts']} concepts")
        except Exception as e:
            print(f"[RAG] Error loading index: {e}")
    
    def _load_ontology(self):
        """Load concepts and narratives from pickle"""
        concepts_path = self.assets_dir / 'ontology_concepts.pkl'
        narratives_path = self.assets_dir / 'ontology_narratives.pkl'
        
        try:
            with open(concepts_path, 'rb') as f:
                self.conceptos = pickle.load(f)
            with open(narratives_path, 'rb') as f:
                self.narrativas = pickle.load(f)
            print(f"[RAG] Ontology loaded: {len(self.conceptos)} concepts")
        except Exception as e:
            print(f"[RAG] Error loading ontology: {e}")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Generate normalized embedding with mean pooling"""
        text = query if isinstance(query, str) else str(query)
        with torch.no_grad():
            toks = self.tokenizer.encode_plus(
                text,
                padding=True,
                max_length=64,
                truncation=True,
                return_tensors="pt"
            )
            toks_on_device = {k: v.to(self.device) for k, v in toks.items()}
            
            outputs = self.model(**toks_on_device)
            last_hidden = outputs.last_hidden_state
            mask = toks_on_device["attention_mask"].unsqueeze(-1)
            
            # Mean pooling
            sum_vec = (last_hidden * mask).sum(dim=1)
            len_vec = mask.sum(dim=1).clamp(min=1)
            mean_vec = sum_vec / len_vec
            
            emb = mean_vec.cpu().numpy()
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            normalized_emb = emb / np.clip(norm, 1e-12, None)
            
            return normalized_emb.astype("float32")
    
    def search(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        """Search FAISS for similar concepts"""
        if self.faiss_index is None or self.model is None:
            print("[RAG] WARNING: RAG system not available")
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
            print(f"[RAG] Error in search: {e}")
            return []
    
    def retrieve_multi(self, queries: List[str], k: int = 5) -> List[Tuple[str, str, float]]:
        """Execute multiple queries and merge results by maximum similarity"""
        if not queries:
            return []
        
        pool = {}
        for q in queries:
            res = self.search(q, k)
            for concepto, narrativa, sim in res:
                prev = pool.get(concepto)
                if (prev is None) or (sim > prev[1]):
                    pool[concepto] = (narrativa, sim)
        
        fused = [(c, n, s) for c, (n, s) in pool.items()]
        fused.sort(key=lambda x: x[2], reverse=True)
        return fused[:k]


# ============================================================================
# SNOMED CODER
# ============================================================================

class SNOMEDCoder:
    """SNOMED-CT code assignment with deterministic selection"""
    
    def __init__(self, rag_retriever: RAGRetriever, openai_client: OpenAI, system_prompt: str):
        self.rag = rag_retriever
        self.client = openai_client
        self.system_prompt = system_prompt
        
        # Load prompts
        self.prompt_config = PromptLoader.load_prompt("coding")
    
    def code_entities(self, entities: List[Dict], verbose: bool = True) -> List[Dict]:
        """Code entities using SNOMED-CT"""
        if verbose:
            print(f"[CODING] Coding {len(entities)} entities...")
        
        coded_entities = []
        for entity in entities:
            codes = self.assign_codes(
                entity=entity['span_text'],
                location=entity.get('anatomical_location', 'No especificado'),
                presence=entity.get('presence', 'presente'),
                verbose=verbose
            )
            coded_entity = {**entity, **codes}
            coded_entities.append(coded_entity)
        
        return coded_entities
    
    def assign_codes(self, entity: str, location: str, presence: str, verbose: bool = False) -> dict:
        """Assign SNOMED-CT codes with deterministic selection"""
        # 1. Retrieve candidates
        ent_results = self._retrieve_candidates(query=entity, context_type="ENTITY", verbose=verbose)
        anat_results = self._retrieve_candidates(query=location, context_type="ANATOMY", verbose=verbose) \
            if location and location != "No especificado" else []
        
        # 2. Deterministic selection (top-1 or default)
        entity_code = self._pick_top_code(ent_results, Config.RAG_THRESHOLD)
        if entity_code is None and Config.RAG_ALLOW_FALLBACK:
            entity_code = Config.FALLBACK_CODE
        anatomy_code = self._pick_top_code(anat_results, Config.RAG_THRESHOLD) or Config.DEFAULT_ANATOMY
        presence_code = Config.PRESENCE_MAP.get(str(presence).lower(), Config.PRESENCE_MAP["presente"])
        
        # 3. Optional LLM validation (restricted to candidates)
        if Config.RAG_USE_LLM_VALIDATION:
            if verbose:
                print(f"[CODING]   -> LLM validation with {Config.RAG_LLM_MODEL}...")
            
            contexto_entity = self._format_context("ENTITY", entity, ent_results)
            contexto_anatomy = self._format_context("ANATOMY", location, anat_results) if anat_results else "--- ANATOMY NOT SPECIFIED ---\n"
            valid_entity_list = [c for c, _, _ in ent_results]
            valid_anatomy_list = [c for c, _, _ in anat_results]
            
            prompt = PromptLoader.render_template(
                self.prompt_config["template"],
                {
                    "entity": entity,
                    "location": location,
                    "presence": presence,
                    "contexto_entity": contexto_entity,
                    "contexto_anatomy": contexto_anatomy,
                    "valid_entity_codes": json.dumps(valid_entity_list, ensure_ascii=False),
                    "valid_anatomy_codes": json.dumps(valid_anatomy_list, ensure_ascii=False)
                }
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=Config.RAG_LLM_MODEL,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=Config.RAG_LLM_TEMPERATURE,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
                
                # Restrict to candidates
                proposed_entity = str(result.get("entity_code", entity_code or "")).strip()
                proposed_anatomy = str(result.get("anatomy_code", anatomy_code)).strip()
                proposed_presence = str(result.get("presence_code", presence_code)).strip()
                
                if valid_entity_list:
                    if proposed_entity.isdigit() and proposed_entity in valid_entity_list:
                        entity_code = proposed_entity
                else:
                    if proposed_entity.isdigit():
                        entity_code = proposed_entity
                
                if valid_anatomy_list:
                    if proposed_anatomy.isdigit() and proposed_anatomy in valid_anatomy_list:
                        anatomy_code = proposed_anatomy
                else:
                    if proposed_anatomy.isdigit():
                        anatomy_code = proposed_anatomy
                
                if proposed_presence.isdigit():
                    presence_code = proposed_presence
            
            except Exception as e:
                if verbose:
                    print(f"[CODING]   WARNING: LLM validation error: {e}")
        
        # Validate codes
        if entity_code is not None and not str(entity_code).isdigit():
            if verbose:
                print(f"[CODING]   WARNING: Invalid entity_code: '{entity_code}', discarding")
            entity_code = None
        
        if not str(anatomy_code).isdigit():
            if verbose:
                print(f"[CODING]   WARNING: Invalid anatomy_code: '{anatomy_code}', using default")
            anatomy_code = Config.DEFAULT_ANATOMY
        
        if not str(presence_code).isdigit():
            if verbose:
                print(f"[CODING]   WARNING: Invalid presence_code: '{presence_code}', using default")
            presence_code = Config.PRESENCE_MAP["presente"]
        
        if verbose:
            print(f"[CODING]   OK: entity={entity_code if entity_code else '∅'}, anatomy={anatomy_code}, presence={presence_code}")
        
        return {
            "entity_code": str(entity_code) if entity_code else "",
            "anatomy_code": str(anatomy_code),
            "presence_code": str(presence_code)
        }
    
    def _retrieve_candidates(self, query: str, context_type: str, verbose: bool) -> List[Tuple[str, str, float]]:
        """Retrieve, deduplicate and filter candidates by threshold"""
        if not query or query == "No especificado":
            return []
        
        TOP_K = Config.RAG_TOP_K
        THRESHOLD = Config.RAG_THRESHOLD
        
        if context_type == "ENTITY":
            # Heuristic: don't add clinical suffix for procedures/scales/scores
            q = (query or "").lower()
            looks_like_proc_or_score = bool(re.search(
                r'\b(angiograph|thrombect|coiling|endarterect|angioplast|stent|tici|nihss|aspects|gcs|mrs|rankin|score|scale)\b',
                q
            ))
            
            results_main = self.rag.search(query, k=TOP_K)
            results_clinical = []
            if Config.RAG_QUERY_SUFFIX and not looks_like_proc_or_score:
                query_clinical = f"{query} {Config.RAG_QUERY_SUFFIX}".strip()
                results_clinical = self.rag.search(query_clinical, k=TOP_K)
            
            combined = {}
            for concepto, narrativa, sim in (results_main + results_clinical):
                if concepto not in combined or sim > combined[concepto][1]:
                    combined[concepto] = (narrativa, sim)
            
            results = [(c, n, s) for c, (n, s) in combined.items()]
        else:
            results = self.rag.search(query, k=min(TOP_K, 15))
        
        # Filter and sort
        filtered = [(c, n, s) for c, n, s in results if s >= THRESHOLD]
        filtered.sort(key=lambda x: x[2], reverse=True)
        
        if verbose:
            if filtered:
                best_code, _, best_sim = filtered[0]
                print(f"[CODING]   -> RAG {context_type}: {len(filtered)} concepts (best: {best_code}, sim: {best_sim:.3f})")
            else:
                print(f"[CODING]   -> RAG {context_type}: 0 results (sim < {THRESHOLD})")
        
        return filtered
    
    def _pick_top_code(self, results: List[Tuple[str, str, float]], threshold: float) -> Optional[str]:
        """Pick top-1 code if exists and exceeds threshold"""
        if not results:
            return None
        best_code, _, best_sim = results[0]
        return best_code if best_sim >= threshold and str(best_code).isdigit() else None
    
    def _format_context(self, context_type: str, query: str, results: List[Tuple[str, str, float]]) -> str:
        """Build context block for LLM"""
        if not results:
            return f"--- {context_type} CODES for '{query}' ---\n--- NO CODES FOUND ---\n"
        
        MAX_DISPLAY = Config.RAG_MAX_DISPLAY
        context = f"\n--- {context_type} CODES for '{query}' ---\n"
        for idx, (concepto, narrativa, sim) in enumerate(results[:MAX_DISPLAY], 1):
            context += f"OPTION {idx} [SIM: {sim:.2f}]: CODE: {concepto} | {narrativa[:120]}\n"
        return context


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class NERPipeline:
    """
    Complete NER pipeline for processing one clinical note at a time
    
    Pipeline stages:
    1. Text normalization (CRLF -> LF) with offset mapping
    2. NER: Entity extraction with GPT-4o (returns entities with offsets)
    3. Deduplication (by occurrence)
    4. RAG + Coding: SNOMED-CT code assignment
    5. Span matching: Exact text location
    6. Offset remapping to original text
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
        if verbose:
            print("=" * 80)
            print("NER Pipeline - Initialization")
            print("=" * 80)
        
        # Initialize OpenAI client
        api_key = Config.get_openai_api_key()
        self.client = OpenAI(api_key=api_key)
        
        # Load prompts
        ner_prompt = PromptLoader.load_prompt("ner_v5")
        system_prompt_data = PromptLoader.load_prompt("system")
        system_prompt = system_prompt_data["content"]
        
        # Initialize components
        self.rag = RAGRetriever(Config.ASSETS_DIR)
        self.ner = NERExtractor(
            self.client,
            ner_prompt,
            {
                "model": Config.OPENAI_MODEL,
                "temperature": Config.OPENAI_TEMPERATURE,
                "max_tokens": Config.OPENAI_MAX_TOKENS,
                "top_p": Config.OPENAI_TOP_P
            },
            system_prompt=system_prompt
        )
        self.coder = SNOMEDCoder(self.rag, self.client, system_prompt=system_prompt)
        
        # Exclusion regex for noise
        self.exclude_re = re.compile(
            r"(?i)\b("
            r"nihss|aspects|tici|gcs|mrs|modified\s+rankin|"
            r"score|scale|report|unit|neurointensive|"
            r"pattern of infarction|small infarct volume|"
            r"collateral score|collateral circulation"
            r")\b"
        )
        
        if verbose:
            print("[OK] Pipeline initialized successfully")
            print("=" * 80)
    
    def process_note(self, text: str, note_id: str = None) -> List[Dict]:
        """
        Process a single clinical note
        
        Args:
            text: Clinical note text
            note_id: Optional note identifier for logging
            
        Returns:
            List of entity dictionaries with SNOMED-CT codes and offsets
        """
        if self.verbose and note_id:
            print(f"\n{'=' * 80}")
            print(f"Processing note {note_id}")
            print(f"{'=' * 80}")
        
        # Text normalization with offset mapping
        original_text = text
        text_norm, mapping = self._normalize_text_with_mapping(original_text)
        text = text_norm
        
        # Chunking (always, even for small texts)
        chunks = self._chunk_text(text)
        
        # NER extraction
        all_entities = []
        for i, (chunk, base) in enumerate(chunks):
            if self.verbose and len(chunks) > 1:
                print(f"\n[NER] Processing chunk {i + 1}/{len(chunks)} (base={base})...")
            
            chunk_entities = self.ner.extract_entities(chunk)
            
            # Adjust offsets from chunk to document
            for e in chunk_entities:
                if isinstance(e.get("start"), int):
                    e["start"] += base
                if isinstance(e.get("end"), int):
                    e["end"] += base
            
            all_entities.extend(chunk_entities)
        
        if not all_entities:
            if self.verbose:
                print("[WARNING] No entities detected")
            return []
        
        # Deduplication
        entities = self._deduplicate_entities(all_entities)
        
        # RAG + Coding
        coded_entities = self.coder.code_entities(entities, verbose=self.verbose)
        
        # Span matching (on normalized text)
        final_entities_norm = self._locate_spans(coded_entities, text)
        
        # Remap offsets to original text
        final_entities = []
        for ent in final_entities_norm:
            s_norm = ent["start"]
            e_norm = ent["end"]
            s_orig, e_orig = self._map_norm_span_to_original(s_norm, e_norm, mapping, len(original_text))
            
            ent_out = dict(ent)
            ent_out["start"] = s_orig
            ent_out["end"] = e_orig
            ent_out["span_text_real"] = original_text[s_orig:e_orig]
            final_entities.append(ent_out)
        
        if self.verbose:
            print(f"\n[OK] Processing complete: {len(final_entities)} entities")
        
        return final_entities
    
    def _normalize_text_with_mapping(self, original_text: str) -> Tuple[str, List[int]]:
        """Normalize CRLF/CR to LF with offset mapping"""
        norm_chars = []
        mapping = []
        
        i = 0
        n = len(original_text)
        while i < n:
            ch = original_text[i]
            if ch == "\r":
                if i + 1 < n and original_text[i + 1] == "\n":
                    norm_chars.append("\n")
                    mapping.append(i)
                    i += 2
                else:
                    norm_chars.append("\n")
                    mapping.append(i)
                    i += 1
            else:
                norm_chars.append(ch)
                mapping.append(i)
                i += 1
        
        normalized = "".join(norm_chars)
        return normalized, mapping
    
    def _map_norm_span_to_original(self, s_norm: int, e_norm: int, mapping: List[int], orig_len: int) -> Tuple[int, int]:
        """Convert normalized offsets to original text offsets"""
        if not mapping:
            return s_norm, e_norm
        
        # Start
        if s_norm < 0:
            s_norm = 0
        if s_norm >= len(mapping):
            start_orig = orig_len
        else:
            start_orig = mapping[s_norm]
        
        # End
        if e_norm <= 0:
            end_orig = 0
        elif e_norm - 1 >= len(mapping):
            end_orig = orig_len
        else:
            end_orig = mapping[e_norm - 1] + 1
        
        if end_orig < start_orig:
            end_orig = start_orig
        return start_orig, end_orig
    
    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split text into chunks with overlap"""
        chunk_size = 3000
        overlap = 300
        
        if len(text) <= chunk_size:
            return [(text, 0)]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append((text[start:end], start))
            if end >= len(text):
                break
            start = end - overlap
        
        if self.verbose:
            print(f"[CHUNKING] {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
        
        return chunks
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicates (same occurrence only)"""
        seen = set()
        unique = []
        
        for e in entities:
            start = e.get("start")
            end = e.get("end")
            key = (
                e.get("full_span", e["span_text"]),
                e.get("anatomical_location", ""),
                e.get("presence", ""),
                start if isinstance(start, int) else None,
                end if isinstance(end, int) else None,
            )
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        if self.verbose and len(entities) > len(unique):
            print(f"[DEDUP] {len(entities)} -> {len(unique)} entities (by occurrence)")
        
        return unique
    
    def _append_located(self, located_entities: List[Dict], base_entity: Dict, text: str, s: int, e: int):
        """Add located entity with optional span tightening"""
        if Config.SPAN_TIGHTEN:
            s, e = TextUtils.tighten_span_boundaries(text, s, e)
        ent = dict(base_entity)
        ent["start"] = s
        ent["end"] = e
        ent["span_text_real"] = text[s:e]
        located_entities.append(ent)
    
    def _locate_spans(self, entities: List[Dict], text: str) -> List[Dict]:
        """Locate exact spans in text with strict matching policy"""
        located_entities = []
        
        for entity in entities:
            core_entity = entity["span_text"]
            full_span = (entity.get("full_span") or core_entity) or core_entity
            
            start = entity.get("start")
            end = entity.get("end")
            
            # Use provided offsets if valid and exact
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                snippet = text[start:end]
                if snippet == full_span:
                    self._append_located(located_entities, entity, text, start, end)
                    continue
                
                # Try to correct near offset
                nearby = TextUtils.find_exact_span_near(full_span, text, approx_start=start, window=80)
                if nearby:
                    s2, e2 = nearby
                    self._append_located(located_entities, entity, text, s2, e2)
                    continue
                
                # Try with core_entity
                nearby_core = TextUtils.find_exact_span_near(core_entity, text, approx_start=start, window=80)
                if nearby_core:
                    s3, e3 = nearby_core
                    self._append_located(located_entities, entity, text, s3, e3)
                    continue
                
                # Global exact match
                global_match = TextUtils.find_exact_span(full_span, text)
                if global_match:
                    s4, e4 = global_match
                    self._append_located(located_entities, entity, text, s4, e4)
                    continue
                
                # Case-insensitive fallback
                ci = TextUtils.find_first_case_insensitive(full_span, text)
                if ci:
                    s5, e5 = ci
                    self._append_located(located_entities, entity, text, s5, e5)
                    continue
                
                if self.verbose:
                    print(f"[SPAN] Discarding entity (unreliable offsets, no exact match): '{full_span[:40]}'")
                continue
            
            # No offsets -> search for unique exact match
            exact_global = TextUtils.find_exact_span(full_span, text)
            if exact_global:
                s, e = exact_global
                self._append_located(located_entities, entity, text, s, e)
                continue
            
            # Case-insensitive fallback
            ci_global = TextUtils.find_first_case_insensitive(full_span, text)
            if ci_global:
                s2, e2 = ci_global
                self._append_located(located_entities, entity, text, s2, e2)
                continue
            
            if self.verbose:
                print(f"[SPAN] No offsets and no exact match: discarding '{full_span[:40]}'")
        
        return located_entities
    
    def _is_excluded(self, ent: dict) -> bool:
        """Check if entity should be excluded (noise)"""
        txt = (
            str(ent.get("full_span") or "") + " " +
            str(ent.get("span_text") or "")
        )
        return bool(self.exclude_re.search(txt))


# ============================================================================
# PUBLIC API
# ============================================================================

def extract_entities_from_note(note_text: str, note_id: str = None, verbose: bool = True) -> List[Dict]:
    """
    Extract SNOMED-CT coded entities from a single clinical note
    
    Args:
        note_text: The clinical note text
        note_id: Optional identifier for logging
        verbose: Whether to print detailed logs
        
    Returns:
        List of entity dictionaries with fields:
        - start: character offset (0-based, inclusive)
        - end: character offset (0-based, exclusive)
        - span_text: original entity text
        - span_text_real: exact text from document
        - entity_code: SNOMED-CT code
        - anatomy_code: SNOMED-CT anatomy code
        - presence_code: SNOMED-CT presence code
        - anatomical_location: location description
        - presence: presence status (presente/ausente/incierto)
    """
    pipeline = NERPipeline(verbose=verbose)
    entities = pipeline.process_note(note_text, note_id=note_id)
    
    # Filter out excluded entities
    filtered = []
    for ent in entities:
        # Filter negated/uncertain and generic fallback
        pres_text = str(ent.get("presence", "")).strip().lower()
        if pres_text in {"ausente", "incierto"}:
            continue
        
        concept_id = str(ent.get("entity_code", "")).strip()
        if not concept_id or concept_id == Config.FALLBACK_CODE:
            continue
        
        # Filter noise
        if pipeline._is_excluded(ent):
            continue
        
        filtered.append(ent)
    
    return filtered


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Process a single note
    sample_note = """
    HOSPITAL COURSE: This is a 67-year-old right-handed male with a history of hypertension 
    and diabetes mellitus who presented with acute onset of left-sided weakness and speech 
    difficulties. The patient was last seen normal 3 hours prior to presentation.
    """
    
    entities = extract_entities_from_note(sample_note, note_id="test_1", verbose=True)
    
    print(f"\n{'=' * 80}")
    print(f"Extracted {len(entities)} entities:")
    for i, ent in enumerate(entities, 1):
        print(f"{i}. [{ent['start']}-{ent['end']}] {ent['span_text_real']} -> {ent['entity_code']}")
    print(f"{'=' * 80}")
