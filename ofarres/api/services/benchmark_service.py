"""
Benchmark Service

Handles NER and RAG evaluation logic.
Uses the actual NER models from backend/src/NER and metrics from backend/src/utils/metrics.py
"""

import json
import time
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

from ..models.schemas import (
    NERBenchmarkResponse,
    ModelBenchmarkResult,
    NoteMetrics,
    NERModelInfo,
    SequentialContribution,
    BenchmarkMode
)

# Add backend to path for imports
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))


# ==============================================================================
# METRICS FUNCTIONS (Ported from backend/src/utils/metrics.py)
# ==============================================================================

def calculate_iou(pred_span: Dict[str, int], gt_span: Dict[str, int]) -> float:
    """Calculate Intersection over Union for two spans."""
    p_start, p_end = pred_span.get('start'), pred_span.get('end')
    g_start, g_end = gt_span.get('start'), gt_span.get('end')

    if None in [p_start, p_end, g_start, g_end]:
        return 0.0

    intersection_start = max(p_start, g_start)
    intersection_end = min(p_end, g_end)
    intersection_length = max(0, intersection_end - intersection_start)

    pred_length = p_end - p_start
    gt_length = g_end - g_start
    union_length = pred_length + gt_length - intersection_length

    if union_length == 0:
        return 1.0 if intersection_length == 0 else 0.0

    return intersection_length / union_length


def text_containment_match(pred_text: str, gt_text: str) -> bool:
    """Check if there's text containment between prediction and GT."""
    pred_norm = pred_text.lower().strip()
    gt_norm = gt_text.lower().strip()
    return gt_norm in pred_norm or pred_norm in gt_norm


def get_detailed_matches(
    preds: List[Dict], 
    gt: List[Dict], 
    iou_thresh: float,
    note_text: str = ""
) -> Tuple[List, List, List]:
    """
    Matching Logic "Text Containment + IoU Overlap" (RAG-Ready Recall).
    
    Returns: (tp_pairs, fp_preds, fn_gts)
    """
    MIN_IOU_OVERLAP = 0.1
    
    tp_pairs = []
    matched_gt_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()
    
    # Ensure predictions have text
    preds_with_text = []
    for p in preds:
        p_copy = dict(p)
        if 'text' not in p_copy and note_text:
            p_copy['text'] = note_text[p_copy['start']:p_copy['end']]
        preds_with_text.append(p_copy)
    
    # Match GT to predictions
    for g_idx, g_item in enumerate(gt):
        best_match_score = -1.0
        best_pred_idx = None
        best_iou = 0.0
        
        gt_text = g_item.get('text', '')
        if not gt_text and note_text:
            gt_text = note_text[g_item['start']:g_item['end']]
        
        for p_idx, p in enumerate(preds_with_text):
            if p_idx in matched_pred_indices:
                continue
            
            iou = calculate_iou(p, g_item)
            
            if iou <= MIN_IOU_OVERLAP:
                continue
            
            pred_text = p.get('text', '')
            
            if not text_containment_match(pred_text, gt_text):
                continue
            
            if iou > best_match_score:
                best_match_score = iou
                best_pred_idx = p_idx
                best_iou = iou
        
        if best_pred_idx is not None:
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(best_pred_idx)
            tp_pairs.append((preds_with_text[best_pred_idx], g_item, best_iou))
    
    fn_gts = [gt[i] for i in range(len(gt)) if i not in matched_gt_indices]
    fp_preds = [preds_with_text[i] for i in range(len(preds_with_text)) if i not in matched_pred_indices]
    
    return tp_pairs, fp_preds, fn_gts


def calculate_pr_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, and F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def deduplicate_predictions(preds: List[Dict]) -> List[Dict]:
    """Remove exact duplicates (same start/end) keeping metadata."""
    seen = set()
    unique = []
    for p in preds:
        k = (p['start'], p['end'])
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique


# ==============================================================================
# NER WORKER LOADER
# ==============================================================================

def load_ner_worker(ner_id: str, config: Dict) -> Any:
    """Load a NER worker from the registry config."""
    try:
        mod = importlib.import_module(config['module'])
        cls = getattr(mod, config['class'])
        kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
        return cls(**kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to load {ner_id}: {e}")
        return None


# ==============================================================================
# BENCHMARK SERVICE
# ==============================================================================

class BenchmarkService:
    """Service for running NER and RAG benchmarks."""
    
    def __init__(self):
        self._backend_path = Path(__file__).parent.parent.parent / "backend"
        self._data_path = self._backend_path / "data"
        self._config_path = self._backend_path / "config" / "ner_registry.json"
        self._registry: Optional[Dict] = None
    
    def _load_registry(self) -> Dict:
        """Load NER model registry."""
        if self._registry is None:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._registry = json.load(f)
            else:
                self._registry = {}
        return self._registry
    
    def get_available_models(self) -> List[NERModelInfo]:
        """Get list of available NER models from registry."""
        registry = self._load_registry()
        
        models = []
        
        # Add models from registry only (no ground_truth baseline)
        for model_id, config in registry.items():
            # Try to load the model to check if it's available
            worker = load_ner_worker(model_id, config)
            models.append(NERModelInfo(
                id=model_id,
                name=config.get('class', model_id),
                description=f"Module: {config.get('module', 'unknown')}",
                available=worker is not None
            ))
        
        return models
    
    def _load_default_data(self) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
        """Load default notes and ground truth."""
        notes_path = self._data_path / "notes.json"
        gt_path = self._data_path / "ground_truth.json"
        
        if not notes_path.exists():
            raise FileNotFoundError(f"Notes file not found: {notes_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        with open(notes_path, 'r', encoding='utf-8') as f:
            notes_list = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
        
        notes_dict = {n['note_id']: n['text'] for n in notes_list}
        gt_dict = {g['note_id']: g['annotations'] for g in gt_list}
        
        return notes_dict, gt_dict
    
    def run_benchmark(
        self,
        mode: str = "all",
        model_id: Optional[str] = None,
        iou_threshold: float = 0.25,
        verbose: bool = False
    ) -> NERBenchmarkResponse:
        """
        Run NER benchmark.
        
        Args:
            mode: 'all' | 'assembly' | 'single'
            model_id: Required for 'single' mode
            iou_threshold: IoU threshold for matching
            verbose: Enable verbose output
        
        Returns:
            NERBenchmarkResponse with results
        """
        start_time = time.time()
        
        # Load data
        notes_dict, gt_dict = self._load_default_data()
        registry = self._load_registry()
        
        total_entities = sum(len(g) for g in gt_dict.values())
        
        # Determine which models to run
        if mode == "single":
            if model_id and model_id in registry:
                model_ids = [model_id]
            else:
                raise ValueError(f"Unknown model: {model_id}. Available: {list(registry.keys())}")
            benchmark_mode = BenchmarkMode.SINGLE
        elif mode == "assembly":
            model_ids = list(registry.keys())
            benchmark_mode = BenchmarkMode.ASSEMBLY
        else:  # mode == "all"
            model_ids = list(registry.keys())
            benchmark_mode = BenchmarkMode.ALL
        
        results: List[ModelBenchmarkResult] = []
        sequential_contribution: Optional[List[SequentialContribution]] = None
        
        if mode == "assembly":
            # Assembly mode: Run all models together
            result = self._run_assembly_benchmark(
                model_ids, registry, notes_dict, gt_dict, iou_threshold
            )
            results.append(result)
            
            # Calculate sequential contribution
            sequential_contribution = self._calculate_sequential_contribution(
                model_ids, registry, notes_dict, gt_dict, iou_threshold
            )
        else:
            # All or Single mode: Run each model individually
            for mid in model_ids:
                if mid not in registry:
                    continue
                result = self._run_single_model_benchmark(
                    mid, registry[mid], notes_dict, gt_dict, iou_threshold
                )
                if result:
                    results.append(result)
        
        # Sort by F1-Harmonic descending
        results.sort(key=lambda x: x.f1_harmonic, reverse=True)
        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        return NERBenchmarkResponse(
            mode=benchmark_mode,
            iou_threshold=iou_threshold,
            results=results,
            sequential_contribution=sequential_contribution,
            total_processing_time_ms=total_time_ms,
            notes_processed=len(notes_dict),
            total_entities=total_entities
        )
    
    def run_benchmark_stream(
        self,
        mode: str = "all",
        model_id: Optional[str] = None,
        iou_threshold: float = 0.25,
        verbose: bool = False
    ):
        """
        Generator that yields progress updates during benchmark execution.
        Uses actual timing for ETA calculation (like tqdm).
        
        Yields:
            dict with keys: type, current, total, percentage, eta_seconds, message, data (optional)
        """
        start_time = time.time()
        
        # Load data
        yield {
            "type": "status",
            "message": "Loading data...",
            "percentage": 0,
            "eta_seconds": None
        }
        
        notes_dict, gt_dict = self._load_default_data()
        registry = self._load_registry()
        
        total_entities = sum(len(g) for g in gt_dict.values())
        num_notes = len(notes_dict)
        
        # Determine which models to run
        if mode == "single":
            if model_id and model_id in registry:
                model_ids = [model_id]
            else:
                raise ValueError(f"Unknown model: {model_id}. Available: {list(registry.keys())}")
            benchmark_mode = BenchmarkMode.SINGLE
        elif mode == "assembly":
            model_ids = list(registry.keys())
            benchmark_mode = BenchmarkMode.ASSEMBLY
        else:  # mode == "all"
            model_ids = list(registry.keys())
            benchmark_mode = BenchmarkMode.ALL
        
        results: List[ModelBenchmarkResult] = []
        sequential_contribution: Optional[List[SequentialContribution]] = None
        
        yield {
            "type": "status",
            "message": f"Starting benchmark with mode={mode}, {len(model_ids)} model(s), {num_notes} notes",
            "percentage": 0,
            "eta_seconds": None
        }
        
        if mode == "assembly":
            # Assembly mode: Track progress across notes
            for progress_event in self._run_assembly_benchmark_stream(
                model_ids, registry, notes_dict, gt_dict, iou_threshold
            ):
                yield progress_event
            
            # Final result
            assembly_result = self._run_assembly_benchmark(
                model_ids, registry, notes_dict, gt_dict, iou_threshold
            )
            results.append(assembly_result)
            
            # Sequential contribution
            yield {
                "type": "status",
                "message": "Calculating sequential contribution...",
                "percentage": 95,
                "eta_seconds": None
            }
            sequential_contribution = self._calculate_sequential_contribution(
                model_ids, registry, notes_dict, gt_dict, iou_threshold
            )
        else:
            # All or Single mode: Track progress per model and note
            total_operations = len(model_ids) * num_notes
            completed_operations = 0
            operation_times = []  # Track time per operation for ETA
            
            for model_idx, mid in enumerate(model_ids):
                if mid not in registry:
                    continue
                
                yield {
                    "type": "model_start",
                    "message": f"Processing model: {mid}",
                    "current_model": mid,
                    "model_index": model_idx + 1,
                    "total_models": len(model_ids),
                    "percentage": int((completed_operations / total_operations) * 100),
                    "eta_seconds": self._calculate_eta(operation_times, total_operations - completed_operations)
                }
                
                # Run benchmark for this model with progress
                result = self._run_single_model_benchmark_stream(
                    mid, 
                    registry[mid], 
                    notes_dict, 
                    gt_dict, 
                    iou_threshold,
                    completed_operations,
                    total_operations,
                    operation_times
                )
                
                if result:
                    results.append(result["result"])
                    operation_times = result["operation_times"]
                    completed_operations = result["completed_operations"]
                    
                    yield {
                        "type": "model_complete",
                        "message": f"Completed model: {mid}",
                        "current_model": mid,
                        "model_index": model_idx + 1,
                        "total_models": len(model_ids),
                        "percentage": int((completed_operations / total_operations) * 100),
                        "eta_seconds": self._calculate_eta(operation_times, total_operations - completed_operations),
                        "result": result["result"].model_dump()
                    }
        
        # Sort results
        results.sort(key=lambda x: x.f1_harmonic, reverse=True)
        total_time_ms = int((time.time() - start_time) * 1000)
        
        final_response = NERBenchmarkResponse(
            mode=benchmark_mode,
            iou_threshold=iou_threshold,
            results=results,
            sequential_contribution=sequential_contribution,
            total_processing_time_ms=total_time_ms,
            notes_processed=num_notes,
            total_entities=total_entities
        )
        
        yield {
            "type": "complete",
            "message": "Benchmark complete!",
            "percentage": 100,
            "eta_seconds": 0,
            "data": final_response.model_dump()
        }
    
    def _calculate_eta(self, operation_times: List[float], remaining_ops: int) -> Optional[float]:
        """Calculate ETA based on average operation time (like tqdm)."""
        if not operation_times or remaining_ops <= 0:
            return None
        avg_time = sum(operation_times) / len(operation_times)
        return round(avg_time * remaining_ops, 1)
    
    def _run_single_model_benchmark_stream(
        self,
        model_id: str,
        config: Dict,
        notes_dict: Dict[str, str],
        gt_dict: Dict[str, List[Dict]],
        iou_threshold: float,
        completed_operations: int,
        total_operations: int,
        operation_times: List[float]
    ) -> Optional[Dict]:
        """Run benchmark for a single NER model, returning result plus timing info."""
        worker = load_ner_worker(model_id, config)
        if not worker:
            return None
        
        start_time = time.time()
        
        per_note_metrics = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        f1_harmonic_scores = []
        f1_arithmetic_scores = []
        
        for note_id, note_text in notes_dict.items():
            note_start = time.time()
            
            gt_annotations = gt_dict.get(note_id, [])
            
            try:
                pred_annotations = worker.extract_entities(note_text)
            except Exception as e:
                print(f"[ERROR] {model_id} failed on {note_id}: {e}")
                pred_annotations = []
            
            tp_pairs, fp_preds, fn_gts = get_detailed_matches(
                pred_annotations, gt_annotations, iou_threshold, note_text
            )
            
            tp = len(tp_pairs)
            fp = len(fp_preds)
            fn = len(fn_gts)
            
            metrics = calculate_pr_f1(tp, fp, fn)
            
            per_note_metrics.append(NoteMetrics(
                note_id=note_id,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                tp=tp, fp=fp, fn=fn
            ))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            f1_harmonic_scores.append(metrics['f1'])
            f1_arithmetic_scores.append((metrics['precision'] + metrics['recall']) / 2.0)
            
            # Track timing
            note_time = time.time() - note_start
            operation_times.append(note_time)
            completed_operations += 1
        
        global_metrics = calculate_pr_f1(total_tp, total_fp, total_fn)
        f1_harmonic = sum(f1_harmonic_scores) / len(f1_harmonic_scores) if f1_harmonic_scores else 0.0
        f1_arithmetic = sum(f1_arithmetic_scores) / len(f1_arithmetic_scores) if f1_arithmetic_scores else 0.0
        
        result = ModelBenchmarkResult(
            model_id=model_id,
            precision=global_metrics['precision'],
            recall=global_metrics['recall'],
            f1_micro=global_metrics['f1'],
            f1_macro=f1_harmonic,
            f1_harmonic=f1_harmonic,
            f1_arithmetic=f1_arithmetic,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            processing_time_s=time.time() - start_time,
            per_note_metrics=per_note_metrics
        )
        
        return {
            "result": result,
            "operation_times": operation_times,
            "completed_operations": completed_operations
        }
    
    def _run_assembly_benchmark_stream(
        self,
        model_ids: List[str],
        registry: Dict,
        notes_dict: Dict[str, str],
        gt_dict: Dict[str, List[Dict]],
        iou_threshold: float
    ):
        """Generator for assembly mode progress updates."""
        # Load all workers first
        yield {
            "type": "status",
            "message": "Loading models...",
            "percentage": 5,
            "eta_seconds": None
        }
        
        workers = {}
        for mid in model_ids:
            if mid in registry:
                w = load_ner_worker(mid, registry[mid])
                if w:
                    workers[mid] = w
        
        num_notes = len(notes_dict)
        operation_times = []
        
        for note_idx, (note_id, note_text) in enumerate(notes_dict.items()):
            note_start = time.time()
            
            # Process this note through all workers
            gt_annotations = gt_dict.get(note_id, [])
            all_preds = []
            
            for wid, worker in workers.items():
                try:
                    preds = worker.extract_entities(note_text)
                    for p in preds:
                        p['source'] = wid
                    all_preds.extend(preds)
                except Exception as e:
                    print(f"[ERROR] {wid} failed on {note_id}: {e}")
            
            note_time = time.time() - note_start
            operation_times.append(note_time)
            
            percentage = int(((note_idx + 1) / num_notes) * 90) + 5  # 5-95% range
            remaining = num_notes - note_idx - 1
            eta = self._calculate_eta(operation_times, remaining)
            
            yield {
                "type": "progress",
                "message": f"Processing note {note_idx + 1}/{num_notes}",
                "current": note_idx + 1,
                "total": num_notes,
                "percentage": percentage,
                "eta_seconds": eta
            }
    
    def _run_single_model_benchmark(
        self,
        model_id: str,
        config: Dict,
        notes_dict: Dict[str, str],
        gt_dict: Dict[str, List[Dict]],
        iou_threshold: float
    ) -> Optional[ModelBenchmarkResult]:
        """Run benchmark for a single NER model."""
        worker = load_ner_worker(model_id, config)
        if not worker:
            return None
        
        start_time = time.time()
        
        per_note_metrics = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        f1_harmonic_scores = []
        f1_arithmetic_scores = []
        
        for note_id, note_text in notes_dict.items():
            gt_annotations = gt_dict.get(note_id, [])
            
            # Extract entities using the NER model
            try:
                pred_annotations = worker.extract_entities(note_text)
            except Exception as e:
                print(f"[ERROR] {model_id} failed on {note_id}: {e}")
                pred_annotations = []
            
            tp_pairs, fp_preds, fn_gts = get_detailed_matches(
                pred_annotations, gt_annotations, iou_threshold, note_text
            )
            
            tp = len(tp_pairs)
            fp = len(fp_preds)
            fn = len(fn_gts)
            
            metrics = calculate_pr_f1(tp, fp, fn)
            
            per_note_metrics.append(NoteMetrics(
                note_id=note_id,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                tp=tp, fp=fp, fn=fn
            ))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            f1_harmonic_scores.append(metrics['f1'])
            f1_arithmetic_scores.append((metrics['precision'] + metrics['recall']) / 2.0)
        
        global_metrics = calculate_pr_f1(total_tp, total_fp, total_fn)
        f1_harmonic = sum(f1_harmonic_scores) / len(f1_harmonic_scores) if f1_harmonic_scores else 0.0
        f1_arithmetic = sum(f1_arithmetic_scores) / len(f1_arithmetic_scores) if f1_arithmetic_scores else 0.0
        
        return ModelBenchmarkResult(
            model_id=model_id,
            precision=global_metrics['precision'],
            recall=global_metrics['recall'],
            f1_micro=global_metrics['f1'],
            f1_macro=f1_harmonic,
            f1_harmonic=f1_harmonic,
            f1_arithmetic=f1_arithmetic,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            processing_time_s=time.time() - start_time,
            per_note_metrics=per_note_metrics
        )
    
    def _run_assembly_benchmark(
        self,
        model_ids: List[str],
        registry: Dict,
        notes_dict: Dict[str, str],
        gt_dict: Dict[str, List[Dict]],
        iou_threshold: float
    ) -> ModelBenchmarkResult:
        """Run benchmark with all models combined (ensemble)."""
        start_time = time.time()
        
        # Load all workers
        workers = {}
        for mid in model_ids:
            if mid in registry:
                w = load_ner_worker(mid, registry[mid])
                if w:
                    workers[mid] = w
        
        per_note_metrics = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        f1_harmonic_scores = []
        f1_arithmetic_scores = []
        
        for note_id, note_text in notes_dict.items():
            gt_annotations = gt_dict.get(note_id, [])
            
            # Collect predictions from all workers
            all_preds = []
            for wid, worker in workers.items():
                try:
                    preds = worker.extract_entities(note_text)
                    for p in preds:
                        p['source'] = wid
                    all_preds.extend(preds)
                except Exception as e:
                    print(f"[ERROR] {wid} failed on {note_id}: {e}")
            
            # Deduplicate
            pred_annotations = deduplicate_predictions(all_preds)
            
            tp_pairs, fp_preds, fn_gts = get_detailed_matches(
                pred_annotations, gt_annotations, iou_threshold, note_text
            )
            
            tp = len(tp_pairs)
            fp = len(fp_preds)
            fn = len(fn_gts)
            
            metrics = calculate_pr_f1(tp, fp, fn)
            
            per_note_metrics.append(NoteMetrics(
                note_id=note_id,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                tp=tp, fp=fp, fn=fn
            ))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            f1_harmonic_scores.append(metrics['f1'])
            f1_arithmetic_scores.append((metrics['precision'] + metrics['recall']) / 2.0)
        
        global_metrics = calculate_pr_f1(total_tp, total_fp, total_fn)
        f1_harmonic = sum(f1_harmonic_scores) / len(f1_harmonic_scores) if f1_harmonic_scores else 0.0
        f1_arithmetic = sum(f1_arithmetic_scores) / len(f1_arithmetic_scores) if f1_arithmetic_scores else 0.0
        
        return ModelBenchmarkResult(
            model_id="ASSEMBLY",
            precision=global_metrics['precision'],
            recall=global_metrics['recall'],
            f1_micro=global_metrics['f1'],
            f1_macro=f1_harmonic,
            f1_harmonic=f1_harmonic,
            f1_arithmetic=f1_arithmetic,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            processing_time_s=time.time() - start_time,
            per_note_metrics=per_note_metrics
        )
    
    def _calculate_sequential_contribution(
        self,
        model_ids: List[str],
        registry: Dict,
        notes_dict: Dict[str, str],
        gt_dict: Dict[str, List[Dict]],
        iou_threshold: float
    ) -> List[SequentialContribution]:
        """
        Calculate incremental contribution of each worker.
        How many NEW TPs does each model find that previous ones didn't?
        """
        MIN_IOU_OVERLAP = 0.1
        
        total_gt_count = sum(len(g) for g in gt_dict.values())
        if total_gt_count == 0:
            return []
        
        covered_gt_ids: Set[Tuple[str, int, int]] = set()
        contributions: List[SequentialContribution] = []
        cumulative_recall = 0.0
        
        for wid in model_ids:
            if wid not in registry:
                continue
            
            worker = load_ner_worker(wid, registry[wid])
            if not worker:
                continue
            
            new_tps_count = 0
            
            for note_id, note_text in notes_dict.items():
                try:
                    preds = worker.extract_entities(note_text)
                except:
                    preds = []
                
                gt_list = gt_dict.get(note_id, [])
                
                for gt_item in gt_list:
                    gt_uid = (note_id, gt_item['start'], gt_item['end'])
                    
                    if gt_uid in covered_gt_ids:
                        continue
                    
                    gt_text = gt_item.get('text', note_text[gt_item['start']:gt_item['end']])
                    
                    found = False
                    for p in preds:
                        iou = calculate_iou(p, gt_item)
                        if iou <= MIN_IOU_OVERLAP:
                            continue
                        
                        pred_text = note_text[p['start']:p['end']]
                        
                        if text_containment_match(pred_text, gt_text):
                            found = True
                            break
                    
                    if found:
                        covered_gt_ids.add(gt_uid)
                        new_tps_count += 1
            
            incremental_recall = new_tps_count / total_gt_count if total_gt_count > 0 else 0.0
            cumulative_recall += incremental_recall
            
            contributions.append(SequentialContribution(
                model_id=wid,
                incremental_recall=incremental_recall,
                cumulative_recall=cumulative_recall
            ))
        
        return contributions
