# src/utils/metrics.py

from typing import List, Dict, Set, Tuple

# ==============================================================================
# 1. HERRAMIENTAS DE SPAN (EL CÁLCULO DE IoU)
# (Esto se queda igual)
# ==============================================================================

def calculate_iou(pred_span: Dict[str, int], gt_span: Dict[str, int]) -> float:
    """
    Calcula la métrica Intersection over Union (IoU) para dos spans.
    Cada span es un dict {'start': int, 'end': int}.
    """
    p_start, p_end = pred_span.get('start'), pred_span.get('end')
    g_start, g_end = gt_span.get('start'), gt_span.get('end')

    # Manejar offsets faltantes (común en predicciones de NER)
    if None in [p_start, p_end, g_start, g_end]:
        return 0.0

    # Calcular la intersección (el solapamiento)
    intersection_start = max(p_start, g_start)
    intersection_end = min(p_end, g_end)
    intersection_length = max(0, intersection_end - intersection_start)

    # Calcular la unión (el área total cubierta por ambos)
    pred_length = p_end - p_start
    gt_length = g_end - g_start
    union_length = pred_length + gt_length - intersection_length

    if union_length == 0:
        return 1.0 if intersection_length == 0 else 0.0

    return intersection_length / union_length


# ==============================================================================
# 2. HELPER MATEMÁTICO (P, R, F1)
# (Esto se queda igual, es reutilizable)
# ==============================================================================

def _calculate_pr_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Helper matemático para calcular Precisión, Recall y F1."""
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ==============================================================================
# 3. MÉTRICAS SOTA (PIPELINE COMPLETO: NER + Coding)
# (Estas son tus funciones actuales. Miden concept_id + IoU)
# ==============================================================================

def _find_strict_matches(predictions: List[Dict], 
                         ground_truth: List[Dict], 
                         iou_threshold: float = 0.5
                         ) -> Tuple[int, int, int]:
    """
    MATCHING SOTA (COMPLETO): Un "acierto" requiere concept_id Y IoU > threshold.
    Devuelve: (TP, FP, FN)
    """
    tp = 0
    matched_gt_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()

    for pred_idx, pred in enumerate(predictions):
        best_match_gt_idx = -1
        best_iou = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue

            # --- COINCIDENCIA DE CÓDIGO ---
            # (En el pipeline completo, la predicción tiene 'entity_code')
            pred_code = str(pred.get('entity_code'))
            gt_code = str(gt.get('concept_id'))

            if pred_code != gt_code:
                continue

            # --- COINCIDENCIA DE SPAN (IoU) ---
            iou = calculate_iou(pred, gt)
            
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match_gt_idx = gt_idx
        
        if best_match_gt_idx != -1:
            tp += 1
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(best_match_gt_idx)

    fp = len(predictions) - len(matched_pred_indices)
    fn = len(ground_truth) - len(matched_gt_indices)
    
    return tp, fp, fn

def calculate_macro_average_f1(all_predictions: Dict[str, List[Dict]], 
                               all_ground_truth: Dict[str, List[Dict]], 
                               iou_threshold: float = 0.5
                               ) -> Dict[str, float]:
    """
    Calcula el F1-Score SOTA "Macro-Average" (tu idea de "suma/5").
    Mide el F1 Estricto (COMPLETO) para cada nota y luego hace la media aritmética.
    """
    f1_scores_per_note = []
    
    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        
        tp_i, fp_i, fn_i = _find_strict_matches(
            pred_annotations, 
            gt_annotations, 
            iou_threshold
        )
        
        metrics_i = _calculate_pr_f1(tp_i, fp_i, fn_i)
        f1_scores_per_note.append(metrics_i['f1'])

    if not f1_scores_per_note:
        return {"macro_f1": 0.0}
        
    macro_f1 = sum(f1_scores_per_note) / len(f1_scores_per_note)
    
    return {"macro_f1": macro_f1}

def calculate_micro_average_f1(all_predictions: Dict[str, List[Dict]], 
                               all_ground_truth: Dict[str, List[Dict]], 
                               iou_threshold: float = 0.5
                               ) -> Dict[str, float]:
    """
    Calcula el F1-Score SOTA "Micro-Average" (COMPLETO) como media aritmética.
    Calcula F1 para cada nota y luego hace la media aritmética: (x1+x2+...+xn)/n.
    """
    f1_scores_per_note = []

    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        
        tp_i, fp_i, fn_i = _find_strict_matches(
            pred_annotations, 
            gt_annotations, 
            iou_threshold
        )
        
        metrics_i = _calculate_pr_f1(tp_i, fp_i, fn_i)
        f1_scores_per_note.append(metrics_i['f1'])

    if not f1_scores_per_note:
        return {"micro_f1": 0.0}
        
    micro_f1 = sum(f1_scores_per_note) / len(f1_scores_per_note)
    
    return {"micro_f1": micro_f1}


# ==============================================================================
# 4. MÉTRICAS DE *SOLO NER* (AISLADO)
# (¡NUEVO! Esto es lo que necesitas para tu diagnose_NER.py)
# ==============================================================================

# --- Minimum IoU for physical overlap (Text Containment logic) ---
MIN_IOU_OVERLAP = 0.1


def text_containment_match(pred_text: str, gt_text: str) -> bool:
    """
    Verifica si hay una relación de contención textual entre predicción y GT.
    
    Criterios (después de normalizar a lowercase y strip):
    - Context Expansion: GT está contenido en Pred (ej: "acute hemorrhage" contiene "hemorrhage")
    - Partial Match: Pred está contenido en GT
    
    Returns: True si hay contención en cualquier dirección.
    """
    pred_norm = pred_text.lower().strip()
    gt_norm = gt_text.lower().strip()
    
    return gt_norm in pred_norm or pred_norm in gt_norm


def _find_ner_span_matches(predictions: List[Dict], 
                           ground_truth: List[Dict], 
                           iou_threshold: float = 0.5,
                           note_text: str = None
                           ) -> Tuple[int, int, int]:
    """
    MATCHING (SOLO NER) con Text Containment + IoU:
    
    NEW MATCHING CRITERIA (RAG-Ready Recall):
    1. Condition A: IoU > 0.1 (physical overlap)
    2. Condition B: Text containment (GT in Pred OR Pred in GT)
    3. Constraint: 1-to-1 matching (protects against "Bad Merge")
    
    Devuelve: (TP, FP, FN)
    """
    tp = 0
    matched_gt_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()
    
    # Ensure predictions have text field
    preds_with_text = []
    for p in predictions:
        p_copy = dict(p)
        if 'text' not in p_copy and note_text:
            p_copy['text'] = note_text[p_copy['start']:p_copy['end']]
        preds_with_text.append(p_copy)

    for gt_idx, gt in enumerate(ground_truth):
        best_match_pred_idx = -1
        best_iou = -1
        
        gt_text = gt.get('text', '')
        if not gt_text and note_text:
            gt_text = note_text[gt['start']:gt['end']]

        for pred_idx, pred in enumerate(preds_with_text):
            if pred_idx in matched_pred_indices:
                continue

            # --- COINCIDENCIA DE SPAN (IoU) ---
            iou = calculate_iou(pred, gt)
            
            # Condition A: Physical overlap
            if iou <= MIN_IOU_OVERLAP:
                continue
            
            pred_text = pred.get('text', '')
            
            # Condition B: Text containment
            if not text_containment_match(pred_text, gt_text):
                continue
            
            # Valid match - track best by IoU
            if iou > best_iou:
                best_iou = iou
                best_match_pred_idx = pred_idx
        
        if best_match_pred_idx != -1:
            tp += 1
            matched_pred_indices.add(best_match_pred_idx)
            matched_gt_indices.add(gt_idx)

    fp = len(predictions) - len(matched_pred_indices)
    fn = len(ground_truth) - len(matched_gt_indices)
    
    return tp, fp, fn

def calculate_ner_macro_f1(all_predictions: Dict[str, List[Dict]], 
                           all_ground_truth: Dict[str, List[Dict]], 
                           iou_threshold: float = 0.5,
                           all_notes: Dict[str, str] = None
                           ) -> Dict[str, float]:
    """
    Calcula el F1-Score de SOLO NER (Macro-Average) con Text Containment + IoU.
    Mide el F1 de Span para cada nota y luego hace la media.
    """
    f1_scores_per_note = []
    
    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        note_text = all_notes.get(note_id, '') if all_notes else None
        
        # Usa la nueva función de matching con Text Containment + IoU
        tp_i, fp_i, fn_i = _find_ner_span_matches(
            pred_annotations, 
            gt_annotations, 
            iou_threshold,
            note_text
        )
        
        metrics_i = _calculate_pr_f1(tp_i, fp_i, fn_i)
        f1_scores_per_note.append(metrics_i['f1'])

    if not f1_scores_per_note:
        return {"macro_f1": 0.0}
        
    macro_f1 = sum(f1_scores_per_note) / len(f1_scores_per_note)
    
    return {"macro_f1": macro_f1}

def calculate_ner_micro_f1(all_predictions: Dict[str, List[Dict]], 
                           all_ground_truth: Dict[str, List[Dict]], 
                           iou_threshold: float = 0.5,
                           all_notes: Dict[str, str] = None
                           ) -> Dict[str, float]:
    """
    Calcula el F1-Score de SOLO NER (Micro-Average) con Text Containment + IoU.
    Calcula F1 para cada nota y luego hace la media aritmética: (x1+x2+...+xn)/n.
    """
    f1_scores_per_note = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        note_text = all_notes.get(note_id, '') if all_notes else None
        
        # Usa la nueva función de matching con Text Containment + IoU
        tp_i, fp_i, fn_i = _find_ner_span_matches(
            pred_annotations, 
            gt_annotations, 
            iou_threshold,
            note_text
        )
        
        total_tp += tp_i
        total_fp += fp_i
        total_fn += fn_i
        
        metrics_i = _calculate_pr_f1(tp_i, fp_i, fn_i)
        f1_scores_per_note.append(metrics_i['f1'])

    if not f1_scores_per_note:
        return {"micro_f1": 0.0, "recall": 0.0, "precision": 0.0, "tp": 0, "fp": 0, "fn": 0}
        
    micro_f1 = sum(f1_scores_per_note) / len(f1_scores_per_note)
    
    # Also calculate global metrics
    global_metrics = _calculate_pr_f1(total_tp, total_fp, total_fn)
    
    return {
        "micro_f1": micro_f1,
        "recall": global_metrics['recall'],
        "precision": global_metrics['precision'],
        "f1": global_metrics['f1'],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }