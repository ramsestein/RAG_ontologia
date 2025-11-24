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

def _find_ner_span_matches(predictions: List[Dict], 
                           ground_truth: List[Dict], 
                           iou_threshold: float = 0.5
                           ) -> Tuple[int, int, int]:
    """
    MATCHING (SOLO NER): Un "acierto" requiere SOLO IoU > threshold.
    Ignora el concept_id.
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

            # --- COINCIDENCIA DE SPAN (IoU) ---
            # (Esta es la única comprobación que hacemos)
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

def calculate_ner_macro_f1(all_predictions: Dict[str, List[Dict]], 
                           all_ground_truth: Dict[str, List[Dict]], 
                           iou_threshold: float = 0.5
                           ) -> Dict[str, float]:
    """
    Calcula el F1-Score de SOLO NER (Macro-Average).
    Mide el F1 de Span (IoU) para cada nota y luego hace la media.
    """
    f1_scores_per_note = []
    
    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        
        # Usa la nueva función de matching de SOLO NER
        tp_i, fp_i, fn_i = _find_ner_span_matches(
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

def calculate_ner_micro_f1(all_predictions: Dict[str, List[Dict]], 
                           all_ground_truth: Dict[str, List[Dict]], 
                           iou_threshold: float = 0.5
                           ) -> Dict[str, float]:
    """
    Calcula el F1-Score de SOLO NER (Micro-Average) como media aritmética.
    Calcula F1 para cada nota y luego hace la media aritmética: (x1+x2+...+xn)/n.
    """
    f1_scores_per_note = []

    for note_id, gt_annotations in all_ground_truth.items():
        pred_annotations = all_predictions.get(note_id, [])
        
        # Usa la nueva función de matching de SOLO NER
        tp_i, fp_i, fn_i = _find_ner_span_matches(
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