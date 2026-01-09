#!/usr/bin/env python3
"""
evaluator_stage3.py - Evaluador de métricas para Stage 3 (SemanticFisher)

Compara las predicciones del SemanticFisher (stage3_llm.json) con el ground truth
y calcula métricas de precisión, recall y F1 por nota y en promedio.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set


def load_json(path: str) -> Any:
    """Carga un archivo JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str) -> None:
    """Guarda datos en formato JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_predictions_from_stage3(stage3_data: Dict) -> Dict[str, Set[str]]:
    """
    Extrae los códigos predichos agrupados por nota.
    
    Args:
        stage3_data: Datos del stage3_llm.json
    
    Returns:
        Diccionario {id_nota: set(códigos_predichos)}
    """
    predictions = {}
    results = stage3_data.get('results', [])
    
    for result in results:
        note_id = str(result.get('id', ''))
        code = result.get('code')
        
        # Solo considerar candidatos con código asignado
        if code:
            if note_id not in predictions:
                predictions[note_id] = set()
            predictions[note_id].add(code)
    
    return predictions


def calculate_note_metrics(gt_codes: List[str], pred_codes: Set[str]) -> Tuple[int, int, int, float, float, float, Set[str], Set[str], Set[str]]:
    """
    Calcula métricas para una nota individual.
    
    Args:
        gt_codes: Lista de códigos del ground truth
        pred_codes: Set de códigos predichos
    
    Returns:
        Tupla con (TP, FP, FN, Precision, Recall, F1, TPs, FPs, FNs)
    """
    gt = set(gt_codes)
    pred = pred_codes if pred_codes else set()
    
    tp_set = gt.intersection(pred)
    fp_set = pred - gt
    fn_set = gt - pred
    
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    
    # Manejar caso edge: si ambos conjuntos están vacíos
    if not gt and not pred:
        precision, recall, f1 = 1.0, 1.0, 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, precision, recall, f1, tp_set, fp_set, fn_set


def run_evaluation():
    """Función principal del evaluador."""
    # Definir rutas
    base_path = Path(__file__).parent.parent.parent  # new_ofarres/
    gt_path = base_path / 'test' / 'llm' / 'ground_truth.json'
    pred_path = base_path / 'src' / 'NER' / 'output' / 'stage3_llm.json'
    output_path = base_path / 'src' / 'NER' / 'output' / 'evaluation_stage3.json'
    
    # Buffer para guardar en archivo
    output_lines = []
    
    def print_and_save(text: str):
        print(text)
        output_lines.append(text)
    
    print_and_save("=" * 100)
    print_and_save("📊 EVALUADOR STAGE 3: SemanticFisher (High-Recall NER)")
    print_and_save("=" * 100)
    print_and_save(f"📅 Fecha de evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_and_save(f"📂 Ground Truth: {gt_path}")
    print_and_save(f"📂 Predicciones: {pred_path}")
    print_and_save("")
    
    # Cargar datos
    gt_data = load_json(gt_path)
    stage3_data = load_json(pred_path)
    
    # Extraer predicciones
    pred_map = extract_predictions_from_stage3(stage3_data)
    
    # Mostrar metadatos del stage3
    metadata = stage3_data.get('metadata', {})
    print_and_save(f"✓ Ground truth cargado: {len(gt_data)} notas")
    print_and_save(f"✓ Predicciones Stage 3 cargadas:")
    print_and_save(f"    - Total entidades: {metadata.get('total_entities', 'N/A')}")
    print_and_save(f"    - Threshold: {metadata.get('threshold', 'N/A')}")
    print_and_save(f"    - Notas con predicciones: {len(pred_map)}")
    print_and_save("")
    
    # Tabla de resultados
    print_and_save("=" * 100)
    print_and_save("RESULTADOS POR NOTA")
    print_and_save("=" * 100)
    
    header = f"{'Nota ID':<12} | {'note_id':<8} | {'GT':<4} | {'Pred':<4} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec.':<7} | {'Rec.':<7} | {'F1':<7}"
    print_and_save(header)
    print_and_save("-" * len(header))
    
    all_metrics = []
    detailed_results = []
    
    for note in gt_data:
        n_id = str(note['id'])
        note_num = note.get('note_id', 'N/A')
        gt_codes = note['found_codes']
        pred_codes = pred_map.get(n_id, set())
        
        tp, fp, fn, prec, rec, f1, tp_set, fp_set, fn_set = calculate_note_metrics(gt_codes, pred_codes)
        
        all_metrics.append({
            "note_id": n_id,
            "note_num": note_num,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn
        })
        
        detailed_results.append({
            "id": n_id,
            "note_id": note_num,
            "gt_codes": gt_codes,
            "pred_codes": list(pred_codes),
            "true_positives": list(tp_set),
            "false_positives": list(fp_set),
            "false_negatives": list(fn_set),
            "metrics": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            }
        })
        
        # Usar los últimos 8 chars del ID para legibilidad
        short_id = n_id[-8:]
        print_and_save(f"{short_id:<12} | {note_num:<8} | {len(gt_codes):<4} | {len(pred_codes):<4} | {tp:<4} | {fp:<4} | {fn:<4} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f}")
    
    print_and_save("-" * len(header))

    # Totals
    total_tp = sum(m['tp'] for m in all_metrics)
    total_fp = sum(m['fp'] for m in all_metrics)
    total_fn = sum(m['fn'] for m in all_metrics)

    # MEDIAN-BASED METRICS (arithmetic median for precision and recall)
    from statistics import median
    precisions = [m['precision'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    prec_med = median(precisions) if precisions else 0.0
    rec_med = median(recalls) if recalls else 0.0
    # F1 as harmonic mean of median precision and median recall
    f1_med = 2 * (prec_med * rec_med) / (prec_med + rec_med) if (prec_med + rec_med) > 0 else 0.0

    print_and_save(f"{ 'MEDIAN':<12} | {'-':<8} | {'-':<4} | {'-':<4} | {'-':<4} | {'-':<4} | {'-':<4} | {prec_med:<7.4f} | {rec_med:<7.4f} | {f1_med:<7.4f}")
    print_and_save("=" * 100)
    
    # Resumen final
    print_and_save("")
    print_and_save("=" * 100)
    print_and_save("📈 RESUMEN FINAL - STAGE 3")
    print_and_save("=" * 100)
    print_and_save(f"Total de notas evaluadas: {len(gt_data)}")
    print_and_save(f"Total True Positives (TP): {total_tp}")
    print_and_save(f"Total False Positives (FP): {total_fp}")
    print_and_save(f"Total False Negatives (FN): {total_fn}")
    print_and_save("")
    print_and_save("📊 MEDIAN METRICS (arithmetic median)")
    print_and_save(f"   Precision (median): {prec_med:.4f} ({prec_med*100:.2f}%)")
    print_and_save(f"   Recall    (median): {rec_med:.4f} ({rec_med*100:.2f}%)")
    print_and_save(f"   F1-Score (harmonic): {f1_med:.4f} ({f1_med*100:.2f}%)")
    print_and_save("=" * 100)
    
    # Guardar resultados en JSON (median metrics only)
    evaluation_output = {
        "metadata": {
            "stage": "stage3_llm",
            "evaluation_date": datetime.now().isoformat(),
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "total_notes": len(gt_data)
        },
        "summary": {
            "median_metrics": {
                "precision": round(prec_med, 4),
                "recall": round(rec_med, 4),
                "f1": round(f1_med, 4)
            },
            "totals": {
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn
            }
        },
        "per_note_results": detailed_results
    }
    
    save_json(evaluation_output, output_path)
    print_and_save(f"\n✅ Resultados guardados en: {output_path}")
    
    # Guardar reporte en texto
    report_path = base_path / 'src' / 'NER' / 'output' / 'evaluation_stage3_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print_and_save(f"✅ Reporte de texto guardado en: {report_path}")
    
    return evaluation_output


if __name__ == "__main__":
    run_evaluation()
