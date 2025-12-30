import json
import numpy as np
from pathlib import Path

# CONFIGURACIÓN DE RUTAS
GT_PATH = "test/ground_truth/llm_labels.json"
PRED_PATH = "src/NER/output/final_predictions.json"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_note_metrics(gt_codes, pred_codes):
    gt = set(gt_codes)
    pred = set(pred_codes)
    
    tp = len(gt.intersection(pred))
    fp = len(pred - gt)
    fn = len(gt - pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gt and not pred else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gt and not pred else 0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, precision, recall, f1

def run_evaluation():
    print(f"{'='*90}")
    print(f"📊 EVALUADOR NER: CONSISTENCIA POR NOTA MÉDICA")
    print(f"{'='*90}\n")
    
    gt_data = load_json(GT_PATH)
    pred_data = load_json(PRED_PATH)
    
    # Mapear predicciones por ID para facilitar búsqueda
    pred_map = {str(item['id']): item['found_codes'] for item in pred_data}
    
    all_metrics = []
    
    header = f"{'Nota ID':<20} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Prec.':<6} | {'Rec.':<6} | {'F1':<6}"
    print(header)
    print("-" * len(header))
    
    for note in gt_data:
        n_id = str(note['id'])
        gt_codes = note['found_codes']
        pred_codes = pred_map.get(n_id, [])
        
        tp, fp, fn, prec, rec, f1 = calculate_note_metrics(gt_codes, pred_codes)
        
        all_metrics.append({
            "prec": prec,
            "rec": rec,
            "f1": f1
        })
        
        # Mostrar detalle por nota (usamos los últimos 6 chars del ID para legibilidad)
        short_id = n_id[-8:]
        print(f"{short_id:<20} | {tp:<4} | {fp:<4} | {fn:<4} | {prec:<6.2f} | {rec:<6.2f} | {f1:<6.2f}")

    # CÁLCULO DE MEDIAS (Macro-Average)
    avg_prec = np.mean([m['prec'] for m in all_metrics])
    avg_rec = np.mean([m['rec'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])
    
    print("-" * len(header))
    print(f"{'MEDIA ARITMÉTICA':<20} | {'-':<4} | {'-':<4} | {'-':<4} | {avg_prec:<6.2f} | {avg_rec:<6.2f} | {avg_f1:<6.2f}")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    run_evaluation()