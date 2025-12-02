# benchmarks/diagnose_NER.py
#
# OBJETIVO: Orquestador SOTA para diagnosticar modelos NER.
# UPDATES:
# - FIX: Lógica de Matching "Text Containment + IoU" para RAG-Ready Recall.
#   - Condition A: IoU > 0.1 (overlap físico mínimo)
#   - Condition B: Contención textual (GT en Pred O Pred en GT)
#   - Constraint: 1-to-1 matching (protege contra "Bad Merge")
# - FEATURE: Análisis secuencial de contribución (Non-Redundant Recall).
# - FIX: Tabla de resultados detallada con F1 Harmónico/Aritmético.

import json
import time
import sys
import argparse
import importlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Any

# --- Arreglo de Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports de Módulos ---
from src.utils.metrics import (
    calculate_ner_micro_f1, 
    _calculate_pr_f1,
    calculate_iou
)

# --- Constantes ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"


def setup_argparser() -> argparse.ArgumentParser:
    epilog_text = """
Ejemplos:
  python benchmarks/diagnose_NER.py assembly --iou 0.25 -v
  python benchmarks/diagnose_NER.py all
    """
    parser = argparse.ArgumentParser(
        description="Diagnóstico NER.",
        epilog=epilog_text,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("target", metavar="TARGET", type=str, 
                        help="ID del NER (ej: SBc5), 'all' o 'assembly'.")
    parser.add_argument("--iou", type=float, default=0.25, help="Umbral IoU (Def: 0.25)")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Ver tabla detallada de aciertos/fallos.")
    return parser

def load_data(notes_p, gt_p):
    print(f"[Loader] Notes: {notes_p}")
    print(f"[Loader] GT:    {gt_p}")
    if not notes_p.exists() or not gt_p.exists():
        raise FileNotFoundError("Faltan archivos JSON en /data")
    with open(notes_p, 'r', encoding='utf-8') as f: n = json.load(f)
    with open(gt_p, 'r', encoding='utf-8') as f: g = json.load(f)
    return {i['note_id']: i['text'] for i in n}, {i['note_id']: i['annotations'] for i in g}

def load_ner_worker(ner_id: str, config: Dict) -> Any:
    try:
        mod = importlib.import_module(config['module'])
        cls = getattr(mod, config['class'])
        kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
        # print(f"[Worker] Cargando {ner_id}...") 
        return cls(**kwargs)
    except Exception as e:
        print(f"[ERROR] {ner_id}: {e}")
        return None

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
    
    # Context Expansion: GT['text'] is substring of Pred['text']
    # O Partial Match: Pred['text'] is substring of GT['text']
    return gt_norm in pred_norm or pred_norm in gt_norm


def get_detailed_matches(preds: List[Dict], gt: List[Dict], iou_thresh: float, note_text: str = None) -> Tuple[List, List, List]:
    """
    Matching Logic "Text Containment + IoU Overlap" (RAG-Ready Recall):
    
    NUEVO CRITERIO TP:
    1. Condition A: IoU > 0.1 (overlap físico mínimo para asegurar misma ubicación)
    2. Condition B: Contención textual (GT en Pred O Pred en GT)
    
    CONSTRAINT: 1-to-1 matching. Una predicción solo cuenta para UN GT.
    Esto protege contra el escenario "Bad Merge" donde una predicción cubre
    múltiples entidades GT (ej: "headache and vomiting" cubriendo 2 GT).
    
    Args:
        preds: Lista de predicciones con keys: start, end, (opcionalmente 'text')
        gt: Lista de ground truth con keys: start, end, text
        iou_thresh: Umbral IoU mínimo (default 0.1 para overlap)
        note_text: Texto de la nota para extraer texto de predicciones si no lo tienen
    """
    # Umbral IoU mínimo para asegurar overlap físico (Condition A)
    MIN_IOU_OVERLAP = 0.1
    
    tp_pairs = [] 
    matched_gt_indices = set()
    matched_pred_indices = set()  # Para 1-to-1 matching
    
    # Asegurar que las predicciones tienen el campo 'text'
    preds_with_text = []
    for p in preds:
        p_copy = dict(p)
        if 'text' not in p_copy and note_text:
            p_copy['text'] = note_text[p_copy['start']:p_copy['end']]
        preds_with_text.append(p_copy)
    
    # 1. Recall Scan (Barrido sobre Ground Truth)
    # Para cada GT, buscar la MEJOR predicción que cumpla ambos criterios
    for g_idx, g_item in enumerate(gt):
        best_match_score = -1.0
        best_pred_idx = None
        best_iou = 0.0
        
        gt_text = g_item.get('text', '')
        
        for p_idx, p in enumerate(preds_with_text):
            # Skip si esta predicción ya fue asignada a otro GT (1-to-1 constraint)
            if p_idx in matched_pred_indices:
                continue
            
            iou = calculate_iou(p, g_item)
            
            # Condition A: Debe haber overlap físico mínimo
            if iou <= MIN_IOU_OVERLAP:
                continue
            
            pred_text = p.get('text', '')
            
            # Condition B: Contención textual
            if not text_containment_match(pred_text, gt_text):
                continue
            
            # Ambos criterios cumplidos: es un match válido
            # Usamos IoU como score para elegir el mejor match
            if iou > best_match_score:
                best_match_score = iou
                best_pred_idx = p_idx
                best_iou = iou
        
        if best_pred_idx is not None:
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(best_pred_idx)
            tp_pairs.append((preds_with_text[best_pred_idx], g_item, best_iou))

    fn_gts = [gt[i] for i in range(len(gt)) if i not in matched_gt_indices]

    # 2. Precision Scan (Barrido sobre Predicciones)
    # FP: Predicciones que no matchearon con ningún GT
    fp_preds = [preds_with_text[i] for i in range(len(preds_with_text)) if i not in matched_pred_indices]

    return tp_pairs, fp_preds, fn_gts

def deduplicate_predictions(preds: List[Dict]) -> List[Dict]:
    """Elimina duplicados exactos (mismo start/end) manteniendo metadata."""
    seen = set()
    unique = []
    for p in preds:
        k = (p['start'], p['end'])
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique

def print_detailed_guesses_log(matches, fps, text):
    all_guesses = []
    for p, g, iou in matches:
        src = p.get('source', 'UNK')[:3]
        all_guesses.append({
            'start': p['start'], 'end': p['end'], 'text': text[p['start']:p['end']],
            'status': '✅ TP', 'iou': iou, 'gt': text[g['start']:g['end']], 'src': src
        })
    for p in fps:
        src = p.get('source', 'UNK')[:3]
        all_guesses.append({
            'start': p['start'], 'end': p['end'], 'text': text[p['start']:p['end']],
            'status': '❌ FP', 'iou': 0.0, 'gt': "-", 'src': src
        })
        
    all_guesses.sort(key=lambda x: x['start'])
    
    print("-" * 110)
    print(f"{'STAT':<4} | {'SRC':<3} | {'SPAN':<9} | {'PREDICCION':<30} | {'IoU':<4} | {'GT MATCH'}")
    print("-" * 110)
    for x in all_guesses:
        p_txt = (x['text'][:28] + '..') if len(x['text']) > 28 else x['text']
        g_txt = (x['gt'][:25] + '..') if len(x['gt']) > 25 else x['gt']
        p_txt = p_txt.replace('\n', ' ')
        g_txt = g_txt.replace('\n', ' ')
        print(f"{x['status']:<4} | {x['src']:<3} | {x['start']}-{x['end']:<5} | {p_txt:<30} | {x['iou']:<4.2f} | {g_txt}")
    print("-" * 110)

def print_note_report(note_id, time_s, m, matches, fps, fns, text, verbose):
    print(f"\n 📝 --- Nota {note_id} ---")
    print(f"    T: {time_s:.2f}s | TP: {len(matches)} | FP: {len(fps)} | FN: {len(fns)}")
    print(f"    Prec: {m['precision']:.1%} | Rec: {m['recall']:.1%} | F1: {m['f1']:.1%}")
    
    if verbose:
        print_detailed_guesses_log(matches, fps, text)
    
    if fns:
        print(f"    ⚠️  MISSING (FN):")
        for g in fns:
            t_clean = text[g['start']:g['end']].replace('\n', ' ')
            print(f"       [{g['start']}-{g['end']}] '{t_clean}'")

def calculate_sequential_contribution(worker_ids: List[str], notes: Dict, gt_data: Dict, iou_th: float, registry: Dict) -> Dict[str, float]:
    """
    Calcula la contribución INCREMENTAL de cada worker usando la nueva lógica
    de Text Containment + IoU.
    Es decir: ¿Cuántos TP *nuevos* encuentra el worker X que los anteriores no encontraron?
    """
    MIN_IOU_OVERLAP = 0.1  # Mismo umbral que get_detailed_matches
    
    total_gt_count = sum(len(g) for g in gt_data.values())
    if total_gt_count == 0: return {}
    
    covered_gt_ids = set() # Set of (note_id, start, end) strings/tuples
    sequential_results = {} # {worker_id: count_of_new_tps}
    
    # Cache worker instances
    print("\n ⏳ Analizando contribución secuencial (esto puede tardar)...")
    
    for wid in worker_ids:
        worker = load_ner_worker(wid, registry[wid])
        if not worker: continue
        
        new_tps_count = 0
        
        for nid, text in notes.items():
            w_preds = worker.extract_entities(text)
            gt_list = gt_data.get(nid, [])
            
            # Usamos la lógica de matching "Text Containment + IoU"
            # Para cada item del GT, chequeamos si este worker lo encuentra
            for gt_item in gt_list:
                # ID único para este GT item
                gt_uid = (nid, gt_item['start'], gt_item['end'])
                
                # Si ya fue encontrado por un worker anterior, lo ignoramos
                if gt_uid in covered_gt_ids:
                    continue
                
                gt_text = gt_item.get('text', '')
                
                # Si no ha sido encontrado, verificamos si este worker lo encuentra
                found = False
                for p in w_preds:
                    iou = calculate_iou(p, gt_item)
                    # Condition A: IoU overlap mínimo
                    if iou <= MIN_IOU_OVERLAP:
                        continue
                    
                    # Extraer texto de la predicción
                    pred_text = text[p['start']:p['end']]
                    
                    # Condition B: Contención textual
                    if text_containment_match(pred_text, gt_text):
                        found = True
                        break
                
                if found:
                    covered_gt_ids.add(gt_uid)
                    new_tps_count += 1
        
        sequential_results[wid] = new_tps_count / total_gt_count if total_gt_count > 0 else 0
    
    return sequential_results

def run_benchmark(ids, registry, notes, gt_data, iou_th, mode, verbose):
    results = []
    
    # 1. Setup Workers Logic
    loaded_workers = {} 
    
    run_ids = []
    if mode == 'assembly':
        print(f"🧩 MODO ASSEMBLY: {', '.join(ids)}")
        # Pre-load all for assembly execution
        for nid in ids:
            w = load_ner_worker(nid, registry[nid])
            if w: loaded_workers[nid] = w
        if not loaded_workers: return []
        run_ids = ["ASSEMBLY"]
    else:
        run_ids = ids

    # 2. Execution Loop
    for run_id in run_ids:
        if mode == 'assembly':
            active_workers = loaded_workers
            label = "ASSEMBLY"
        else:
            # Single mode: Load specific worker
            print(f"\n🔬 MODELO: {run_id}")
            w = load_ner_worker(run_id, registry[run_id])
            if not w: continue
            active_workers = {run_id: w}
            label = run_id
        
        all_preds = {}
        note_f1s_harmonic = []
        note_f1s_arithmetic = []
        t_total = 0
        
        for nid, text in notes.items():
            t0 = time.time()
            raw_preds = []
            
            # A. Extract
            for wid, worker in active_workers.items():
                w_preds = worker.extract_entities(text)
                for p in w_preds: p['source'] = wid 
                raw_preds.extend(w_preds)
            
            # B. Deduplicate (Union simple para maximizar cobertura)
            preds = deduplicate_predictions(raw_preds)
            
            t_total += (time.time() - t0)
            all_preds[nid] = preds
            
            # C. Evaluate (passing note_text for text containment matching)
            matches, fps, fns = get_detailed_matches(preds, gt_data.get(nid, []), iou_th, note_text=text)
            m = _calculate_pr_f1(len(matches), len(fps), len(fns))
            
            # Store harmonic F1 (standard F1 = 2PR/(P+R))
            note_f1s_harmonic.append(m['f1'])
            
            # Calculate and store arithmetic F1 (arithmetic mean of P and R = (P+R)/2)
            f1_arith_per_note = (m['precision'] + m['recall']) / 2.0
            note_f1s_arithmetic.append(f1_arith_per_note)
            
            print_note_report(nid, time.time()-t0, m, matches, fps, fns, text, verbose)

        # D. Metrics
        # Nota: Recalculamos usando la nueva lógica "Text Containment + IoU"
        
        total_tp = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th, note_text=notes[nid])[0]) for nid in notes)
        total_fp = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th, note_text=notes[nid])[1]) for nid in notes)
        total_fn = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th, note_text=notes[nid])[2]) for nid in notes)
        
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        
        # F1-Harmonic: Arithmetic mean of per-note harmonic F1 scores
        f1_harm = sum(note_f1s_harmonic) / len(note_f1s_harmonic) if note_f1s_harmonic else 0
        
        # F1-Arithmetic: Arithmetic mean of per-note arithmetic F1 scores
        arith_f1 = sum(note_f1s_arithmetic) / len(note_f1s_arithmetic) if note_f1s_arithmetic else 0
        
        res_entry = {
            "ID": label, 
            "F1-Harm": f1_harm, 
            "F1-Arith": arith_f1,
            "Prec": prec, 
            "Rec": recall, 
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Time": t_total
        }

        # E. Sequential Contribution Analysis (Only for Assembly)
        if mode == 'assembly':
            # Pass the original 'ids' list to preserve user-defined order
            seq_contrib = calculate_sequential_contribution(ids, notes, gt_data, iou_th, registry)
            res_entry['SeqContrib'] = seq_contrib
            
        results.append(res_entry)

    return results

def main():
    args = setup_argparser().parse_args()
    try:
        with open(CONFIG_PATH) as f: reg = json.load(f)
        notes, gt = load_data(NOTES_PATH, GT_PATH)
    except Exception as e: return print(f"[ERROR] {e}")

    mode = 'single'
    if args.target.lower() == 'all': ids = list(reg.keys())
    elif args.target.lower() == 'assembly': 
        ids = list(reg.keys())
        mode = 'assembly'
    else: ids = [args.target]

    print(f"🚀 Benchmark [IoU > {args.iou}]")
    res = run_benchmark(ids, reg, notes, gt, args.iou, mode, args.verbose)
    
    if res:
        # Sort by Harmonic F1 desc
        res.sort(key=lambda x: x['F1-Harm'], reverse=True)
        
        print("\n" + "="*115)
        print(f" 🏆 RESULTADOS FINALES ({mode.upper()}) - IoU > {args.iou}")
        print("="*115)
        print(f"{'ID':<20} | {'F1-Harmonic':<12} | {'F1-Arithmetic':<14} | {'Precision':<10} | {'Recall':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Time':<8}")
        print("-" * 115)
        
        for r in res:
            print(f"{r['ID']:<20} | {r['F1-Harm']:<12.4f} | {r['F1-Arith']:<14.4f} | {r['Prec']:<10.4f} | {r['Rec']:<8.4f} | {r['TP']:<4} | {r['FP']:<4} | {r['FN']:<4} | {r['Time']:<8.2f}")
        print("="*115)

        # Print Sequential Contribution Table if present
        if mode == 'assembly' and 'SeqContrib' in res[0]:
            print("\n" + "="*80)
            print(f" 🎯 ANÁLISIS DE CONTRIBUCIÓN SECUENCIAL (Non-Redundant)")
            print(f"    Orden de ejecución: {', '.join(ids)}")
            print("-" * 80)
            print(f"{'Model':<20} | {'+ Recall Absoluto':<20} | {'Recall Acumulado'}")
            print("-" * 80)
            
            running_total = 0.0
            # Iterate in the original order defined in 'ids'
            for wid in ids:
                val = res[0]['SeqContrib'].get(wid, 0.0)
                running_total += val
                print(f"{wid:<20} | +{val:<19.1%} | {running_total:.1%}")
            print("="*80)

if __name__ == "__main__":
    main()