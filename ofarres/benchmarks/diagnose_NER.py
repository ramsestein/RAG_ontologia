# benchmarks/diagnose_NER.py
#
# OBJETIVO: Orquestador SOTA para diagnosticar modelos NER.
# UPDATES:
# - FIX: Lógica de Matching "Coverage" para 100% Recall en entidades anidadas.
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

def get_detailed_matches(preds: List[Dict], gt: List[Dict], iou_thresh: float) -> Tuple[List, List, List]:
    """
    Matching Logic "Coverage" (Cobertura):
    1. RECALL: Iteramos sobre el GT. Si una entidad GT es cubierta por CUALQUIER
       predicción (aunque esa predicción ya se haya usado), cuenta como TP.
       Esto soluciona el problema de entidades anidadas ("Hemorrhage" dentro de "SAH").
    
    2. PRECISION: Una predicción es FP solo si no toca NINGUNA entidad GT.
    """
    tp_pairs = [] 
    matched_gt_indices = set()
    
    # 1. Recall Scan (Barrido sobre Ground Truth)
    for g_idx, g_item in enumerate(gt):
        best_iou = -1.0
        best_pred = None
        for p in preds:
            iou = calculate_iou(p, g_item)
            if iou > iou_thresh and iou > best_iou:
                best_iou = iou
                best_pred = p
        
        if best_pred:
            matched_gt_indices.add(g_idx)
            tp_pairs.append((best_pred, g_item, best_iou))

    fn_gts = [gt[i] for i in range(len(gt)) if i not in matched_gt_indices]

    # 2. Precision Scan (Barrido sobre Predicciones)
    fp_preds = []
    for p in preds:
        is_useful = False
        for g in gt:
            if calculate_iou(p, g) > iou_thresh:
                is_useful = True
                break
        if not is_useful:
            fp_preds.append(p)

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
    Calcula la contribución INCREMENTAL de cada worker.
    Es decir: ¿Cuántos TP *nuevos* encuentra el worker X que los anteriores no encontraron?
    """
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
            
            # Usamos la lógica de matching "Coverage"
            # Para cada item del GT, chequeamos si este worker lo encuentra
            for gt_item in gt_list:
                # ID único para este GT item
                gt_uid = (nid, gt_item['start'], gt_item['end'])
                
                # Si ya fue encontrado por un worker anterior, lo ignoramos
                if gt_uid in covered_gt_ids:
                    continue
                
                # Si no ha sido encontrado, verificamos si este worker lo encuentra
                found = False
                for p in w_preds:
                    if calculate_iou(p, gt_item) > iou_th:
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
        note_f1s = []
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
            
            # C. Evaluate
            matches, fps, fns = get_detailed_matches(preds, gt_data.get(nid, []), iou_th)
            m = _calculate_pr_f1(len(matches), len(fps), len(fns))
            note_f1s.append(m['f1'])
            
            print_note_report(nid, time.time()-t0, m, matches, fps, fns, text, verbose)

        # D. Metrics
        # Nota: calculate_ner_micro_f1 usa set intersection simple, 
        # recalcular metrics basados en los conteos "Coverage" locales es más preciso para Nested.
        
        total_tp = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th)[0]) for nid in notes)
        total_fp = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th)[1]) for nid in notes)
        total_fn = sum(len(get_detailed_matches(all_preds[nid], gt_data.get(nid, []), iou_th)[2]) for nid in notes)
        
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        f1_harm = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
        
        arith_f1 = sum(note_f1s) / len(note_f1s) if note_f1s else 0
        
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