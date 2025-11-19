# benchmarks/diagnose_NER.py
#
# OBJETIVO: Orquestador SOTA para diagnosticar modelos NER (Individuales o Ensamble).
#
# MODOS:
# 1. Individual: python diagnose_NER.py SBc5
# 2. Comparativo: python diagnose_NER.py all
# 3. Ensamble: python diagnose_NER.py assembly  <-- ¡NUEVO!

import json
import time
import sys
import argparse
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

# --- Arreglo de Paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SRC_PATH = PROJECT_ROOT / "src"
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
    parser = argparse.ArgumentParser(description="Diagnóstico NER SOTA.")
    parser.add_argument("target", metavar="TARGET", type=str, 
                        help="ID del NER (ej: SBc5), 'all' o 'assembly'.")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU (Def: 0.5)")
    return parser

def load_data(notes_p, gt_p):
    print(f"[Loader] Cargando datos...")
    with open(notes_p, 'r', encoding='utf-8') as f: n = json.load(f)
    with open(gt_p, 'r', encoding='utf-8') as f: g = json.load(f)
    return {i['note_id']: i['text'] for i in n}, {i['note_id']: i['annotations'] for i in g}

def load_ner_worker(ner_id: str, config: Dict) -> Any:
    try:
        mod = importlib.import_module(config['module'])
        cls = getattr(mod, config['class'])
        kwargs = {k: v for k, v in config.items() if k not in ['module', 'class']}
        print(f"[Worker] Cargando {ner_id} ({config['class']})...")
        return cls(**kwargs)
    except Exception as e:
        print(f"[ERROR] Fallo cargando {ner_id}: {e}")
        return None

def get_detailed_matches(preds, gt, iou_thresh=0.5):
    matched_gt, matched_pred = set(), set()
    pairs = []
    
    for p_idx, p in enumerate(preds):
        best_idx, best_iou = -1, -1.0
        for g_idx, g in enumerate(gt):
            if g_idx in matched_gt: continue
            iou = calculate_iou(p, g)
            if iou > iou_thresh and iou > best_iou:
                best_iou, best_idx = iou, g_idx
        
        if best_idx != -1:
            pairs.append((p, gt[best_idx], best_iou))
            matched_pred.add(p_idx)
            matched_gt.add(best_idx)

    un_p = [preds[i] for i in range(len(preds)) if i not in matched_pred]
    un_g = [gt[i] for i in range(len(gt)) if i not in matched_gt]
    return pairs, un_p, un_g

def deduplicate_predictions(preds: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []
    for p in preds:
        k = (p['start'], p['end'])
        if k not in seen:
            seen.add(k)
            unique.append(p)
    return unique

def print_note_report(note_id, time_s, metrics, matches, fps, fns, text):
    print(f"\n  --- Nota {note_id} ---")
    print(f"    Tiempo: {time_s:.2f}s | TP: {len(matches)} | FP: {len(fps)} | FN: {len(fns)}")
    print(f"    Prec: {metrics['precision']:.1%} | Rec: {metrics['recall']:.1%} | F1: {metrics['f1']:.1%}")
    
    if matches:
        print(f"    ✅ Aciertos (Top 5):")
        for p, g, iou in matches[:5]:
            print(f"       [{p['start']}-{p['end']}] '{text[p['start']:p['end']][:30]}' (IoU={iou:.2f})")
    if fns:
        print(f"    ⚠️ Perdidos (Top 5 FN):")
        for g in fns[:5]:
            print(f"       [{g['start']}-{g['end']}] '{text[g['start']:g['end']][:30]}'")

def run_benchmark(ids, registry, notes, gt_data, iou_th, mode):
    results = []
    
    # Configurar workers
    workers = []
    if mode == 'assembly':
        print(f"🧩 MODO ASSEMBLY: {', '.join(ids)}")
        for nid in ids:
            if w := load_ner_worker(nid, registry[nid]): workers.append(w)
        if not workers: return []
        run_ids = ["ASSEMBLY"] # Etiqueta para la tabla
    else:
        run_ids = ids # Iterar uno por uno

    for run_id in run_ids:
        if mode != 'assembly':
            print(f"\n🔬 DIAGNOSTICANDO: {run_id}")
            w = load_ner_worker(run_id, registry[run_id])
            if not w: continue
            current_workers = [w]
        else:
            current_workers = workers # Usar todos

        all_preds = {}
        note_f1s = []
        t_total = 0

        for nid, text in notes.items():
            t0 = time.time()
            # Extracción (y unión si es assembly)
            raw_preds = []
            for w in current_workers:
                raw_preds.extend(w.extract_entities(text))
            
            preds = deduplicate_predictions(raw_preds)
            t_total += (time.time() - t0)
            all_preds[nid] = preds
            
            # Métricas por nota
            matches, fps, fns = get_detailed_matches(preds, gt_data.get(nid, []), iou_th)
            m = _calculate_pr_f1(len(matches), len(fps), len(fns))
            note_f1s.append(m['f1'])
            
            print_note_report(nid, time.time()-t0, m, matches, fps, fns, text)

        # Métricas Globales
        global_m = calculate_ner_micro_f1(all_preds, gt_data, iou_th)
        arithmetic_f1 = sum(note_f1s) / len(note_f1s) if note_f1s else 0
        
        results.append({
            "ID": run_id,
            "F1-Harmonic": global_m['f1'],      # Harmonic (Global)
            "F1-Arithmetic": arithmetic_f1,       # Arithmetic (Mean of notes)
            "Precision": global_m['precision'],
            "Recall": global_m['recall'],
            "TP": global_m['tp'], "FP": global_m['fp'], "FN": global_m['fn'],
            "Time": t_total
        })

    return results

def main():
    args = setup_argparser().parse_args()
    try:
        with open(CONFIG_PATH) as f: reg = json.load(f)
        notes, gt = load_data(NOTES_PATH, GT_PATH)
    except Exception as e: return print(f"[ERROR] Datos no encontrados: {e}")

    mode = 'single'
    if args.target.lower() == 'all': ids = list(reg.keys())
    elif args.target.lower() == 'assembly': 
        ids = list(reg.keys())
        mode = 'assembly'
    else: 
        if args.target not in reg: return print(f"[ERROR] ID '{args.target}' no existe.")
        ids = [args.target]

    res = run_benchmark(ids, reg, notes, gt, args.iou, mode)
    
    if res:
            res.sort(key=lambda x: x['F1-Harmonic'], reverse=True)
            print("\n" + "="*115)
            print(f" RESULTADOS FINALES ({mode.upper()}) - IoU > {args.iou}")
            print("="*115)
            
            # HEADER CORREGIDO CON ANCHOS SUFICIENTES
            # ID (20) | F1-Harm (12) | F1-Arit (14) | Prec (10) | Rec (8) | TP (4) | FP (4) | FN (4) | Time (8)
            print(f"{'ID':<20} | {'F1-Harmonic':<12} | {'F1-Arithmetic':<14} | {'Precision':<10} | {'Recall':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Time':<8}")
            print("-" * 115)
            
            for r in res:
                print(f"{r['ID']:<20} | {r['F1-Harmonic']:<12.4f} | {r['F1-Arithmetic']:<14.4f} | {r['Precision']:<10.4f} | {r['Recall']:<8.4f} | {r['TP']:<4} | {r['FP']:<4} | {r['FN']:<4} | {r['Time']:<8.2f}")
            print("="*115)

if __name__ == "__main__":
    main()