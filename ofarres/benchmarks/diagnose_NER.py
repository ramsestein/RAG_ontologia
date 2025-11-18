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
    calculate_ner_macro_f1,
    _find_ner_span_matches,
    _calculate_pr_f1,
    calculate_iou
)

# --- Constantes ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"


def setup_argparser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        description="Diagnóstico Modular de Modelos NER (Benchmark SOTA).",
        formatter_class=formatter,
        epilog="""
-------------------------------------------------------------------
CASOS DE USO:
-------------------------------------------------------------------
1. Probar un modelo:      python benchmarks/diagnose_NER.py SBc5
2. Comparar todos:        python benchmarks/diagnose_NER.py all
3. Probar ENSAMBLE (Unión): python benchmarks/diagnose_NER.py assembly
-------------------------------------------------------------------
"""
    )
    parser.add_argument(
        "target", 
        metavar="TARGET",
        type=str,
        help="ID del NER, 'all' (comparar) o 'assembly' (unir todos)."
    )
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU (Default: 0.5)")
    return parser


def load_data(notes_path: Path, gt_path: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
    print(f"[Loader] Cargando datos de {notes_path.parent}...")
    with open(notes_path, 'r', encoding='utf-8') as f:
        notes_list = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    return {i['note_id']: i['text'] for i in notes_list}, {i['note_id']: i['annotations'] for i in gt_list}


def load_ner_worker(ner_id: str, config: Dict[str, Any]) -> Any:
    module_name = config['module']
    class_name = config['class']
    init_args = {k: v for k, v in config.items() if k not in ['module', 'class']}
    
    print(f"[WorkerLoader] Cargando worker '{ner_id}': {module_name}.{class_name}")
    try:
        module = importlib.import_module(module_name)
        NerClass = getattr(module, class_name)
        return NerClass(**init_args)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el worker '{ner_id}': {e}")
        return None


def get_detailed_matches(predictions, ground_truth, iou_threshold=0.5):
    matched_gt_indices = set()
    matched_pred_indices = set()
    matched_pairs = []

    for pred_idx, pred in enumerate(predictions):
        best_match_gt_idx = -1
        best_iou = -1.0
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_indices: continue
            iou = calculate_iou(pred, gt)
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match_gt_idx = gt_idx
        
        if best_match_gt_idx != -1:
            matched_pairs.append((pred, ground_truth[best_match_gt_idx], best_iou))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(best_match_gt_idx)

    unmatched_preds = [predictions[i] for i in range(len(predictions)) if i not in matched_pred_indices]
    unmatched_gts = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt_indices]
    return matched_pairs, unmatched_preds, unmatched_gts


def deduplicate_predictions(predictions: List[Dict]) -> List[Dict]:
    """Elimina predicciones duplicadas exactas (mismo start/end)."""
    unique = []
    seen = set()
    for p in predictions:
        # Usamos tupla (start, end) como clave única
        key = (p['start'], p['end'])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def print_detailed_report(note_text, matched_pairs, unmatched_preds, unmatched_gts):
    print("    -----------------------------------------------------------------")
    print("    INFORME DETALLADO (SPAN IoU > 0.5)")
    print("    -----------------------------------------------------------------")
    print(f"\n    ✅ Aciertos (TPs): {len(matched_pairs)}")
    if matched_pairs:
        print(f"      {'PREDICCIÓN (NER)':<45} | {'GROUND TRUTH':<45} | {'IoU':<5}")
        print("      " + "-" * 99)
        for pred, gt, iou in matched_pairs[:10]: # Limitamos a 10 para no saturar
            p_txt = note_text[pred['start']:pred['end']].replace('\n', ' ')[:40]
            g_txt = note_text[gt['start']:gt['end']].replace('\n', ' ')[:40]
            print(f"      {f'[{pred['start']}:{pred['end']}] {p_txt}':<45} | {f'[{gt['start']}:{gt['end']}] {g_txt}':<45} | {iou:.2f}")
        if len(matched_pairs) > 10: print(f"      ... y {len(matched_pairs)-10} más")

    print(f"\n    ⚠️ Falsos Negativos (FNs - Perdidos): {len(unmatched_gts)}")
    if unmatched_gts:
        print(f"      {'GROUND TRUTH FALTANTE':<45}")
        print("      " + "-" * 45)
        for gt in unmatched_gts[:10]:
            g_txt = note_text[gt['start']:gt['end']].replace('\n', ' ')[:40]
            print(f"      {f'[{gt['start']}:{gt['end']}] {g_txt}':<45}")
        if len(unmatched_gts) > 10: print(f"      ... y {len(unmatched_gts)-10} más")


def run_benchmark(ner_ids: List[str], registry: Dict, notes_data: Dict, gt_data: Dict, iou_threshold: float, mode: str) -> List[Dict]:
    results_table = []

    # Si estamos en modo assembly, tratamos la lista de IDs como un solo "Super-Modelo"
    if mode == 'assembly':
        print(f"\n=======================================================")
        print(f"🧩 MODO ASSEMBLY: Uniendo {len(ner_ids)} modelos...")
        print(f"   Modelos: {', '.join(ner_ids)}")
        print(f"=======================================================")
        
        # 1. Cargar todos los workers
        workers = []
        for nid in ner_ids:
            if nid in registry:
                w = load_ner_worker(nid, registry[nid])
                if w: workers.append(w)
        
        if not workers:
            print("[ERROR] No se pudo cargar ningún worker para el ensamblaje.")
            return []

        all_predictions = {}
        per_note_metrics = {}
        total_time = 0

        # 2. Procesar notas con el ensamble
        print("[INFO] Ejecutando ensamble sobre notas...")
        for note_id, text in notes_data.items():
            gt_annotations = gt_data.get(note_id, [])
            start_time = time.time()
            
            # A. Obtener predicciones de TODOS los workers
            combined_preds = []
            for w in workers:
                combined_preds.extend(w.extract_entities(text))
            
            # B. Deduplicar (Unión)
            unique_preds = deduplicate_predictions(combined_preds)
            
            note_time = time.time() - start_time
            total_time += note_time
            all_predictions[note_id] = unique_preds
            
            # C. Calcular métricas de la nota
            matched, un_p, un_g = get_detailed_matches(unique_preds, gt_annotations, iou_threshold)
            tp, fp, fn = len(matched), len(un_p), len(un_g)
            metrics = _calculate_pr_f1(tp, fp, fn)
            per_note_metrics[note_id] = metrics

            print(f"\n  --- Nota {note_id} (Ensamble) ---")
            print(f"    Tiempo Combinado: {note_time:.2f}s")
            print(f"    Brutas: {len(combined_preds)} -> Únicas: {len(unique_preds)}")
            print(f"    TP: {tp:<3} | FP: {fp:<3} | FN: {fn:<3}")
            print(f"    Recall: {metrics['recall']:.4f} | Precisión: {metrics['precision']:.4f}")
            
            print_detailed_report(text, matched, un_p, un_g)

        # D. Métricas Globales del Ensamble
        micro = calculate_ner_micro_f1(all_predictions, gt_data, iou_threshold)
        f1_scores = [m['f1'] for m in per_note_metrics.values()]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        results_table.append({
            "NER ID": "ASSEMBLY_UNION",
            "F1-Micro": micro['f1'],
            "F1-Macro": macro_f1,
            "Precision": micro['precision'],
            "Recall": micro['recall'],
            "TP": micro['tp'], "FP": micro['fp'], "FN": micro['fn'],
            "Tiempo (s)": total_time
        })

    # Modo estándar (Iterar uno por uno)
    else:
        for ner_id in ner_ids:
            print(f"\n=== DIAGNOSTICANDO: {ner_id} ===")
            config = registry.get(ner_id)
            if not config: continue
            worker = load_ner_worker(ner_id, config)
            if not worker: continue
            
            all_predictions = {}
            per_note_metrics = {}
            total_time = 0
            
            for note_id, text in notes_data.items():
                gt_annotations = gt_data.get(note_id, [])
                start_time = time.time()
                preds = worker.extract_entities(text)
                total_time += (time.time() - start_time)
                all_predictions[note_id] = preds
                
                matched, un_p, un_g = get_detailed_matches(preds, gt_annotations, iou_threshold)
                per_note_metrics[note_id] = _calculate_pr_f1(len(matched), len(un_p), len(un_g))

            micro = calculate_ner_micro_f1(all_predictions, gt_data, iou_threshold)
            f1_scores = [m['f1'] for m in per_note_metrics.values()]
            macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            
            results_table.append({
                "NER ID": ner_id,
                "F1-Micro": micro['f1'], "F1-Macro": macro,
                "Precision": micro['precision'], "Recall": micro['recall'],
                "TP": micro['tp'], "FP": micro['fp'], "FN": micro['fn'],
                "Tiempo (s)": total_time
            })

    return results_table


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f: registry = json.load(f)
        notes_data, gt_data = load_data(NOTES_PATH, GT_PATH)
    except FileNotFoundError:
        print("[ERROR] No se encontraron los ficheros de datos o config.")
        return

    mode = 'single'
    if args.target.lower() == 'all':
        ids = list(registry.keys())
    elif args.target.lower() == 'assembly':
        ids = list(registry.keys())
        mode = 'assembly' # Activamos el modo ensamble
    else:
        ids = [args.target]
        if args.target not in registry:
            print(f"[ERROR] ID '{args.target}' no encontrado.")
            return

    results = run_benchmark(ids, registry, notes_data, gt_data, args.iou, mode)
    
    if results:
        results.sort(key=lambda x: x['F1-Micro'], reverse=True)
        print("\n" + "="*110)
        print(f"RESULTADOS FINALES ({'ENSAMBLE' if mode=='assembly' else 'COMPARATIVA'}) - IoU @ {args.iou}")
        print("="*110)
        header = ["NER ID", "F1-Mic", "F1-Mac", "Prec", "Rec", "TP", "FP", "FN", "Time"]
        print(f"{header[0]:<20} | {header[1]:<7} | {header[2]:<7} | {header[3]:<7} | {header[4]:<7} | {header[5]:<4} | {header[6]:<4} | {header[7]:<4} | {header[8]:<6}")
        print("-" * 110)
        for r in results:
            print(f"{r['NER ID']:<20} | {r['F1-Micro']:.4f} | {r['F1-Macro']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['TP']:<4} | {r['FP']:<4} | {r['FN']:<4} | {r['Tiempo (s)']:<6.2f}")
        print("="*110)

if __name__ == "__main__":
    main()