# benchmarks/diagnose_NER.py
#
# OBJETIVO:
# Orquestador SOTA para diagnosticar y comparar múltiples modelos NER
# de forma aislada.
#
# LÓGICA:
# 1. Lee un ID de modelo (ej: 'scibert') o 'all' desde la línea de comandos.
# 2. Carga el "menú" de modelos desde 'config/ner_registry.json'.
# 3. Carga dinámicamente el "worker" de 'src/NER/' (ej: spacy_ner.py).
# 4. Ejecuta el worker sobre las 5 notas.
# 5. Compara las predicciones (solo spans) contra el 'ground_truth.json'
#    usando métricas SOTA (Micro y Macro F1-Score con IoU).
# 6. Imprime una tabla comparativa de resultados Y un desglose detallado por nota.

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
    _find_ner_span_matches,  # Importa la función de matching de SOLO-NER
    _calculate_pr_f1,
    calculate_iou         # Importa IoU para el reporte detallado
)

# --- Constantes ---
CONFIG_PATH = PROJECT_ROOT / "config" / "ner_registry.json"
NOTES_PATH = PROJECT_ROOT / "data" / "notes.json"
GT_PATH = PROJECT_ROOT / "data" / "ground_truth.json"


def setup_argparser() -> argparse.ArgumentParser:
    """Configura el parser de argumentos con ayuda y ejemplos."""
    
    formatter = argparse.RawDescriptionHelpFormatter
    
    parser = argparse.ArgumentParser(
        description="Diagnóstico Modular de Modelos NER (Benchmark SOTA).",
        formatter_class=formatter,
        epilog="""
-------------------------------------------------------------------
CASOS DE USO DE EJEMPLO:
-------------------------------------------------------------------

1. Probar un solo modelo NER (ej: 'scispacy_bc5cdr'):
   python benchmarks/diagnose_NER.py scispacy_bc5cdr

2. Probar TODOS los modelos registrados en ner_registry.json:
   python benchmarks/diagnose_NER.py all

3. Mostrar este mensaje de ayuda:
   python benchmarks/diagnose_NER.py -h
"""
    )
    
    # --- ARREGLO: Usar un argumento posicional ---
    parser.add_argument(
        "target",  # <-- Argumento posicional
        metavar="NER_ID | all",
        type=str,
        help="El ID del NER a probar (definido en config/ner_registry.json) o 'all' para probarlos todos."
    )
    
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.5,
        help="Umbral de IoU para considerar un 'Acierto' (TP). Default: 0.5"
    )
    
    return parser


def load_data(notes_path: Path, gt_path: Path) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
    """Carga los JSON y los convierte en diccionarios para búsqueda rápida."""
    print(f"[Loader] Cargando datos de {notes_path.parent}...")
    with open(notes_path, 'r', encoding='utf-8') as f:
        notes_list = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_list = json.load(f)
    
    notes_data = {item['note_id']: item['text'] for item in notes_list}
    gt_data = {item['note_id']: item['annotations'] for item in gt_list}
    
    print(f"[Loader] Datos cargados: {len(notes_data)} notas, {len(gt_data)} ground truths.")
    return notes_data, gt_data


def load_ner_worker(ner_id: str, config: Dict[str, Any]) -> Any:
    """Carga dinámicamente el 'worker' NER basado en el ID del registro."""
    module_name = config['module']
    class_name = config['class']
    init_args = {k: v for k, v in config.items() if k not in ['module', 'class']}
    
    print(f"[WorkerLoader] Cargando worker '{ner_id}': {module_name}.{class_name}")
    
    try:
        module = importlib.import_module(module_name)
        NerClass = getattr(module, class_name)
        worker = NerClass(**init_args)
        return worker
    except ModuleNotFoundError:
        print(f"\n[ERROR] ModuleNotFoundError: No se pudo encontrar el módulo '{module_name}'.")
        print(f"Asegúrate de que el fichero '{module_name.replace('.', '/')}.py' existe.")
    except ImportError:
         print(f"\n[ERROR] ImportError: No se pudo importar '{class_name}' desde '{module_name}'.")
    except Exception as e:
        print(f"\n[ERROR] No se pudo cargar el worker '{ner_id}': {e}")
        
    return None


# --- (Lógica de tu script anterior) ---
def get_detailed_matches(predictions: List[Dict], 
                         ground_truth: List[Dict], 
                         iou_threshold: float = 0.5
                         ) -> Tuple[List, List, List]:
    """
    Compara predicciones y ground truth para obtener listas detalladas de
    TPs, FPs, y FNs.
    """
    matched_gt_indices: Set[int] = set()
    matched_pred_indices: Set[int] = set()
    matched_pairs: List[Tuple[Dict, Dict, float]] = []

    for pred_idx, pred in enumerate(predictions):
        best_match_gt_idx = -1
        best_iou = -1.0
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue
            
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


def print_detailed_report(
    note_text: str,
    matched_pairs: List, 
    unmatched_preds: List, 
    unmatched_gts: List
):
    """Imprime una tabla detallada de TPs, FPs, y FNs para una nota."""
    
    print("    -----------------------------------------------------------------")
    print("    INFORME DETALLADO DE ACIERTOS Y FALLOS (SPAN IoU > 0.5)")
    print("    -----------------------------------------------------------------")

    # 1. Imprimir Aciertos (TPs)
    print(f"\n    ✅ Aciertos (TPs): {len(matched_pairs)}")
    if matched_pairs:
        print(f"      {'PREDICCIÓN (NER)':<45} | {'GROUND TRUTH (GT)':<45} | {'IoU':<5}")
        print("      " + "-" * 99)
        for pred, gt, iou in matched_pairs:
            pred_text = note_text[pred['start']:pred['end']]
            gt_text = note_text[gt['start']:gt['end']]
            pred_span = f"[{pred['start']}:{pred['end']}] '{pred_text}'"
            gt_span = f"[{gt['start']}:{gt['end']}] '{gt_text}'"
            print(f"      {pred_span:<45} | {gt_span:<45} | {iou:.3f}")

    # 2. Imprimir Falsos Positivos (FPs)
    print(f"\n    ❌ Falsos Positivos (FPs): {len(unmatched_preds)}")
    if unmatched_preds:
        print("      (Spans que tu NER encontró pero no están en el GT)")
        print(f"      {'PREDICCIÓN (NER)':<45}")
        print("      " + "-" * 45)
        for pred in unmatched_preds:
            pred_text = note_text[pred['start']:pred['end']]
            pred_span = f"[{pred['start']}:{pred['end']}] '{pred_text}'"
            print(f"      {pred_span:<45}")

    # 3. Imprimir Falsos Negativos (FNs)
    print(f"\n    ⚠️ Falsos Negativos (FNs): {len(unmatched_gts)}")
    if unmatched_gts:
        print("      (Spans del GT que tu NER no encontró)")
        print(f"      {'GROUND TRUTH (GT)':<45}")
        print("      " + "-" * 45)
        for gt in unmatched_gts:
            gt_text = note_text[gt['start']:gt['end']]
            gt_span = f"[{gt['start']}:{gt['end']}] '{gt_text}'"
            print(f"      {gt_span:<45}")


def run_benchmark(
    ner_ids: List[str], 
    registry: Dict, 
    notes_data: Dict, 
    gt_data: Dict, 
    iou_threshold: float
) -> List[Dict]:
    """Ejecuta el benchmark para una lista de IDs de NER y devuelve los resultados."""
    results_table = []

    for ner_id in ner_ids:
        print(f"\n=======================================================")
        print(f"🔬 DIAGNOSTICANDO: {ner_id}")
        print(f"=======================================================")
        
        config = registry.get(ner_id)
        if not config:
            print(f"WARN: NERid '{ner_id}' no encontrado en ner_registry.json. Omitiendo.")
            continue

        worker = load_ner_worker(ner_id, config)
        if not worker:
            continue
            
        all_predictions: Dict[str, List[Dict]] = {}
        per_note_metrics: Dict[str, Dict] = {}
        total_time = 0

        # Bucle por nota para métricas detalladas
        print("[INFO] Procesando notas individualmente...")
        for note_id, text in notes_data.items():
            gt_annotations = gt_data.get(note_id, [])
            
            start_time = time.time()
            ner_predictions = worker.extract_entities(text) 
            note_time = time.time() - start_time
            
            total_time += note_time
            all_predictions[note_id] = ner_predictions
            
            # --- Calcular TPs, FPs, FNs para el reporte detallado ---
            matched_pairs, unmatched_preds, unmatched_gts = get_detailed_matches(
                ner_predictions, 
                gt_annotations, 
                iou_threshold
            )
            
            tp_i = len(matched_pairs)
            fp_i = len(unmatched_preds)
            fn_i = len(unmatched_gts)
            
            # Calcular P, R, F1 para la nota
            note_metrics = _calculate_pr_f1(tp_i, fp_i, fn_i)
            per_note_metrics[note_id] = note_metrics
            
            # --- Imprimir el informe detallado por nota ---
            print(f"\n  --- Nota {note_id} ---")
            print(f"    Tiempo: {note_time:.2f}s")
            print(f"    GT: {len(gt_annotations):<3} | Predichas: {len(ner_predictions):<3}")
            print(f"    TP: {tp_i:<3} | FP: {fp_i:<3} | FN: {fn_i:<3}")
            print(f"    Precisión: {note_metrics['precision']:.4f}")
            print(f"    Recall:    {note_metrics['recall']:.4f}")
            print(f"    F1-Score:  {note_metrics['f1']:.4f}")
            
            print_detailed_report(text, matched_pairs, unmatched_preds, unmatched_gts)


        print(f"\n[INFO] {ner_id} procesó {len(notes_data)} notas en {total_time:.2f}s")
        print(f"[INFO] Calculando métricas agregadas (IoU > {iou_threshold})...")

        # Calcular métricas Micro (agregadas)
        micro_metrics = calculate_ner_micro_f1(all_predictions, gt_data, iou_threshold)
        
        # Calcular métricas Macro (promedio "suma/5")
        f1_scores = [metrics['f1'] for metrics in per_note_metrics.values()]
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        results_table.append({
            "NER ID": ner_id,
            "F1-Micro": micro_metrics['f1'],
            "F1-Macro": macro_f1,
            "Precision": micro_metrics['precision'],
            "Recall": micro_metrics['recall'],
            "TP": micro_metrics['tp'],
            "FP": micro_metrics['fp'],
            "FN": micro_metrics['fn'],
            "Tiempo (s)": total_time
        })

    return results_table


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    # --- 1. Cargar Configuración y Datos ---
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        notes_data, gt_data = load_data(NOTES_PATH, GT_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] Fichero no encontrado: {e.filename}")
        print("Asegúrate de que 'config/ner_registry.json' y 'data/notes.json', 'data/ground_truth.json' existen.")
        return

    # --- ARREGLO 2: Leer el argumento posicional 'target' ---
    if args.target.lower() == 'all':
        ner_ids_to_run = list(registry.keys())
    else:
        ner_ids_to_run = [args.target]
        if args.target not in registry:
            print(f"[ERROR] NER ID '{args.target}' no encontrado en {CONFIG_PATH}")
            print(f"IDs disponibles: {list(registry.keys())}")
            return

    # --- 3. Ejecutar Benchmark ---
    results = run_benchmark(ner_ids_to_run, registry, notes_data, gt_data, args.iou)
    
    # --- 4. Imprimir Resultados ---
    if results:
        results.sort(key=lambda x: x['F1-Micro'], reverse=True)
        
        print("\n\n======================================================================================================")
        print(f"RESULTADOS FINALES DEL BENCHMARK DE NER (SOTA - SOLO SPAN IoU @ {args.iou})")
        print("======================================================================================================")
        
        header = [
            "NER ID", "F1-Micro", "F1-Macro", "Precision", 
            "Recall", "TP", "FP", "FN", "Tiempo (s)"
        ]
        print(f"{header[0]:<22} | {header[1]:<10} | {header[2]:<10} | {header[3]:<10} | {header[4]:<10} | {header[5]:<5} | {header[6]:<5} | {header[7]:<5} | {header[8]:<10}")
        print("-" * 102)
        
        for res in results:
            print(f"{res['NER ID']:<22} | {res['F1-Micro']:<10.4f} | {res['F1-Macro']:<10.4f} | {res['Precision']:<10.4f} | {res['Recall']:<10.4f} | {res['TP']:<5} | {res['FP']:<5} | {res['FN']:<5} | {res['Tiempo (s)']:<10.2f}")
        
        print("======================================================================================================")
        print(f"🏆 MEJOR NER (por F1-Micro): {results[0]['NER ID']} (F1: {results[0]['F1-Micro']:.4f})")
        print("======================================================================================================")

if __name__ == "__main__":
    main()