#!/usr/bin/env python3
"""
Métricas de evaluación para SNOMED CT Entity Linking
Basado en las métricas oficiales de la competencia
"""


# TODO: # DEPRECATED?? revisar si esto está siendo usado en algún sitio


import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict

class EntityLinkingMetrics:
    """Calculador de métricas para entity linking"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_span_overlap(self, pred_start: int, pred_end: int, 
                             true_start: int, true_end: int) -> float:
        """Calcula el overlap entre dos spans de texto"""
        # Calcular intersección
        overlap_start = max(pred_start, true_start)
        overlap_end = min(pred_end, true_end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        # Calcular IoU (Intersection over Union)
        intersection = overlap_end - overlap_start
        union = max(pred_end, true_end) - min(pred_start, true_start)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_iou_metrics(self, predictions: pd.DataFrame, 
                            ground_truth: pd.DataFrame, 
                            iou_threshold: float = 0.5) -> Dict:
        """
        Calcula métricas IoU como en la competencia original
        
        Args:
            predictions: DataFrame con columnas [note_id, start, end, concept_id, span_text]
            ground_truth: DataFrame con columnas [note_id, start, end, concept_id, span_text]
            iou_threshold: Umbral mínimo de IoU para considerar match
        """
        
        # Agrupar por note_id
        results_by_note = {}
        
        for note_id in ground_truth['note_id'].unique():
            pred_note = predictions[predictions['note_id'] == note_id]
            true_note = ground_truth[ground_truth['note_id'] == note_id]
            
            # Calcular matches para esta nota
            matches = []
            used_predictions = set()
            used_ground_truth = set()
            
            for true_idx, true_row in true_note.iterrows():
                best_match = None
                best_iou = 0.0
                
                for pred_idx, pred_row in pred_note.iterrows():
                    if pred_idx in used_predictions:
                        continue
                    
                    # Calcular IoU espacial
                    spatial_iou = self.calculate_span_overlap(
                        pred_row['start'], pred_row['end'],
                        true_row['start'], true_row['end']
                    )
                    
                    if spatial_iou >= iou_threshold and spatial_iou > best_iou:
                        # Verificar si el concepto coincide
                        concept_match = pred_row['concept_id'] == true_row['concept_id']
                        
                        if concept_match:
                            best_match = (pred_idx, spatial_iou, True)
                            best_iou = spatial_iou
                        elif best_match is None:
                            # Guardar match espacial sin concepto correcto
                            best_match = (pred_idx, spatial_iou, False)
                            best_iou = spatial_iou
                
                if best_match is not None:
                    matches.append({
                        'true_idx': true_idx,
                        'pred_idx': best_match[0],
                        'iou': best_match[1],
                        'concept_correct': best_match[2],
                        'true_concept': true_row['concept_id'],
                        'pred_concept': pred_note.iloc[best_match[0] - pred_note.index[0]]['concept_id']
                    })
                    used_predictions.add(best_match[0])
                    used_ground_truth.add(true_idx)
            
            # Calcular métricas para esta nota
            true_positives = sum(1 for m in matches if m['concept_correct'])
            false_positives = len(pred_note) - len(used_predictions) + sum(1 for m in matches if not m['concept_correct'])
            false_negatives = len(true_note) - len(used_ground_truth)
            
            # Métricas por nota
            precision = true_positives / len(pred_note) if len(pred_note) > 0 else 0.0
            recall = true_positives / len(true_note) if len(true_note) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # IoU promedio para matches válidos
            valid_ious = [m['iou'] for m in matches if m['concept_correct']]
            avg_iou = np.mean(valid_ious) if valid_ious else 0.0
            
            results_by_note[note_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'avg_iou': avg_iou,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'matches': matches
            }
        
        # Calcular métricas globales
        total_tp = sum(r['true_positives'] for r in results_by_note.values())
        total_fp = sum(r['false_positives'] for r in results_by_note.values())
        total_fn = sum(r['false_negatives'] for r in results_by_note.values())
        
        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0
        
        # IoU macro-promedio
        all_ious = []
        for result in results_by_note.values():
            all_ious.extend([m['iou'] for m in result['matches'] if m['concept_correct']])
        macro_avg_iou = np.mean(all_ious) if all_ious else 0.0
        
        return {
            'global_metrics': {
                'precision': global_precision,
                'recall': global_recall,
                'f1': global_f1,
                'macro_avg_iou': macro_avg_iou,
                'total_predictions': len(predictions),
                'total_ground_truth': len(ground_truth),
                'total_true_positives': total_tp,
                'total_false_positives': total_fp,
                'total_false_negatives': total_fn
            },
            'per_note_metrics': results_by_note
        }
    
    def calculate_category_metrics(self, predictions: pd.DataFrame, 
                                 ground_truth: pd.DataFrame) -> Dict:
        """Calcula métricas por categoría de entidad (finding, procedure, body)"""
        
        # Mapeo de códigos SNOMED a categorías (simplificado)
        category_map = {
            # Findings
            "230690007": "finding",  # stroke
            "50582007": "finding",   # hemiparesis
            "87486003": "finding",   # aphasia
            "8011004": "finding",    # dysarthria
            "25064002": "finding",   # headache
            "422587007": "finding",  # nausea
            "422400008": "finding",  # vomiting
            "50960005": "finding",   # hemorrhage
            "55342001": "finding",   # infarct
            "52674009": "finding",   # ischemia
            "415582006": "finding",  # stenosis
            "26036001": "finding",   # occlusion
            "432101006": "finding",  # aneurysm
            "230691006": "finding",  # penumbra
            
            # Procedures
            "433112001": "procedure", # thrombectomy
            "77343006": "procedure",  # angiography
            "77477000": "procedure",  # CT
            "113091000": "procedure", # MRI
            "450893003": "procedure", # ASPECTS/NIHSS/TICI
            "387467008": "procedure", # tPA
            
            # Body structures
            "69930009": "body",      # middle cerebral artery
            "86547008": "body",      # internal carotid artery
            "67889009": "body",      # basilar artery
        }
        
        # Agregar categorías a los DataFrames
        def add_category(df):
            df = df.copy()
            df['category'] = df['concept_id'].map(category_map).fillna('unknown')
            return df
        
        pred_with_cat = add_category(predictions)
        true_with_cat = add_category(ground_truth)
        
        # Calcular métricas por categoría
        category_results = {}
        
        for category in ['finding', 'procedure', 'body', 'unknown']:
            pred_cat = pred_with_cat[pred_with_cat['category'] == category]
            true_cat = true_with_cat[true_with_cat['category'] == category]
            
            if len(true_cat) > 0:  # Solo calcular si hay ground truth para esta categoría
                cat_metrics = self.calculate_iou_metrics(pred_cat, true_cat)
                category_results[category] = cat_metrics['global_metrics']
        
        return category_results
    
    def print_results(self, results: Dict, strategy_name: str):
        """Imprime resultados de manera legible"""
        print(f"\n{'='*60}")
        print(f"RESULTADOS - {strategy_name}")
        print(f"{'='*60}")
        
        global_metrics = results['global_metrics']
        print(f"\n📊 MÉTRICAS GLOBALES:")
        print(f"   Precision: {global_metrics['precision']:.4f}")
        print(f"   Recall:    {global_metrics['recall']:.4f}")
        print(f"   F1-Score:  {global_metrics['f1']:.4f}")
        print(f"   Avg IoU:   {global_metrics['macro_avg_iou']:.4f}")
        
        print(f"\n📈 CONTEOS:")
        print(f"   Predicciones:     {global_metrics['total_predictions']}")
        print(f"   Ground Truth:     {global_metrics['total_ground_truth']}")
        print(f"   True Positives:   {global_metrics['total_true_positives']}")
        print(f"   False Positives:  {global_metrics['total_false_positives']}")
        print(f"   False Negatives:  {global_metrics['total_false_negatives']}")
        
        # Métricas por nota
        per_note = results['per_note_metrics']
        note_f1s = [metrics['f1'] for metrics in per_note.values()]
        note_ious = [metrics['avg_iou'] for metrics in per_note.values()]
        
        print(f"\n📋 POR NOTA:")
        print(f"   F1 promedio:  {np.mean(note_f1s):.4f} (std: {np.std(note_f1s):.4f})")
        print(f"   IoU promedio: {np.mean(note_ious):.4f} (std: {np.std(note_ious):.4f})")
        
        return global_metrics['f1']  # Retorna F1 para comparación
