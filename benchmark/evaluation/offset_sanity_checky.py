#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# Rutas (ajústalas si hace falta)
ROOT = Path(__file__).resolve().parents[1]
pred_path = ROOT / "results" / "rag_gpt_test" / "predictions.csv"
gt_path   = ROOT / "data" / "train_annotations.csv"

def main():
    pred = pd.read_csv(pred_path)
    gt   = pd.read_csv(gt_path)

    # Emparejar por (note_id, concept_id, span_text) para analizar solo offsets
    key_cols = ["note_id","concept_id","span_text"]
    merged = pred.merge(gt, on=key_cols, suffixes=("_pred","_gt"))
    if merged.empty:
        print("No hay emparejamientos por (note_id, concept_id, span_text).")
        return

    # Diferencias de offsets
    merged["d_start"] = merged["start_pred"] - merged["start_gt"]
    merged["d_end"]   = merged["end_pred"]   - merged["end_gt"]

    print("\n=== HISTOGRAMAS DE DIFERENCIAS (pred - gt) ===")
    print("Δstart:")
    print(merged["d_start"].value_counts().sort_index().head(10))
    print("\nΔend:")
    print(merged["d_end"].value_counts().sort_index().head(10))

    # Heurísticas rápidas
    n = len(merged)
    off_by_one_end = (merged["d_end"]== -1).sum()/n
    plus_one_end   = (merged["d_end"]==  1).sum()/n
    same_end       = (merged["d_end"]==  0).sum()/n

    print(f"\nProporción Δend = -1: {off_by_one_end:.2%}  |  Δend = +1: {plus_one_end:.2%}  |  Δend = 0: {same_end:.2%}")
    print("Si Δend=-1 domina -> tu benchmark usa end EXCLUSIVO (pon EVAL_END_INCLUSIVE=false).")

    # Impacto del recorte (si muchos Δstart>0 o Δend<0 → tighten te acorta)
    many_trim = ((merged["d_start"]>0) | (merged["d_end"]<0)).mean()
    print(f"Posible impacto de recorte de bordes: {many_trim:.2%}")

if __name__ == "__main__":
    main()
