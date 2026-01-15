import json
import os
from datetime import datetime

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_SYSTEM_FILE = os.path.join(BASE_DIR, "outputs/ground_truth.json")
BASELINE_FILE = os.path.join(BASE_DIR, "../test/ground_truth.json") 
HTML_OUTPUT = os.path.join(BASE_DIR, "outputs/final_report.html")

# --- CSS STYLES ---
CSS = """
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #ddd; }
    .metric-val { font-size: 2em; font-weight: bold; color: #2980b9; }
    .metric-label { color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; }
    
    .note-card { border: 1px solid #e1e4e8; border-radius: 6px; margin-bottom: 30px; overflow: hidden; }
    .note-header { background: #ecf0f1; padding: 10px 15px; font-weight: bold; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; }
    .note-content { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
    .text-panel { padding: 15px; border-right: 1px solid #eee; font-size: 0.9em; color: #555; background: #fafafa; max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
    .comparison-panel { padding: 0; }
    
    table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
    th { background-color: #f8f9fa; color: #555; }
    
    .badge { padding: 3px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold; color: white; }
    .match { background-color: #27ae60; } 
    .extra { background-color: #2980b9; } 
    .miss { background-color: #c0392b; }  
    
    .row-match { background-color: #eafaf1; }
    .row-extra { background-color: #ebf5fb; }
    .row-miss { background-color: #fdedec; }
</style>
"""

def generate_report():
    print(f"📊 Generando reporte comparativo (Modo Código)...")
    
    # 1. Load Data
    try:
        with open(CURRENT_SYSTEM_FILE, 'r', encoding='utf-8') as f:
            # New System: Key is note_id (string)
            current_data = {str(n.get('note_id', n.get('id'))): n for n in json.load(f)}
        
        with open(BASELINE_FILE, 'r', encoding='utf-8') as f:
            # Opus System: Key is note_id (int -> convert to string to match)
            baseline_data = {str(n.get('note_id')): n for n in json.load(f)}
            
        print(f"   -> Cargado Sistema Actual: {len(current_data)} notas")
        print(f"   -> Cargado Opus 4.5: {len(baseline_data)} notas")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 2. Metrics Accumulators
    total_tp, total_fp, total_fn = 0, 0, 0
    note_reports = []

    # 3. Compare common notes
    common_ids = sorted(list(set(current_data.keys()) & set(baseline_data.keys())))
    
    for nid in common_ids:
        c_note = current_data[nid]
        b_note = baseline_data[nid]
        
        # A. Get Current System Codes & Text Mapping
        # Structure: list of objects {'entity_text': '...', 'code': '...'}
        curr_ents_list = c_note.get('ground_truth_entities', c_note.get('final_entities', []))
        
        curr_code_map = {} # Code -> List of texts (one code might appear multiple times with diff text)
        curr_codes = set()
        
        for ent in curr_ents_list:
            code = ent.get('code', ent.get('final_code', 'NULL')).strip()
            text = ent.get('entity_text', ent.get('text', '')).strip()
            
            if code not in ["NULL", "MISSING"]:
                curr_codes.add(code)
                if code not in curr_code_map: curr_code_map[code] = []
                curr_code_map[code].append(text)

        # B. Get Opus Codes
        # Structure: found_codes: ["code1", "code2"]
        base_codes = set(b_note.get('found_codes', []))
        
        # C. Calculate Set Differences (CODE BASED)
        tp_codes = curr_codes.intersection(base_codes)
        fp_codes = curr_codes - base_codes
        fn_codes = base_codes - curr_codes
        
        total_tp += len(tp_codes)
        total_fp += len(fp_codes)
        total_fn += len(fn_codes)
        
        # D. Prepare Data for HTML Table
        rows = []
        
        # Matches (Green)
        for code in tp_codes:
            texts = ", ".join(list(set(curr_code_map.get(code, ["Unknown"]))))
            rows.append({"status": "match", "text": texts, "code": code})
            
        # New/Extra (Blue)
        for code in fp_codes:
            texts = ", ".join(list(set(curr_code_map.get(code, ["Unknown"]))))
            rows.append({"status": "extra", "text": texts, "code": code})
            
        # Missed (Red)
        for code in fn_codes:
            # Opus does not provide text, so we mark it unknown
            rows.append({"status": "miss", "text": "<i>Unknown (Opus Code Only)</i>", "code": code})
        
        # Sort rows by code
        rows.sort(key=lambda x: x['code'])

        # Get Text Content
        if 'original_note' in c_note and isinstance(c_note['original_note'], dict):
             text_content = f"HISTORY:\n{c_note['original_note'].get('history', '')}\n\nFINDINGS:\n{c_note['original_note'].get('findings', '')}"
        else:
             text_content = "Texto original no disponible."

        note_reports.append({
            "id": nid,
            "text": text_content,
            "rows": rows,
            "stats": (len(tp_codes), len(fp_codes), len(fn_codes))
        })

    # 4. Global Stats
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 5. Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>LLM Report: Voting vs Opus</title>
        {CSS}
    </head>
    <body>
        <div class="container">
            <h1>Reporte de Evaluación: Voting System vs Opus 4.5</h1>
            <p>Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-val">{f1:.2%}</div>
                    <div class="metric-label">F1 Score (Códigos)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{precision:.2%}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-val">{recall:.2%}</div>
                    <div class="metric-label">Recall</div>
                </div>
            </div>

            <h2>Detalle por Nota ({len(common_ids)} comparadas)</h2>
    """

    for note in note_reports:
        tp, fp, fn = note['stats']
        local_f1 = 2 * (tp / (tp+fp) * tp / (tp+fn)) / (tp/(tp+fp) + tp/(tp+fn)) if (tp+fp > 0 and tp+fn > 0) else 0
        if tp == 0 and fp == 0 and fn == 0: local_f1 = 0 # Handle edge case
        
        # Safe F1 calc
        if (2*tp + fp + fn) > 0:
             local_f1 = (2 * tp) / (2 * tp + fp + fn)
        else:
             local_f1 = 0

        html += f"""
        <div class="note-card">
            <div class="note-header">
                <span>Nota ID: {note['id']}</span>
                <span>F1 Local: {local_f1:.2f}</span>
            </div>
            <div class="note-content">
                <div class="text-panel">{note['text']}</div>
                <div class="comparison-panel">
                    <table>
                        <thead>
                            <tr>
                                <th>Estado</th>
                                <th>Texto (Nuevo Sist.)</th>
                                <th>Código</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for row in note['rows']:
            badge_cls = "match" if row['status'] == "match" else "extra" if row['status'] == "extra" else "miss"
            row_cls = "row-match" if row['status'] == "match" else "row-extra" if row['status'] == "extra" else "row-miss"
            status_text = "MATCH" if row['status'] == "match" else "NUEVO" if row['status'] == "extra" else "MISS"
            
            html += f"""
            <tr class="{row_cls}">
                <td><span class="badge {badge_cls}">{status_text}</span></td>
                <td>{row['text']}</td>
                <td>{row['code']}</td>
            </tr>
            """

        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """

    html += "</div></body></html>"

    with open(HTML_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Reporte Generado: {HTML_OUTPUT}")

if __name__ == "__main__":
    generate_report()