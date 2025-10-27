# -*- coding: utf-8 -*-
"""
Crea una ontología personalizada con los códigos SNOMED-CT del training set
y descripciones enriquecidas para mejorar la búsqueda semántica RAG
"""

import pandas as pd
import os

# Códigos SNOMED-CT con descripciones enriquecidas (basado en ground truth)
SNOMED_CONCEPTS = {
    # Stroke and cerebrovascular
    "230690007": "stroke cerebrovascular accident CVA brain attack ischemic stroke hemorrhagic stroke acute stroke cerebral stroke",
    
    # Symptoms and signs - MUY ESPECÍFICOS
    "13791008": "weakness general weakness motor weakness limb weakness arm weakness leg weakness loss of strength weak muscular weakness",
    "50582007": "hemiparesis left hemiparesis right hemiparesis left sided hemiparesis right sided hemiparesis hemiplegic hemiplegia one-sided paralysis",
    "8011004": "dysarthria slurred speech speech articulation disorder motor speech impairment difficulty speaking slurred articulation",
    
    # Hemorrhage  
    "50960005": "hemorrhage bleeding haemorrhage intracranial hemorrhage brain hemorrhage cerebral hemorrhage subarachnoid hemorrhage ICH SAH blood bleeding",
    
    # Infarct and ischemia
    "55342001": "infarct infarction cerebral infarct brain infarct ischemic infarct acute infarct stroke infarction tissue death necrosis ischemic stroke",
    
    # Procedures
    "433112001": "thrombectomy mechanical thrombectomy endovascular thrombectomy clot removal stent retriever clot extraction thrombectomy procedure",
    "77343006": "angiography cerebral angiography CT angiography CTA MR angiography MRA vessel imaging arteriography vascular imaging angiogram",
    
    # Anatomy
    "69930009": "middle cerebral artery MCA M1 segment M2 segment MCA territory cerebral artery brain artery",
    "26036001": "occlusion arterial occlusion vessel occlusion MCA occlusion vascular occlusion blockage obstruction vessel blockage",
    
    # Ischemic concepts
    "230691006": "penumbra ischemic penumbra tissue at risk salvageable tissue hypoperfusion peri-infarct penumbral tissue",
    "449894001": "recanalization revascularization vessel reopening reperfusion flow restoration vessel opening arterial recanalization",
    
    # Imaging
    "77477000": "CT computed tomography CAT scan head CT brain CT non-contrast CT NCCT CT scan imaging tomography",
    
    # Scales and scores - MUY ESPECÍFICO
    "450893003": "NIHSS National Institutes of Health Stroke Scale ASPECTS Alberta Stroke Program Early CT Score TICI Thrombolysis in Cerebral Infarction score scale assessment clinical scale neurological scale stroke scale",
    
    # Medications
    "387467008": "tPA tissue plasminogen activator alteplase thrombolysis thrombolytic therapy IV tPA intravenous tPA plasminogen activator",
    
    # Comorbidities
    "38341003": "hypertension high blood pressure HTN elevated blood pressure arterial hypertension BP high pressure",
    "73211009": "diabetes diabetes mellitus DM diabetic hyperglycemia glucose mellitus type 2 diabetes",
}

def create_custom_ontology():
    """
    Crea un CSV con la ontología personalizada
    """
    
    # Crear DataFrame
    data = []
    for concepto, narrativa in SNOMED_CONCEPTS.items():
        data.append({
            'concepto': concepto,
            'narrativa': narrativa
        })
    
    df = pd.DataFrame(data)
    
    # Guardar CSV
    output_path = os.path.join(
        os.path.dirname(__file__),
        'assets',
        'custom_stroke_ontology.csv'
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[OK] Ontología personalizada creada: {output_path}")
    print(f"[OK] Total conceptos: {len(df)}")
    print(f"\nConceptos incluidos:")
    for idx, row in df.iterrows():
        print(f"  - {row['concepto']}: {row['narrativa'][:60]}...")
    
    return output_path

if __name__ == "__main__":
    create_custom_ontology()
