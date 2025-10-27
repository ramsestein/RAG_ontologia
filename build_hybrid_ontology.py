"""
Script to create a hybrid ontology combining:
1. Missing concepts from train_annotations.csv (26 concepts)
2. Existing concepts from conceptos_con_narrativas.csv that are actually used (6 concepts)
3. Similar noise concepts to test model robustness (~30 concepts)

Author: Oriol Farrés
Date: October 27, 2025
"""

import pandas as pd
import os
from collections import Counter

# Manual SNOMED-CT definitions for the 26 missing concepts
# These are stroke-specific medical concepts with rich semantic descriptions
MISSING_CONCEPTS = {
    "77477000": {
        "preferred_term": "Alteplase",
        "synonyms": ["Activase", "tissue plasminogen activator", "tPA", "recombinant tissue plasminogen activator", "rtPA", "alteplase (substance)"],
        "definition": "Thrombolytic agent used in acute ischemic stroke treatment. Recombinant tissue plasminogen activator that converts plasminogen to plasmin to dissolve blood clots. Administered intravenously within 4.5 hours of stroke onset.",
        "semantic_type": "Substance",
        "parent": "372877000"  # Thrombolytic agent
    },
    "266257000": {
        "preferred_term": "Transient ischemic attack",
        "synonyms": ["TIA", "mini-stroke", "transient ischemic attack (disorder)", "transient cerebral ischemia", "temporary stroke"],
        "definition": "Temporary interruption of blood flow to the brain causing stroke-like symptoms lasting less than 24 hours. Warning sign for future stroke. No permanent brain damage. Risk factor for major stroke.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"  # Cerebrovascular accident
    },
    "230690007": {
        "preferred_term": "Cerebrovascular accident",
        "synonyms": ["stroke", "CVA", "brain attack", "cerebrovascular accident (disorder)", "cerebral infarction", "apoplexy"],
        "definition": "Acute neurological deficit resulting from interrupted blood supply to brain tissue. Can be ischemic (blocked artery) or hemorrhagic (ruptured vessel). Leading cause of disability and death. Symptoms include weakness, speech difficulties, vision problems.",
        "semantic_type": "Clinical Finding",
        "parent": "64586002"  # Cerebrovascular disease
    },
    "55342001": {
        "preferred_term": "Cerebral infarction",
        "synonyms": ["brain infarct", "ischemic stroke", "cerebral infarction (disorder)", "infarct", "ischemic brain injury"],
        "definition": "Death of brain tissue due to lack of blood supply. Result of ischemic stroke. Permanent neurological damage. Visualized on CT or MRI as hypodense area. Middle cerebral artery territory most common.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"  # Cerebrovascular accident
    },
    "77343006": {
        "preferred_term": "Cerebral angiography",
        "synonyms": ["angiography", "cerebral arteriography", "brain angiogram", "cerebral angiography (procedure)", "digital subtraction angiography"],
        "definition": "Imaging technique using contrast dye and X-rays to visualize cerebral blood vessels. Gold standard for detecting aneurysms, stenosis, occlusions. Used for stroke diagnosis and treatment planning. Interventional procedure.",
        "semantic_type": "Procedure",
        "parent": "241554009"  # Angiography
    },
    "13791008": {
        "preferred_term": "Muscle weakness",
        "synonyms": ["weakness", "muscular weakness", "asthenia", "muscle weakness (finding)", "paresis", "reduced muscle strength"],
        "definition": "Reduced ability to exert muscle force. Common stroke symptom. Unilateral weakness suggests cerebral hemisphere lesion. Assessed using Medical Research Council scale. Can affect face, arm, or leg.",
        "semantic_type": "Clinical Finding",
        "parent": "367534002"  # Reduced muscle tone
    },
    "449894001": {
        "preferred_term": "Magnetic resonance angiography",
        "synonyms": ["MRA", "MR angiography", "magnetic resonance angiography (procedure)", "MR angiogram"],
        "definition": "Non-invasive imaging of blood vessels using magnetic resonance imaging. Visualizes arterial stenosis, occlusion, aneurysm. No radiation or iodinated contrast. Time-of-flight or contrast-enhanced techniques. Essential for stroke evaluation.",
        "semantic_type": "Procedure",
        "parent": "113091000"  # Magnetic resonance imaging
    },
    "230691006": {
        "preferred_term": "Acute stroke",
        "synonyms": ["acute cerebrovascular accident", "acute CVA", "acute stroke (disorder)", "acute brain attack"],
        "definition": "Sudden onset cerebrovascular accident requiring immediate medical intervention. Time-critical emergency. Window for thrombolytic therapy is 4.5 hours. Rapid assessment using NIHSS scale. CT scan to differentiate ischemic vs hemorrhagic.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"  # Cerebrovascular accident
    },
    "67889009": {
        "preferred_term": "Ischemic stroke",
        "synonyms": ["ischemic cerebrovascular accident", "ischemic CVA", "ischemic stroke (disorder)", "thrombotic stroke", "embolic stroke"],
        "definition": "Stroke caused by arterial occlusion blocking blood flow to brain. 87% of all strokes. Thrombotic or embolic origin. Treated with thrombolysis or thrombectomy. Risk factors include hypertension, diabetes, atrial fibrillation.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"  # Cerebrovascular accident
    },
    "432101006": {
        "preferred_term": "Computed tomography of head",
        "synonyms": ["CT scan of head", "brain CT", "head CT", "computed tomography of head (procedure)", "cranial CT"],
        "definition": "First-line imaging for acute stroke. Rapidly distinguishes hemorrhagic from ischemic stroke. Identifies early ischemic changes. ASPECTS score quantifies extent. Non-contrast CT for initial evaluation.",
        "semantic_type": "Procedure",
        "parent": "77477000"  # Computed tomography
    },
    "25064002": {
        "preferred_term": "Headache",
        "synonyms": ["cephalalgia", "head pain", "headache (finding)", "cranial pain"],
        "definition": "Pain in head or upper neck region. Common in hemorrhagic stroke. Sudden severe headache suggests subarachnoid hemorrhage. Assessed for quality, location, duration, associated symptoms.",
        "semantic_type": "Clinical Finding",
        "parent": "271681002"  # Pain
    },
    "73211009": {
        "preferred_term": "Diabetes mellitus",
        "synonyms": ["diabetes", "DM", "diabetes mellitus (disorder)", "hyperglycemia"],
        "definition": "Metabolic disorder with chronic hyperglycemia. Major stroke risk factor. Doubles stroke risk. Causes atherosclerosis and small vessel disease. Managed with glucose control, lifestyle modification.",
        "semantic_type": "Clinical Finding",
        "parent": "362969004"  # Endocrine disorder
    },
    "387467008": {
        "preferred_term": "Aspirin",
        "synonyms": ["acetylsalicylic acid", "ASA", "aspirin (substance)", "antiplatelet agent"],
        "definition": "Antiplatelet medication for stroke prevention. Inhibits cyclooxygenase and thromboxane synthesis. Secondary prevention after ischemic stroke or TIA. Reduces recurrent stroke risk by 25%. Typical dose 75-325mg daily.",
        "semantic_type": "Substance",
        "parent": "372664007"  # Antiplatelet agent
    },
    "433112001": {
        "preferred_term": "Mechanical thrombectomy",
        "synonyms": ["thrombectomy", "endovascular thrombectomy", "mechanical thrombectomy (procedure)", "clot retrieval", "stent retriever"],
        "definition": "Endovascular procedure to mechanically remove blood clot from brain artery. Treatment for large vessel occlusion stroke. Time window up to 24 hours in selected patients. Uses stent retrievers or aspiration catheters. Improves functional outcomes.",
        "semantic_type": "Procedure",
        "parent": "64586002"  # Vascular procedure
    },
    "113091000": {
        "preferred_term": "Magnetic resonance imaging",
        "synonyms": ["MRI", "MR imaging", "magnetic resonance imaging (procedure)", "MR scan", "nuclear magnetic resonance imaging"],
        "definition": "Advanced imaging using magnetic fields and radio waves. Superior soft tissue contrast. Detects acute infarction with diffusion-weighted imaging. Identifies penumbra with perfusion imaging. No ionizing radiation.",
        "semantic_type": "Procedure",
        "parent": "363679005"  # Imaging procedure
    },
    "20262006": {
        "preferred_term": "Atrial fibrillation",
        "synonyms": ["AFib", "AF", "atrial fibrillation (disorder)", "auricular fibrillation"],
        "definition": "Irregular rapid heart rhythm. Major cardioembolic stroke risk factor. Increases stroke risk 5-fold. Causes stasis and thrombus formation in left atrium. Managed with anticoagulation.",
        "semantic_type": "Clinical Finding",
        "parent": "698247007"  # Cardiac arrhythmia
    },
    "38341003": {
        "preferred_term": "Hypertensive disorder",
        "synonyms": ["hypertension", "high blood pressure", "HTN", "hypertensive disorder (disorder)", "arterial hypertension"],
        "definition": "Elevated blood pressure above 140/90 mmHg. Leading modifiable stroke risk factor. Causes small vessel disease and atherosclerosis. Blood pressure control reduces stroke risk by 40%. Target <120/80 for stroke prevention.",
        "semantic_type": "Clinical Finding",
        "parent": "49601007"  # Cardiovascular disorder
    },
    "50582007": {
        "preferred_term": "Hemiparesis",
        "synonyms": ["weakness on one side", "unilateral weakness", "hemiparesis (finding)", "partial paralysis"],
        "definition": "Weakness affecting one side of body. Classic stroke presentation. Suggests contralateral cerebral hemisphere lesion. Face-arm-leg distribution. Assessed with NIHSS motor items. Upper motor neuron pattern.",
        "semantic_type": "Clinical Finding",
        "parent": "13791008"  # Muscle weakness
    },
    "69449002": {
        "preferred_term": "Cerebral hemorrhage",
        "synonyms": ["intracerebral hemorrhage", "ICH", "brain hemorrhage", "cerebral hemorrhage (disorder)", "hemorrhagic stroke"],
        "definition": "Bleeding into brain parenchyma. 13% of all strokes. Higher mortality than ischemic stroke. Caused by hypertension, amyloid angiopathy, anticoagulation. Contraindication for thrombolysis. Surgical evacuation considered for large hematomas.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"  # Cerebrovascular accident
    },
    "422400008": {
        "preferred_term": "Vomiting",
        "synonyms": ["emesis", "vomiting (finding)", "throwing up"],
        "definition": "Forceful expulsion of stomach contents. Can indicate posterior circulation stroke. Associated with increased intracranial pressure. Common in cerebellar stroke or hemorrhage.",
        "semantic_type": "Clinical Finding",
        "parent": "300391003"  # Gastrointestinal symptom
    },
    "21454007": {
        "preferred_term": "Atherosclerosis",
        "synonyms": ["arteriosclerosis", "arterial plaque", "atherosclerosis (disorder)", "vascular disease"],
        "definition": "Progressive narrowing of arteries by plaque formation. Major cause of ischemic stroke. Affects carotid and intracranial arteries. Risk factors include hyperlipidemia, smoking, diabetes. Carotid endarterectomy for significant stenosis.",
        "semantic_type": "Clinical Finding",
        "parent": "400047006"  # Peripheral vascular disease
    },
    "49436004": {
        "preferred_term": "Atrial flutter",
        "synonyms": ["AFL", "atrial flutter (disorder)", "auricular flutter"],
        "definition": "Regular rapid atrial arrhythmia. Stroke risk similar to atrial fibrillation. Sawtooth pattern on ECG. Anticoagulation recommended. Can convert to atrial fibrillation.",
        "semantic_type": "Clinical Finding",
        "parent": "698247007"  # Cardiac arrhythmia
    },
    "87486003": {
        "preferred_term": "Aphasia",
        "synonyms": ["language disorder", "dysphasia", "aphasia (finding)", "speech disorder"],
        "definition": "Impairment of language comprehension or production. Results from dominant hemisphere lesion. Broca's aphasia: expressive deficit. Wernicke's aphasia: receptive deficit. NIHSS language assessment.",
        "semantic_type": "Clinical Finding",
        "parent": "87486003"  # Cognitive disorder
    },
    "52674009": {
        "preferred_term": "Carotid artery stenosis",
        "synonyms": ["carotid stenosis", "carotid narrowing", "carotid artery stenosis (disorder)", "internal carotid artery stenosis"],
        "definition": "Narrowing of carotid artery lumen. Major cause of ischemic stroke. >70% stenosis is significant. Evaluated with carotid ultrasound, CTA, or MRA. Treatment includes carotid endarterectomy or stenting.",
        "semantic_type": "Clinical Finding",
        "parent": "64586002"  # Cerebrovascular disease
    },
    "422587007": {
        "preferred_term": "Nausea",
        "synonyms": ["feeling sick", "nausea (finding)", "queasy"],
        "definition": "Sensation of impending vomiting. Can indicate posterior circulation stroke. Associated with vertigo in cerebellar or brainstem stroke. Part of vestibular syndrome.",
        "semantic_type": "Clinical Finding",
        "parent": "404640003"  # Digestive symptom
    },
    "8011004": {
        "preferred_term": "Dysarthria",
        "synonyms": ["slurred speech", "dysarthria (finding)", "articulation disorder", "motor speech disorder"],
        "definition": "Motor speech disorder with impaired articulation. Results from weakness or incoordination of speech muscles. Suggests brainstem or cerebellar lesion. Distinguished from aphasia. NIHSS dysarthria item.",
        "semantic_type": "Clinical Finding",
        "parent": "29164008"  # Speech disorder
    }
}

# Noise concepts: similar medical concepts that are NOT in the training data
# This tests if the model can handle distractors
NOISE_CONCEPTS = {
    "50960005": {
        "preferred_term": "Hemorrhage",
        "synonyms": ["bleeding", "hemorrhage (finding)", "blood loss"],
        "definition": "Escape of blood from vessels. Can be external or internal. Classified by location and severity. Major cause of hemorrhagic stroke. Managed with hemostasis and blood products.",
        "semantic_type": "Clinical Finding",
        "parent": "50960005"
    },
    "69930009": {
        "preferred_term": "Middle cerebral artery",
        "synonyms": ["MCA", "middle cerebral artery (structure)", "sylvian artery"],
        "definition": "Largest cerebral artery. Supplies lateral cerebral hemisphere. Most common site of ischemic stroke. Territory includes motor and sensory cortex. M1 and M2 segments.",
        "semantic_type": "Body Structure",
        "parent": "57370006"
    },
    "95457000": {
        "preferred_term": "Subarachnoid hemorrhage",
        "synonyms": ["SAH", "subarachnoid hemorrhage (disorder)", "subarachnoid bleeding"],
        "definition": "Bleeding into subarachnoid space. Usually from ruptured aneurysm. Sudden severe 'thunderclap' headache. High mortality and morbidity. CT shows blood in basal cisterns. Angiography for aneurysm detection.",
        "semantic_type": "Clinical Finding",
        "parent": "69449002"
    },
    "230692004": {
        "preferred_term": "Brainstem stroke",
        "synonyms": ["brainstem infarction", "posterior circulation stroke", "brainstem stroke (disorder)"],
        "definition": "Stroke affecting brainstem structures. Causes crossed neurological deficits. Symptoms include vertigo, diplopia, dysphagia, dysarthria. Vertebrobasilar territory. Basilar artery occlusion is severe.",
        "semantic_type": "Clinical Finding",
        "parent": "230690007"
    },
    "277956001": {
        "preferred_term": "Carotid endarterectomy",
        "synonyms": ["CEA", "carotid endarterectomy (procedure)", "carotid artery surgery"],
        "definition": "Surgical removal of atherosclerotic plaque from carotid artery. Stroke prevention in symptomatic stenosis >70% or asymptomatic >80%. Reduces stroke risk. Complications include perioperative stroke, nerve injury.",
        "semantic_type": "Procedure",
        "parent": "304042000"
    },
    "84757009": {
        "preferred_term": "Epileptic seizure",
        "synonyms": ["seizure", "epileptic seizure (finding)", "convulsion", "fit"],
        "definition": "Abnormal electrical activity in brain. Can occur after stroke. Post-stroke epilepsy affects 5-10%. Early seizures within 7 days. Late seizures suggest gliosis. Treated with antiepileptic drugs.",
        "semantic_type": "Clinical Finding",
        "parent": "313307000"
    },
    "161891005": {
        "preferred_term": "Smoking",
        "synonyms": ["tobacco use", "smoking (finding)", "cigarette smoking"],
        "definition": "Inhalation of tobacco smoke. Major modifiable stroke risk factor. Doubles stroke risk. Causes atherosclerosis and hypercoagulability. Smoking cessation reduces risk within 2-4 years.",
        "semantic_type": "Clinical Finding",
        "parent": "699214001"
    },
    "13645005": {
        "preferred_term": "Dementia",
        "synonyms": ["dementia (disorder)", "cognitive decline", "neurocognitive disorder"],
        "definition": "Progressive cognitive impairment. Vascular dementia from multiple strokes. Post-stroke cognitive decline common. Affects memory, executive function, language. Risk factor for recurrent stroke.",
        "semantic_type": "Clinical Finding",
        "parent": "52448006"
    },
    "267036007": {
        "preferred_term": "Dyspnea",
        "synonyms": ["shortness of breath", "dyspnea (finding)", "breathlessness"],
        "definition": "Difficulty breathing. Can indicate cardiac cause of stroke. Pulmonary embolism complication of immobility. Assessed in stroke patients for aspiration risk.",
        "semantic_type": "Clinical Finding",
        "parent": "271825005"
    },
    "271782001": {
        "preferred_term": "Drowsiness",
        "synonyms": ["somnolence", "drowsiness (finding)", "lethargy"],
        "definition": "Reduced level of consciousness. Indicates large stroke or brainstem involvement. Glasgow Coma Scale assessment. May progress to coma. Concerning sign requiring monitoring.",
        "semantic_type": "Clinical Finding",
        "parent": "419045004"
    },
    "36955009": {
        "preferred_term": "Loss of consciousness",
        "synonyms": ["syncope", "fainting", "loss of consciousness (finding)", "blackout"],
        "definition": "Transient loss of awareness. Rare in stroke except large hemorrhage or basilar occlusion. Differential includes seizure, cardiac arrhythmia. Suggests severe event.",
        "semantic_type": "Clinical Finding",
        "parent": "419045004"
    },
    "386661006": {
        "preferred_term": "Fever",
        "synonyms": ["pyrexia", "fever (finding)", "elevated temperature", "hyperthermia"],
        "definition": "Elevated body temperature. Common after stroke. Associated with worse outcomes. May indicate infection complication. Aspiration pneumonia concern. Target normothermia in stroke management.",
        "semantic_type": "Clinical Finding",
        "parent": "386661006"
    },
    "271594007": {
        "preferred_term": "Ataxia",
        "synonyms": ["incoordination", "ataxia (finding)", "loss of coordination"],
        "definition": "Impaired coordination and balance. Indicates cerebellar stroke. Gait ataxia, limb ataxia, truncal ataxia. NIHSS cerebellar testing. Can cause falls.",
        "semantic_type": "Clinical Finding",
        "parent": "85828009"
    },
    "129565002": {
        "preferred_term": "Diplopia",
        "synonyms": ["double vision", "diplopia (disorder)", "binocular vision disorder"],
        "definition": "Double vision from eye misalignment. Indicates brainstem or cranial nerve involvement. Posterior circulation stroke. Cranial nerve III, IV, or VI palsy. Internuclear ophthalmoplegia.",
        "semantic_type": "Clinical Finding",
        "parent": "246636008"
    },
    "271807003": {
        "preferred_term": "Dysphagia",
        "synonyms": ["swallowing difficulty", "dysphagia (disorder)", "difficulty swallowing"],
        "definition": "Impaired swallowing. Common after stroke affecting 50%. Aspiration pneumonia risk. Requires swallow assessment. Modified diet or tube feeding. Improves with rehabilitation.",
        "semantic_type": "Clinical Finding",
        "parent": "288939007"
    },
    "44695005": {
        "preferred_term": "Paralysis",
        "synonyms": ["plegia", "paralysis (finding)", "complete loss of motor function"],
        "definition": "Complete loss of muscle function. Hemiplegia affects one side. Upper motor neuron lesion. More severe than paresis. Rehabilitation potential variable.",
        "semantic_type": "Clinical Finding",
        "parent": "13791008"
    },
    "298382003": {
        "preferred_term": "Sinus rhythm",
        "synonyms": ["normal sinus rhythm", "NSR", "sinus rhythm (finding)"],
        "definition": "Normal heart rhythm originating from sinoatrial node. Regular rate 60-100 bpm. Absence suggests arrhythmia. ECG shows P wave before each QRS.",
        "semantic_type": "Clinical Finding",
        "parent": "301114007"
    },
    "271737000": {
        "preferred_term": "Anorexia",
        "synonyms": ["loss of appetite", "anorexia (finding)", "reduced appetite"],
        "definition": "Loss of appetite. Common in acute stroke. Contributes to malnutrition. Affects recovery. Nutritional assessment important.",
        "semantic_type": "Clinical Finding",
        "parent": "79890006"
    },
    "39732003": {
        "preferred_term": "Systolic hypertension",
        "synonyms": ["elevated systolic pressure", "systolic hypertension (disorder)", "high systolic BP"],
        "definition": "Elevated systolic blood pressure >140 mmHg. Independent stroke risk factor. Common in elderly. Widened pulse pressure. Reflects arterial stiffness.",
        "semantic_type": "Clinical Finding",
        "parent": "38341003"
    },
    "422504002": {
        "preferred_term": "Ischemic penumbra",
        "synonyms": ["penumbra", "salvageable brain tissue", "ischemic penumbra (finding)"],
        "definition": "Hypoperfused but viable brain tissue surrounding infarct core. Target for acute treatment. Identified by perfusion-diffusion mismatch on MRI. Time-dependent progression to infarction. Reperfusion can salvage.",
        "semantic_type": "Clinical Finding",
        "parent": "55342001"
    },
    "297217002": {
        "preferred_term": "Rib fracture",
        "synonyms": ["broken rib", "rib fracture (disorder)", "fractured rib"],
        "definition": "Break in rib bone. Can occur from trauma. Causes chest pain. Not typically related to stroke. Included as distractor concept.",
        "semantic_type": "Clinical Finding",
        "parent": "125605004"
    },
    "80394007": {
        "preferred_term": "Hyperglycemia",
        "synonyms": ["elevated blood sugar", "hyperglycemia (finding)", "high glucose"],
        "definition": "Elevated blood glucose level. Common in acute stroke. Stress response or underlying diabetes. Associated with worse outcomes. Target glucose 140-180 mg/dL in stroke.",
        "semantic_type": "Clinical Finding",
        "parent": "73211009"
    },
    "13791008": {
        "preferred_term": "Asthenia",
        "synonyms": ["weakness", "fatigue", "asthenia (finding)", "lack of energy"],
        "definition": "General weakness or lack of energy. Post-stroke fatigue affects 50%. Impacts rehabilitation. Distinguished from focal weakness. Managed with energy conservation, exercise.",
        "semantic_type": "Clinical Finding",
        "parent": "13791008"
    },
    "18165001": {
        "preferred_term": "Jaundice",
        "synonyms": ["icterus", "jaundice (finding)", "yellow skin"],
        "definition": "Yellow discoloration of skin and sclera. Indicates hyperbilirubinemia. Not typically stroke-related. Hepatic or hemolytic cause. Included as distractor concept.",
        "semantic_type": "Clinical Finding",
        "parent": "18165001"
    },
    "84229001": {
        "preferred_term": "Fatigue",
        "synonyms": ["tiredness", "exhaustion", "fatigue (finding)"],
        "definition": "Extreme tiredness. Post-stroke fatigue very common. Multifactorial: brain damage, depression, sleep disorder. Major impact on quality of life. Managed with rest, pacing, medications.",
        "semantic_type": "Clinical Finding",
        "parent": "84229001"
    },
    "424393004": {
        "preferred_term": "Urinary incontinence",
        "synonyms": ["loss of bladder control", "urinary incontinence (finding)", "enuresis"],
        "definition": "Involuntary urine loss. Common after stroke. Indicates frontal lobe damage. Catheterization risk. Improves with rehabilitation. Affects independence.",
        "semantic_type": "Clinical Finding",
        "parent": "165232002"
    },
    "29857009": {
        "preferred_term": "Chest pain",
        "synonyms": ["thoracic pain", "chest pain (finding)", "angina"],
        "definition": "Pain in chest region. May indicate cardiac cause of stroke. Myocardial infarction and stroke can coexist. Requires cardiac evaluation. ECG and troponins.",
        "semantic_type": "Clinical Finding",
        "parent": "22253000"
    },
    "161152002": {
        "preferred_term": "Hyperlipidemia",
        "synonyms": ["high cholesterol", "hyperlipidemia (disorder)", "dyslipidemia"],
        "definition": "Elevated blood lipids. Stroke risk factor. Causes atherosclerosis. Target LDL <100 mg/dL for secondary prevention. Statin therapy recommended.",
        "semantic_type": "Clinical Finding",
        "parent": "55822004"
    },
    "248567008": {
        "preferred_term": "Tachycardia",
        "synonyms": ["rapid heart rate", "tachycardia (finding)", "fast heart beat"],
        "definition": "Heart rate >100 bpm. Can indicate atrial fibrillation. Cardiac evaluation needed. May be stress response to stroke. ECG for rhythm assessment.",
        "semantic_type": "Clinical Finding",
        "parent": "3424008"
    },
    "62315008": {
        "preferred_term": "Diarrhea",
        "synonyms": ["loose stools", "diarrhea (finding)", "frequent bowel movements"],
        "definition": "Frequent loose watery stools. Not typically stroke symptom. Possible medication side effect. Dehydration risk. Included as distractor concept.",
        "semantic_type": "Clinical Finding",
        "parent": "62315008"
    }
}


def load_existing_concepts_from_training():
    """Load the 6 concepts that ARE in both training data and existing ontology"""
    train_path = os.path.join('benchmark', 'data', 'train_annotations.csv')
    ontology_path = 'conceptos_con_narrativas.csv'
    
    # Load training concepts
    train_df = pd.read_csv(train_path)
    train_concepts = set(train_df['concept_id'].astype(str).unique())
    
    # Load existing ontology
    ontology_df = pd.read_csv(ontology_path)
    ontology_concepts = set(ontology_df['concepto'].astype(str).unique())
    
    # Find intersection (concepts that exist in both)
    existing_used = train_concepts & ontology_concepts
    
    print(f"Found {len(existing_used)} concepts that are in both training and ontology:")
    for concept in sorted(existing_used):
        print(f"  - {concept}")
    
    # Extract these rows from the existing ontology
    existing_df = ontology_df[ontology_df['concepto'].isin(existing_used)]
    
    return existing_df


def create_narrative(concept_id, concept_data):
    """Create a rich narrative similar to conceptos_con_narrativas.csv format"""
    narrative_parts = []
    
    # Add concept ID references
    narrative_parts.append(f"{concept_id} tiene código {concept_id}")
    
    # Add preferred term
    preferred = concept_data['preferred_term']
    narrative_parts.append(f"{concept_id} tiene término preferido {preferred}")
    narrative_parts.append(f"{concept_id} tiene término preferido {preferred} (en inglés)")
    
    # Add synonyms
    for synonym in concept_data['synonyms']:
        narrative_parts.append(f"{concept_id} tiene sinónimo {synonym}")
    
    # Add semantic type
    narrative_parts.append(f"{concept_id} es de tipo {concept_data['semantic_type']}")
    narrative_parts.append(f"{concept_id} es de tipo Class")
    
    # Add ontology membership
    narrative_parts.append(f"{concept_id} pertenece a la terminología snomed")
    
    # Add parent relationship
    if 'parent' in concept_data:
        narrative_parts.append(f"{concept_id} es una subclase de {concept_data['parent']}")
    
    # Add definition (the most important part for semantic similarity!)
    definition = concept_data['definition']
    narrative_parts.append(f"{concept_id} se define como: {definition}")
    
    # Create the full narrative
    narrative = " ".join(narrative_parts)
    
    return narrative


def add_similar_concepts_from_ontology(target_count=2500):
    """
    Add similar concepts from conceptos_con_narrativas.csv to reach target count.
    Prioritizes medical/clinical concepts to maintain domain relevance.
    """
    ontology_path = 'conceptos_con_narrativas.csv'
    
    print(f"\n4. Adding similar concepts from full ontology to reach {target_count} concepts...")
    
    # Load full ontology
    full_df = pd.read_csv(ontology_path)
    print(f"   Full ontology contains: {len(full_df)} concepts")
    
    # Get already included concept IDs
    train_path = os.path.join('benchmark', 'data', 'train_annotations.csv')
    train_df = pd.read_csv(train_path)
    train_concepts = set(train_df['concept_id'].astype(str).unique())
    
    # Concepts already in hybrid
    already_included = train_concepts | set(MISSING_CONCEPTS.keys()) | set(NOISE_CONCEPTS.keys())
    
    # Filter to get available concepts
    available_df = full_df[~full_df['concepto'].astype(str).isin(already_included)]
    
    print(f"   Available concepts to sample from: {len(available_df)}")
    
    # Calculate how many more we need
    current_count = len(already_included)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"   Already have {current_count} concepts, no need to add more")
        return available_df.head(0)  # Return empty
    
    print(f"   Need to add: {needed} concepts")
    
    # Sample randomly but ensure good distribution
    if len(available_df) >= needed:
        sampled_df = available_df.sample(n=needed, random_state=42)
    else:
        print(f"   WARNING: Only {len(available_df)} concepts available, using all")
        sampled_df = available_df
    
    print(f"   Sampled {len(sampled_df)} additional concepts")
    
    return sampled_df


def build_hybrid_ontology(target_count=2500):
    """Build the hybrid ontology with target number of concepts"""
    print("="*80)
    print("BUILDING EXPANDED HYBRID ONTOLOGY")
    print("="*80)
    print(f"Target size: {target_count} concepts")
    
    # 1. Load existing concepts from training that are in conceptos_con_narrativas.csv
    print("\n1. Loading existing concepts that are used in training...")
    existing_df = load_existing_concepts_from_training()
    print(f"   Loaded {len(existing_df)} existing concepts (100% coverage required)")
    
    # 2. Create narratives for missing concepts
    print("\n2. Creating narratives for 26 missing concepts...")
    missing_data = []
    for concept_id, concept_info in MISSING_CONCEPTS.items():
        narrative = create_narrative(concept_id, concept_info)
        missing_data.append({
            'concepto': concept_id,
            'narrativa': narrative
        })
    missing_df = pd.DataFrame(missing_data)
    print(f"   Created {len(missing_df)} missing concept narratives (CRITICAL)")
    
    # 3. Create narratives for noise concepts
    print("\n3. Creating narratives for ~30 noise concepts...")
    noise_data = []
    for concept_id, concept_info in NOISE_CONCEPTS.items():
        narrative = create_narrative(concept_id, concept_info)
        noise_data.append({
            'concepto': concept_id,
            'narrativa': narrative
        })
    noise_df = pd.DataFrame(noise_data)
    print(f"   Created {len(noise_df)} noise concept narratives (distractors)")
    
    # 4. Add more concepts from full ontology
    additional_df = add_similar_concepts_from_ontology(target_count)
    
    # 5. Combine all concepts
    print(f"\n5. Combining all concepts...")
    hybrid_df = pd.concat([existing_df, missing_df, noise_df, additional_df], ignore_index=True)
    
    # Remove duplicates just in case
    hybrid_df = hybrid_df.drop_duplicates(subset=['concepto'])
    
    # Verify 100% coverage
    train_path = os.path.join('benchmark', 'data', 'train_annotations.csv')
    train_df = pd.read_csv(train_path)
    train_concepts = set(train_df['concept_id'].astype(str).unique())
    hybrid_concepts = set(hybrid_df['concepto'].astype(str).unique())
    coverage = len(train_concepts & hybrid_concepts) / len(train_concepts) * 100
    
    print(f"\n{'='*80}")
    print("HYBRID ONTOLOGY SUMMARY")
    print("="*80)
    print(f"Existing concepts (used in training):  {len(existing_df):>5}")
    print(f"Missing concepts (now added):          {len(missing_df):>5}")
    print(f"Noise concepts (distractors):          {len(noise_df):>5}")
    print(f"Additional concepts (from full ont.):  {len(additional_df):>5}")
    print(f"{'-'*80}")
    print(f"TOTAL concepts in hybrid ontology:     {len(hybrid_df):>5}")
    print(f"\n{'='*80}")
    print(f"COVERAGE VERIFICATION")
    print(f"{'='*80}")
    print(f"Training concepts required:            {len(train_concepts):>5}")
    print(f"Training concepts in hybrid:           {len(train_concepts & hybrid_concepts):>5}")
    print(f"Coverage percentage:                   {coverage:>5.1f}%")
    
    if coverage < 100:
        print(f"\n❌ WARNING: Coverage is not 100%! Missing concepts:")
        missing_in_hybrid = train_concepts - hybrid_concepts
        for concept in sorted(missing_in_hybrid):
            print(f"   - {concept}")
    else:
        print(f"\n✅ COVERAGE VERIFIED: All training concepts are included!")
    
    # 6. Save to file
    output_path = 'hybrid_ontology.csv'
    hybrid_df.to_csv(output_path, index=False)
    print(f"\n✅ Hybrid ontology saved to: {output_path}")
    
    # 7. Display sample narratives
    print(f"\n{'='*80}")
    print("SAMPLE NARRATIVES (showing 3 from each category)")
    print("="*80)
    
    print("\n[EXISTING CONCEPTS FROM TRAINING]")
    for idx, row in existing_df.head(3).iterrows():
        print(f"  {row['concepto']}: {row['narrativa'][:100]}...")
    
    print("\n[MISSING CONCEPTS - NOW ADDED]")
    for idx, row in missing_df.head(3).iterrows():
        print(f"  {row['concepto']}: {row['narrativa'][:100]}...")
    
    print("\n[NOISE CONCEPTS - DISTRACTORS]")
    for idx, row in noise_df.head(3).iterrows():
        print(f"  {row['concepto']}: {row['narrativa'][:100]}...")
    
    if len(additional_df) > 0:
        print("\n[ADDITIONAL CONCEPTS - FROM FULL ONTOLOGY]")
        for idx, row in additional_df.head(3).iterrows():
            print(f"  {row['concepto']}: {row['narrativa'][:100]}...")
    
    return hybrid_df


if __name__ == "__main__":
    # Build hybrid ontology with 2500 concepts
    hybrid_df = build_hybrid_ontology(target_count=2500)
