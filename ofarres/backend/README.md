# 🏥 Medical NER Pipeline for RAG Systems

> **High-Recall Named Entity Recognition Pipeline for Medical Text**
> 
> Un sistema de extracción de entidades médicas optimizado para sistemas RAG (Retrieval-Augmented Generation), diseñado para maximizar el Recall sin sacrificar precisión.

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#-resumen-ejecutivo)
2. [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
3. [Estructura del Proyecto](#-estructura-del-proyecto)
4. [Workers NER](#-workers-ner)
5. [Pipeline de Postprocesamiento](#-pipeline-de-postprocesamiento)
6. [Sistema de Métricas RAG-Friendly](#-sistema-de-métricas-rag-friendly)
7. [Configuración](#-configuración)
8. [Uso](#-uso)
9. [Benchmarks](#-benchmarks)
10. [Datos](#-datos)

---

## 🎯 Resumen Ejecutivo

### Problema
Los sistemas RAG médicos requieren **alta cobertura (Recall)** de entidades médicas para poder recuperar contexto relevante. Un sistema NER tradicional optimizado para Precision puede perder conceptos críticos.

### Solución
Pipeline de 5 etapas que:
1. **Maximiza Recall** combinando múltiples workers NER
2. **Asigna confianza** mediante sistema de tiers (consenso)
3. **Elimina ruido** progresivamente sin perder True Positives

### Resultados Actuales

```
====================================================================================================
 PIPELINE PERFORMANCE DASHBOARD (RAG-Friendly Metrics)
====================================================================================================
Step Name                           | Entities   | Recall     | Precision  | F1
----------------------------------------------------------------------------------------------------
01_gather_assembly                  | 394        | 100.00%    | 24.62%     | 0.3951
02_assign_ranks                     | 394        | 100.00%    | 24.62%     | 0.3951
03_safe_deduplication               | 389        | 100.00%    | 24.94%     | 0.3992
04_linguistic_filter                | 361        | 100.00%    | 26.87%     | 0.4236
05_semantic_judge                   | 360        | 100.00%    | 26.94%     | 0.4245
====================================================================================================
```

**🎯 100% Recall mantenido a través de todo el pipeline**

---

## 🏗 Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           NER PIPELINE ORCHESTRATOR                              │
│                    (A_pipeline_orchestrator.py)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 01: THE HARVESTER (01_gather_assembly.py)                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Responsabilidad: Ejecutar todos los workers NER y fusionar consenso            │
│                                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                         │
│  │ OntologyNER  │   │   ScispaCy   │   │  AcronymNER  │                         │
│  │   (Exact)    │   │   (SBert)    │   │              │                         │
│  └──────────────┘   └──────────────┘   └──────────────┘                         │
│         │                  │                  │                                  │
│         └──────────────────┴──────────────────┘                                  │
│                            │                                                     │
│                    Merge Duplicates                                              │
│         (same start/end → source: ["OntologyExact", "SBert"])                   │
│                                                                                  │
│  Output: data/ner/01_raw_assembly.json                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 02: THE CLASSIFIER (02_assign_ranks.py)                                    │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Responsabilidad: Asignar Tiers de Confianza basados en consenso de fuentes    │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TIER 1 (Elite/Platinum) - priority: 1                                     │ │
│  │    • Acronyms (cualquier entidad detectada por AcronymNER)                 │ │
│  │    • Consensus (OntologyExact + SBert coinciden en el mismo span)          │ │
│  ├────────────────────────────────────────────────────────────────────────────┤ │
│  │  TIER 2 (Gold/Standard) - priority: 2                                      │ │
│  │    • Solo OntologyExact (respaldado por diccionario médico)                │ │
│  ├────────────────────────────────────────────────────────────────────────────┤ │
│  │  TIER 3 (Bronze/Weak) - priority: 3                                        │ │
│  │    • Solo SBert (predicción del modelo, menor confianza)                   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Output: data/ner/02_ranked.json                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 03: SAFE DEDUPLICATION (03_safe_deduplication.py)                          │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Responsabilidad: Resolver solapamientos ("Russian Doll") sin perder Recall     │
│                                                                                  │
│  Estrategia: Dictionary Sovereign + Coexistence                                  │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  CONFLICT RESOLUTION MATRIX                                                │ │
│  │  ──────────────────────────────────────────────────────────────────────── │ │
│  │  Container Tier 1/2 + Nested Tier ≥ Container:                             │ │
│  │    → ABSORB: Mantener Container, Eliminar Nested                           │ │
│  │    Ejemplo: "Middle Cerebral Artery" absorbe "Artery"                      │ │
│  │                                                                            │ │
│  │  Container Tier 1/2 + Nested Tier < Container:                             │ │
│  │    → COEXIST (Rank Protection): Mantener ambos                             │ │
│  │    Ejemplo: T2 "Alberta...Score" NO absorbe T1 "Stroke"                    │ │
│  │                                                                            │ │
│  │  Container Tier 3:                                                         │ │
│  │    → COEXIST: Mantener ambos (SBert no es confiable para decidir)          │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Output: data/ner/03_deduplicated.json                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 04: LINGUISTIC FILTER (04_linguistic_filter.py)                            │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Responsabilidad: Eliminar basura sintáctica de Tier 3 antes del Cross-Encoder │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TIER 1 & 2: AUTO-PASS (No tocar - respaldados por diccionario)            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TIER 3: APLICAR REGLAS DE EXCLUSIÓN                                       │ │
│  │                                                                            │ │
│  │  Rule 1: Is Header                                                         │ │
│  │    • Texto ALL CAPS y longitud > 3                                         │ │
│  │    • Ejemplos: "IMAGING", "HOSPITAL COURSE", "EXAMINATION"                 │ │
│  │    • Nota: Acrónimos válidos (CT, MRI) son Tier 1, no se tocan             │ │
│  │                                                                            │ │
│  │  Rule 2: Is Ghost                                                          │ │
│  │    • Span contiene SOLO stopwords, puntuación o números                    │ │
│  │    • Ejemplos: "and", "the", ".", "123"                                    │ │
│  │                                                                            │ │
│  │  Rule 3: Is Lonely Modifier                                                │ │
│  │    • Token único con POS tag: ADJ, ADV, DET, PRON, CCONJ                   │ │
│  │    • Ejemplos a eliminar: "Severe", "Left", "The", "And"                   │ │
│  │    • Ejemplos a mantener: "Severe Stroke" (multi-token), "Vomiting" (Noun) │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Dependencias: spacy (en_core_web_sm para velocidad)                            │
│  Output: data/ner/04_linguistically_clean.json                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 05: SEMANTIC JUDGE (05_semantic_judge.py)                                  │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  Responsabilidad: Filtrar "Hard Noise" semánticamente irrelevante               │
│                                                                                  │
│  Modelo: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M params, muy rápido)         │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TIER 1 & 2: AUTO-PASS (No gastar compute en entidades de diccionario)     │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  TIER 3: FILTRADO HÍBRIDO                                                  │ │
│  │                                                                            │ │
│  │  1. BLACKLIST: Términos genéricos comunes en notas clínicas                │ │
│  │     • Demográficos: "male", "female", "patient", "year old"                │ │
│  │     • Temporales: "history", "admission", "discharge", "onset"             │ │
│  │     • Contextuales: "examination", "findings", "status"                    │ │
│  │     • Descriptores: "normal", "abnormal", "mild", "severe"                 │ │
│  │                                                                            │ │
│  │  2. CROSS-ENCODER: Scoring contrastivo para edge cases                     │ │
│  │     • Anchor Médico: "medical condition, symptom, finding, procedure"      │ │
│  │     • Anchor General: "general English word with no medical meaning"       │ │
│  │     • Score = score_medical - score_general                                │ │
│  │     • Threshold: -1.0 (muy conservador para no perder Recall)              │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  Output: data/ner/05_semantically_clean.json                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Proyecto

```
ofarres/
├── .env                          # Variables de entorno (API keys)
├── README.md                     # Este archivo
│
├── config/
│   └── ner_registry.json         # Configuración de workers NER
│
├── data/
│   ├── notes.json                # Notas clínicas de entrada
│   ├── ground_truth.json         # Ground truth con campo "text"
│   ├── ground_truth_no_concept.json
│   ├── generate_ground_truth.py  # Script para generar GT con texto
│   └── ner/                      # Outputs intermedios del pipeline
│       ├── 01_raw_assembly.json
│       ├── 02_ranked.json
│       ├── 03_deduplicated.json
│       ├── 04_linguistically_clean.json
│       └── 05_semantically_clean.json
│
├── ontology/
│   ├── multilingual_ontology.json  # Ontología principal (SNOMED-CT)
│   ├── ontology.json
│   └── hybrid_ontology.csv
│
├── assets/
│   └── ontology/
│       └── ontology_rag.index    # Índice FAISS para RAG
│
├── src/
│   ├── main.py
│   │
│   ├── NER/
│   │   ├── A_pipeline_orchestrator.py  # 🎯 ORQUESTADOR PRINCIPAL
│   │   ├── ontology_ner.py             # Worker: Matching exacto
│   │   ├── spacy_ner.py                # Worker: ScispaCy/SBert
│   │   ├── acronym_ner.py              # Worker: Acrónimos médicos
│   │   └── postprocessor/
│   │       ├── 01_gather_assembly.py   # Step 1: Harvester
│   │       ├── 02_assign_ranks.py      # Step 2: Classifier
│   │       ├── 03_safe_deduplication.py # Step 3: Deduplication
│   │       ├── 04_linguistic_filter.py  # Step 4: Linguistic Filter
│   │       └── 05_semantic_judge.py     # Step 5: Semantic Judge
│   │
│   ├── RAG/
│   │   ├── 01_Retrieval/
│   │   ├── 02_Augmentation/
│   │   └── 03_Generation/
│   │
│   └── utils/
│       ├── metrics.py            # Cálculo de IoU, P/R/F1
│       ├── build_faiss_index.py
│       ├── clean_ground_truth.py
│       ├── enrich_ontology_gemini.py
│       └── ontology_cv2json.py
│
└── benchmarks/
    ├── diagnose_NER.py           # Benchmark individual de workers
    ├── assembly_diagnoseNER.py   # Benchmark del assembly
    └── cross_validation.py       # Validación cruzada
```

---

## 🤖 Workers NER

### 1. OntologyNER (`ontology_ner.py`)

**Estrategia:** Matching exacto basado en diccionario médico (SNOMED-CT)

```python
class OntologyNER:
    """
    Worker NER Genérico y Robusto.
    
    ESTRATEGIA "NO-CHEAT":
    1. Sin reglas morfológicas ad-hoc
    2. Sin listas negras arbitrarias
    3. Usa puramente:
       - Variaciones exactas de la ontología
       - Pluralización estándar del inglés
       - Extracción de núcleo (Head Word)
    """
```

**Características:**
- Usa FlashText para búsqueda O(n) eficiente
- Genera variaciones automáticas (plurales, head words)
- Case-insensitive
- Soporta boundaries especiales (-, /)

**Confiabilidad:** ⭐⭐⭐⭐⭐ (Alta - respaldado por diccionario)

---

### 2. ScispaCyNER (`spacy_ner.py`)

**Estrategia:** Modelo transformer pre-entrenado en texto biomédico

```python
class ScispaCyNER:
    """
    Worker NER que usa un modelo scispaCy pre-entrenado.
    Modelo: en_core_sci_scibert
    """
```

**Características:**
- Modelo: `en_core_sci_scibert` (SciBERT fine-tuned)
- Detecta entidades que no están en el diccionario
- Mayor cobertura, menor precisión

**Confiabilidad:** ⭐⭐⭐ (Media - predicción de modelo)

---

### 3. AcronymNER (`acronym_ner.py`)

**Estrategia:** Especialista en acrónimos médicos (CT, MRI, NIHSS, etc.)

```python
class AcronymNER:
    """
    Worker Especialista en Acrónimos (Stopword-Aware + Boundary Fix).
    
    CORRECCIÓN:
    1. Límites de palabra (-, /, .) para detectar "CT-scan" o "C.T."
    2. Stopwords para permitir siglas cortas bloqueando basura
    """
```

**Características:**
- Case-sensitive (importante para acrónimos)
- Filtro de longitud (2-6 caracteres)
- Blacklist de stopwords (evita "AT", "IN", etc.)
- Soporta variaciones: "CT", "CT-scan", "C.T."

**Confiabilidad:** ⭐⭐⭐⭐⭐ (Alta - acrónimos son inequívocos)

---

## 🔧 Pipeline de Postprocesamiento

### Step 01: The Harvester

**Archivo:** `postprocessor/01_gather_assembly.py`

**Responsabilidad:** Ejecutar todos los workers y fusionar entidades duplicadas

**Lógica de Fusión:**
```python
# Si múltiples workers encuentran el mismo span (start, end):
# Antes: [{"text": "stroke", "source": "OntologyExact"}, 
#         {"text": "stroke", "source": "SBert"}]
# Después: [{"text": "stroke", "source": ["OntologyExact", "SBert"]}]
```

**Output:** `data/ner/01_raw_assembly.json`

---

### Step 02: The Classifier

**Archivo:** `postprocessor/02_assign_ranks.py`

**Responsabilidad:** Asignar tiers de confianza

**Sistema de Tiers:**

| Tier | Nombre | Condición | Ejemplo |
|------|--------|-----------|---------|
| 1 | Elite | Acronyms OR (OntologyExact + SBert) | "CT", "stroke" (consenso) |
| 2 | Gold | Solo OntologyExact | "hemorrhage" |
| 3 | Bronze | Solo SBert | "patient", "history" |

**Output:** `data/ner/02_ranked.json` (añade campo `priority`)

---

### Step 03: Safe Deduplication

**Archivo:** `postprocessor/03_safe_deduplication.py`

**Responsabilidad:** Resolver solapamientos sin perder Recall

**Problema "Russian Doll":**
```
"Alberta Stroke Program Early CT Score"  (Tier 2)
         └── "Stroke"                     (Tier 1 - Acronym)
                     └── "CT"             (Tier 1 - Acronym)
```

**Matriz de Resolución:**

| Container | Nested | Acción | Razón |
|-----------|--------|--------|-------|
| T1/T2 | ≥ Container | ABSORB | Diccionario es confiable |
| T1/T2 | < Container | COEXIST | Rank Protection |
| T3 | Cualquier | COEXIST | SBert no es confiable |

**Output:** `data/ner/03_deduplicated.json`

---

### Step 04: Linguistic Filter

**Archivo:** `postprocessor/04_linguistic_filter.py`

**Responsabilidad:** Eliminar basura sintáctica de Tier 3

**Dependencias:** `spacy` (modelo `en_core_web_sm`)

**Reglas de Exclusión:**

1. **Is Header:** ALL CAPS + longitud > 3
   - ✗ "HOSPITAL COURSE", "IMAGING", "EXAMINATION"
   - ✓ "CT", "MRI" (son Tier 1, no se tocan)

2. **Is Ghost:** Solo stopwords/puntuación/números
   - ✗ "and", "the", "No", "123"

3. **Is Lonely Modifier:** Token único + POS ∈ {ADJ, ADV, DET, PRON, CCONJ}
   - ✗ "Severe", "Left", "The"
   - ✓ "Severe Stroke" (multi-token)

**Output:** `data/ner/04_linguistically_clean.json`

---

### Step 05: Semantic Judge

**Archivo:** `postprocessor/05_semantic_judge.py`

**Responsabilidad:** Filtrar "Hard Noise" semánticamente irrelevante

**Modelo:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params)

**Estrategia Híbrida:**

1. **Blacklist:** Términos genéricos comunes
   ```python
   BLACKLIST = {
       "male", "female", "patient", "history", "admission",
       "examination", "findings", "normal", "abnormal", ...
   }
   ```

2. **Cross-Encoder Contrastivo:**
   ```python
   score = model.predict(MEDICAL_ANCHOR, text) - model.predict(GENERAL_ANCHOR, text)
   # score > THRESHOLD → Mantener
   ```

**Output:** `data/ner/05_semantically_clean.json`

---

## 📊 Sistema de Métricas RAG-Friendly

### Problema con Métricas Tradicionales

Las métricas tradicionales (IoU > 0.5 estricto) penalizan la **expansión de contexto**, que es deseable en RAG:

```
GT:   "hemorrhage"           [start=100, end=110]
Pred: "acute hemorrhage"     [start=94, end=110]

IoU = 10/16 = 0.625  (podría no pasar umbral alto)
Pero semánticamente: ✓ CORRECTO (context expansion)
```

### Criterio RAG-Friendly

**True Positive si:**

1. **Condition A:** IoU > 0.1 (overlap físico mínimo)
2. **Condition B:** Text Containment (bidireccional)
   - GT_text ⊆ Pred_text (Context Expansion)
   - OR Pred_text ⊆ GT_text (Partial Match)

3. **Constraint:** 1-to-1 Matching
   - Una predicción solo cuenta para UN GT
   - Previene "Bad Merges" que inflan Recall

```python
def text_containment_match(pred_text: str, gt_text: str) -> bool:
    pred_norm = pred_text.lower().strip()
    gt_norm = gt_text.lower().strip()
    return gt_norm in pred_norm or pred_norm in gt_norm
```

### Implementación

```python
# src/utils/metrics.py
def calculate_iou(pred_span: Dict, gt_span: Dict) -> float:
    """Intersection over Union para spans."""
    ...

# benchmarks/diagnose_NER.py
def get_detailed_matches(preds, gt, iou_thresh, note_text):
    """Matching con Text Containment + IoU."""
    ...
```

---

## ⚙️ Configuración

### ner_registry.json

Define los workers NER disponibles:

```json
{
    "OntologyExact": {
        "module": "src.NER.ontology_ner",
        "class": "OntologyNER",
        "ontology_path": "ontology/multilingual_ontology.json",
        "min_length": 1
    },
    "SBert": {
        "module": "src.NER.spacy_ner",
        "class": "ScispaCyNER",
        "model_name": "en_core_sci_scibert"
    },
    "Acronyms": {
        "module": "src.NER.acronym_ner",
        "class": "AcronymNER",
        "ontology_path": "ontology/multilingual_ontology.json",
        "max_acronym_len": 8
    }
}
```

---

## 🚀 Uso

### Ejecutar Pipeline Completo

```bash
# Desde el directorio ofarres/
python src/NER/A_pipeline_orchestrator.py
```

**Output:**
```
====================================================================================================
 NER PIPELINE ORCHESTRATOR
 RAG-Friendly High-Recall Entity Extraction
====================================================================================================

[Orchestrator] Loading ground truth and notes...
[Orchestrator] Ground Truth: 97 entities across 5 notes

================================================================================
 STEP 01: THE HARVESTER (Gather Assembly)
================================================================================
...

================================================================================
 PIPELINE PERFORMANCE DASHBOARD (RAG-Friendly Metrics)
================================================================================
Step Name                           | Entities   | Recall     | Precision  | F1
----------------------------------------------------------------------------------------------------
01_gather_assembly                  | 394        | 100.00%    | 24.62%     | 0.3951
02_assign_ranks                     | 394        | 100.00%    | 24.62%     | 0.3951
03_safe_deduplication               | 389        | 100.00%    | 24.94%     | 0.3992
04_linguistic_filter                | 361        | 100.00%    | 26.87%     | 0.4236
05_semantic_judge                   | 360        | 100.00%    | 26.94%     | 0.4245
====================================================================================================
```

### Ejecutar Steps Individuales

```bash
# Step 01: Harvester
python src/NER/postprocessor/01_gather_assembly.py

# Step 02: Classifier
python src/NER/postprocessor/02_assign_ranks.py

# Step 03: Deduplication
python src/NER/postprocessor/03_safe_deduplication.py

# Step 04: Linguistic Filter
python src/NER/postprocessor/04_linguistic_filter.py

# Step 05: Semantic Judge
python src/NER/postprocessor/05_semantic_judge.py
```

### Benchmark Individual de Workers

```bash
# Evaluar un worker específico
python benchmarks/diagnose_NER.py OntologyExact --iou 0.25 -v

# Evaluar todos los workers
python benchmarks/diagnose_NER.py all

# Evaluar el assembly (todos combinados)
python benchmarks/diagnose_NER.py assembly -v
```

---

## 📈 Benchmarks

### diagnose_NER.py

Benchmark principal para evaluar workers individuales y el assembly.

**Modos:**
- `all`: Evalúa cada worker por separado
- `assembly`: Evalúa la combinación de todos
- `<worker_id>`: Evalúa un worker específico

**Flags:**
- `--iou`: Umbral IoU (default: 0.25)
- `-v, --verbose`: Tabla detallada de matches

### assembly_diagnoseNER.py

Benchmark específico para el assembly con análisis de contribución no-redundante.

### cross_validation.py

Validación cruzada para evaluar generalización.

---

## 💾 Datos

### notes.json

Notas clínicas de entrada:

```json
[
    {
        "note_id": "1",
        "text": "A 72-year-old male with a history of hypertension..."
    },
    ...
]
```

### ground_truth.json

Ground truth con anotaciones:

```json
[
    {
        "note_id": "1",
        "annotations": [
            {
                "start": 75,
                "end": 87,
                "concept_id": "38341003",
                "text": "hypertension"
            },
            ...
        ]
    },
    ...
]
```

**Nota:** El campo `text` es generado por `generate_ground_truth.py` para habilitar Text Containment matching.

### Outputs del Pipeline

Cada step genera un JSON con la misma estructura:

```json
[
    {
        "note_id": "1",
        "annotations": [
            {
                "start": 75,
                "end": 87,
                "text": "hypertension",
                "source": ["OntologyExact", "SBert"],
                "priority": 1,
                "semantic_score": 0.8234  // Solo en Step 05
            },
            ...
        ]
    },
    ...
]
```

---

## 📦 Dependencias

```txt
# Core NLP
spacy>=3.7.0
scispacy>=0.5.1
en_core_web_sm
en_core_sci_scibert

# Fast String Matching
flashtext>=2.7

# Transformers
sentence-transformers>=2.2.0

# Data
numpy>=1.20.0
```

### Instalación de Modelos spaCy

```bash
# Modelo ligero para filtro lingüístico
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Modelo SciBERT para NER biomédico
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz
```

---

## 🔮 Trabajo Futuro

1. **Step 06: Cross-Encoder Ranker**
   - Re-ranking de candidatos usando SNOMED-CT embeddings

2. **Step 07: LLM Validator**
   - Validación final con GPT-4 para edge cases

3. **Ontology Linking**
   - Mapeo de entidades a concept_id de SNOMED-CT

4. **Multilingual Support**
   - Extensión a español y catalán usando la ontología multilingüe

---

## 👤 Autor

**Oscar Farrés** - NLP Engineer

---

## 📄 Licencia

Proyecto interno - Todos los derechos reservados.
