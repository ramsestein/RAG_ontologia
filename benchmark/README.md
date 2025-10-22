# 📊 Sistema de Evaluación de Estrategias SNOMED-CT

Sistema de benchmark para evaluar y comparar diferentes estrategias de extracción y vinculación de entidades clínicas usando terminología SNOMED-CT en informes de ictus.

## 📁 Estructura del Directorio

```
benchmark/
├── main.py                          # Script principal de ejecución
├── README.md                        # Este archivo
│
├── data/                            # Conjunto de datos de entrenamiento
│   ├── mimic-iv_notes_training_set.csv   # Notas clínicas (informes CT)
│   └── train_annotations.csv             # Anotaciones ground truth
│
├── strategies/                      # Implementaciones de estrategias
│   ├── 01_kiris.py                 # Estrategia KIRIs (1er lugar)
│   ├── 02_snobert.py               # Estrategia SNOBERT (2do lugar)
│   ├── 03_ollama.py                # Estrategia Ollama (desactivada)
│   └── 04_rag_gpt.py               # Estrategia RAG + GPT-4o
│
├── evaluation/                      # Módulos de evaluación
│   ├── metrics_calculator.py       # Calculador de métricas
│   └── metrics.py                  # Funciones auxiliares de métricas
│
└── results/                         # Resultados de ejecuciones
    ├── 01_execution_DD_MM_YYYY_HH_MM/
    ├── 02_execution_DD_MM_YYYY_HH_MM/
    └── ...
```

## 🎯 ¿Qué Hace Este Sistema?

El sistema de benchmark evalúa diferentes estrategias de **Named Entity Recognition (NER)** y **Entity Linking** para extraer entidades clínicas de informes médicos de tomografía computarizada (CT) de pacientes con sospecha de ictus agudo.

### Entidades Extraídas

Las estrategias buscan identificar y codificar:

- **Hallazgos clínicos**: hemorragia, lesión isquémica, oclusiones arteriales
- **Escalas clínicas**: ASPECTS, NIHSS
- **Ubicaciones anatómicas**: caudado, lenticular, arterias cerebrales, territorios vasculares
- **Valores cuantitativos**: grados de estenosis, scores ASPECTS

Cada entidad se codifica usando:
- **SNOMED-CT** para presencia/ausencia
- **Ontología OWL personalizada** para anatomía y hallazgos

## 🚀 Uso del Sistema

### Instalación de Dependencias

```bash
# Asegúrate de estar en un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r ../requirements.txt
```

### Ejecución Básica

```bash
# Ejecutar todas las estrategias activas (solo muestra resultados)
python main.py

# Ejecutar estrategia específica por ID
python main.py 1         # Ejecuta KIRIs
python main.py 2         # Ejecuta SNOBERT
python main.py 4         # Ejecuta RAG + GPT-4o

# Ejecutar múltiples estrategias
python main.py 1 2 4     # Ejecuta y compara las 3 estrategias
```

### Guardar Resultados

**Por defecto, los resultados NO se guardan en disco**, solo se muestran en consola. Para guardar:

```bash
# Guardar resultados de una estrategia
python main.py 1 -r

# Guardar resultados de todas las estrategias
python main.py -r

# Usando el flag largo
python main.py 1 4 --save-results
```

### Opciones Avanzadas

```bash
# Usar flag -s en lugar de argumentos posicionales
python main.py -s 1 2 -r

# Ver ayuda completa
python main.py --help
```

## 📈 Estrategias Disponibles

### Estrategia 1: KIRIs (1er Lugar - Competición 2024)
**Archivo**: `strategies/01_kiris.py`

- **Enfoque**: Sistema híbrido de diccionario + reglas lingüísticas
- **Ventajas**: Alta precisión, rápido
- **Tecnologías**: spaCy, diccionarios especializados
- **Estado**: ✅ Activa

### Estrategia 2: SNOBERT (2do Lugar)
**Archivo**: `strategies/02_snobert.py`

- **Enfoque**: BERT + SapBERT + embeddings semánticos
- **Ventajas**: Buen recall, maneja variaciones léxicas
- **Tecnologías**: Transformers, sentence embeddings
- **Estado**: ✅ Activa

### Estrategia 3: MITEL Ollama (3er Lugar)
**Archivo**: `strategies/03_ollama.py`

- **Enfoque**: Mistral 7B via Ollama + RAG
- **Ventajas**: Flexible, generativo
- **Tecnologías**: Ollama, Mistral 7B, RAG
- **Estado**: ⚠️ Desactivada (requiere Ollama local)

### Estrategia 4: RAG + GPT-4o (Personalizada)
**Archivo**: `strategies/04_rag_gpt.py`

- **Enfoque**: Sistema RAG con búsqueda semántica + GPT-4o
- **Pipeline**:
  1. **NER con GPT-4o**: Detección inicial de entidades
  2. **RAG con FAISS**: Búsqueda de conceptos similares en ontología
  3. **Codificación con GPT-4o**: Asignación de códigos SNOMED-CT
- **Tecnologías**: 
  - OpenAI GPT-4o
  - FAISS (búsqueda vectorial)
  - SentenceTransformers (embeddings)
  - Ontología personalizada (`conceptos_con_narrativas.csv`)
- **Estado**: ✅ Activa (requiere API key de OpenAI)

#### ⚡ Optimización de Rendimiento (IMPORTANTE)

Esta estrategia usa un **índice Faiss pre-construido** para búsqueda rápida. **Antes de usarla por primera vez**, debes construir el índice:

```bash
# Construcción del índice (una sola vez, tarda ~10 minutos)
cd benchmark
python build_rag_index.py
```

Esto generará archivos en `benchmark/assets/`:
- `ontology.index` - Índice Faiss con 45,440 conceptos
- `ontology_concepts.pkl` - Lista de códigos SNOMED-CT
- `ontology_narrativas.pkl` - Narrativas para generar contexto
- `ontology_metadata.pkl` - Metadatos del índice

**Ventajas de esta arquitectura:**
- 🚀 Inicialización instantánea (~2 segundos vs ~20 minutos)
- 🔄 Consistencia: mismo índice en todas las ejecuciones
- 🧩 Modularidad: separación de responsabilidades (SRP)
- 📦 Extensibilidad: cambios en el índice no afectan la estrategia (OCP)

Ver `RAG_OPTIMIZATION.md` para detalles técnicos completos.

## 📊 Métricas de Evaluación

El sistema calcula las siguientes métricas para cada estrategia:

### Métricas Principales

- **Precision**: `matches / predicciones`
  - ¿Qué porcentaje de las entidades predichas son correctas?
  
- **Recall**: `matches / ground_truth`
  - ¿Qué porcentaje de las entidades reales fueron detectadas?
  
- **F1-Score**: `2 * (precision * recall) / (precision + recall)`
  - Media armónica de precision y recall (métrica principal)
  
- **Coverage**: `notas_con_predicciones / total_notas`
  - ¿Qué porcentaje de notas tienen al menos una predicción?

### Contadores

- **Predictions**: Total de entidades predichas
- **Exact Matches**: Predicciones con `note_id` y `concept_id` correctos
- **Partial Matches**: Predicciones con `note_id` correcto pero `concept_id` incorrecto
- **Ground Truth**: Total de entidades en las anotaciones reales

## 📁 Formato de Resultados

Cuando se usa el flag `-r`, los resultados se guardan en:

```
results/XX_execution_MM_DD_YYYY_HH_MM/
├── evaluation_report.json           # Resumen completo en JSON
├── predictions_01_KIRIs.csv        # Predicciones de KIRIs
├── predictions_02_SNOBERT.csv      # Predicciones de SNOBERT
└── predictions_04_RAG_GPT.csv      # Predicciones de RAG+GPT
```


## 🔧 Configuración de Estrategias

### Activar/Desactivar Estrategias

Editar `main.py`, sección `STRATEGY_CONFIG`:

```python
STRATEGY_CONFIG = {
    1: {
        'id': 1,
        'name': '01_KIRIs',
        'display_name': 'KIRIs REAL (1st Place)',
        'module': '01_kiris',
        'class_name': 'RealKIRIsStrategy',
        'active': True  # Cambiar a False para desactivar
    },
    # ...
}
```

### Agregar Nueva Estrategia

1. **Crear archivo**: `strategies/05_mi_estrategia.py`

2. **Implementar clase** con método `predict()`:

```python
class MiEstrategia:
    def __init__(self):
        # Inicialización
        pass
    
    def predict(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            notes_df: DataFrame con columnas ['note_id', 'text']
        
        Returns:
            DataFrame con columnas:
            - note_id
            - start
            - end
            - concept_id (SNOMED-CT)
            - span_text
            - confidence
        """
        predictions = []
        # ... tu lógica aquí ...
        return pd.DataFrame(predictions)
```

3. **Registrar en `STRATEGY_CONFIG`**:

```python
5: {
    'id': 5,
    'name': '05_mi_estrategia',
    'display_name': 'Mi Estrategia Custom',
    'description': 'Descripción de mi estrategia',
    'module': '05_mi_estrategia',
    'class_name': 'MiEstrategia',
    'active': True
}
```

4. **Ejecutar**:

```bash
python main.py 5 -r
```
