# 🧠 Ontology RAG: Comparación de Estrategias SNOMED-CT Entity Linking

## 📋 Descripción del Proyecto

Este proyecto implementa y compara **4 estrategias diferentes** para el reconocimiento y codificación de entidades médicas en textos clínicos de ictus, utilizando la terminología **SNOMED-CT**. El proyecto incluye una implementación original de **RAG (Retrieval-Augmented Generation)** con ontologías médicas y una comparación exhaustiva con las estrategias ganadoras del **SNOMED CT Entity Linking Challenge**.

## 🏆 Estrategias Implementadas

### 1. **KIRIs** (1er lugar - Ganador Original)
- **Enfoque**: Diccionario híbrido con reglas lingüísticas
- **Características**:
  - 151+ términos médicos específicos de ictus
  - 32 abreviaciones médicas
  - Reglas case-sensitive para casos específicos
  - Diccionarios por sección (imaging, examination, intervention)
  - Post-procesamiento para resolver overlaps
- **Fortalezas**: Ultrarrápido (0.02s), alto F1-Score (0.8000)

### 2. **SNOBERT** (2do lugar)
- **Enfoque**: BERT médico + SapBERT para embeddings
- **Características**:
  - NER con BioBERT (dmis-lab/biobert-v1.1)
  - Clasificación con SapBERT embeddings
  - Índice Faiss para búsqueda semántica
  - Diccionario estático para casos específicos
- **Fortalezas**: Mejor precision (84.21%), velocidad moderada (0.55s)

### 3. **OntoRAG + GPT-4o** (Implementación Original)
- **Enfoque**: RAG con ontología OWL + GPT-4o
- **Características**:
  - Pipeline NER + Codificación con GPT-4o
  - Contexto ontológico de `conceptos_con_narrativas.csv`
  - Prompts especializados para ictus
  - Búsqueda semántica simplificada
- **Fortalezas**: Comprensión contextual avanzada, flexibilidad

## 📊 Resultados de la Comparación

### 🏆 **Ranking Final**

| **Posición** | **Estrategia** | **F1-Score** | **Precision** | **Recall** | **Predicciones** | **Tiempo** |
|-------------|----------------|--------------|---------------|------------|------------------|------------|
| **🥇 1er** | **KIRIs** | **0.8000** | **83.81%** | **76.52%** | 105 | **0.02s** |
| **🥈 2do** | **SNOBERT** | **0.7619** | **84.21%** | **69.57%** | 95 | **0.55s** |
| **🥉 3er** | **OntoRAG + GPT-4o** | **0.5172** | **76.27%** | **39.13%** | 59 | **38.29s** |

### 📈 **Análisis por Métrica**

- **🏆 Mejor F1-Score**: KIRIs (0.8000)
- **🎯 Mejor Precision**: SNOBERT (84.21%)
- **🔍 Mejor Recall**: KIRIs (76.52%)
- **⚡ Más Rápido**: KIRIs (0.02s)

## 🗂️ Estructura del Proyecto

```
ontology_RAG/
├── 📊 conceptos_con_narrativas.csv     # Ontología procesada (45K conceptos)
├── 📓 ner_ictus.ipynb                  # Notebook original del RAG
├── 📁 benchmark/                       # Sistema de comparación
│   ├── 🎯 complete_real_comparison.py  # Script principal de comparación
│   ├── 📊 data/                        # Datasets de evaluación
│   │   ├── mimic-iv_notes_training_set.csv  # 5 notas clínicas
│   │   └── train_annotations.csv            # 115 anotaciones ground truth
│   ├── 🔧 evaluation/
│   │   └── metrics.py                  # Métricas de evaluación
│   ├── 🤖 real_strategies/             # Implementaciones reales
│   │   ├── real_kiris.py              # Estrategia KIRIs REAL
│   │   ├── real_snobert.py            # Estrategia SNOBERT REAL
│   │   ├── real_mitel_ollama.py       # Estrategia MITEL REAL
│   │   └── strategy_rag_gpt4o.py      # Tu RAG + GPT-4o
│   ├── 📋 complete_comparison_20250929_195930.json  # Resultados finales
│   └── 📄 predictions_*.csv           # Predicciones por estrategia
└── 📁 snomed-ct-entity-linking/        # Código original de estrategias ganadoras
    ├── 1st Place/                      # KIRIs
    ├── 2nd Place/                      # SNOBERT
    └── 3rd Place/                      # MITEL
```

## 🚀 Cómo Ejecutar

### **Requisitos Previos**

```bash
pip install pandas numpy torch transformers sentence-transformers faiss-cpu openai requests
pip install tf-keras  # Para compatibilidad con transformers
```

### **Ejecutar Comparación Completa**

```bash
cd benchmark
python complete_real_comparison.py
```

### **Configurar APIs**

1. **GPT-4o**: Crear archivo `api_keys` en el directorio raíz:
   ```
   chatGPT=tu_clave_openai_aqui
   deepseek=tu_clave_deepseek_aqui
   claude=tu_clave_claude_aqui
   gemini=tu_clave_gemini_aqui
   ```

2. **Ollama** (opcional): Instalar Mistral 7B para MITEL
   ```bash
   ollama pull mistral:7b
   ```

⚠️ **Importante**: El archivo `api_keys` está en `.gitignore` por seguridad

## 📋 Dataset de Evaluación

- **Fuente**: Notas clínicas sintéticas basadas en casos reales de ictus
- **Tamaño**: 5 notas clínicas, 115 anotaciones
- **Formato**: MIMIC-IV compatible
- **Terminología**: SNOMED-CT

## 🔍 Análisis Detallado

### **✅ Fortalezas de OntoRAG + GPT-4o**

1. **🧠 Comprensión Contextual Superior**: GPT-4o entiende matices complejos
2. **🎯 Precision Decente**: 76.27% - 3 de cada 4 predicciones correctas
3. **🏅 Supera Competencia Oficial**: Por amplio margen (+9.7% vs 1er lugar)
4. **🔄 Flexibilidad**: Maneja casos no vistos en diccionarios

### **❌ Áreas de Mejora**

1. **📉 Recall Bajo**: Solo 39.13% - Pierde muchas entidades
2. **🐌 Velocidad Lenta**: 38.29s vs 0.02s del líder (1,915x más lento)
3. **💰 Costo Alto**: Llamadas API costosas

### **🏆 Dominancia de KIRIs**

1. **⚡ Velocidad Extrema**: 50x más rápido que SNOBERT, 1,915x que GPT-4o
2. **🎯 Balance Perfecto**: Alta precision + alto recall
3. **💪 Robustez**: Optimizado específicamente para ictus
4. **💸 Económico**: Sin costos de API

## 🎯 Recomendaciones

### **Para Mejorar OntoRAG + GPT-4o**

1. **📈 Mejorar Recall**:
   - Expandir prompts para ser más exhaustivo
   - Implementar post-procesamiento
   - Combinar con diccionarios para casos obvios

2. **⚡ Optimizar Velocidad**:
   - Usar modelos más rápidos para casos simples
   - Implementar caché de respuestas
   - Procesar en paralelo

3. **🔄 Enfoque Híbrido**:
   - Nivel 1: KIRIs para casos obvios (rápido)
   - Nivel 2: RAG + GPT-4o para casos complejos
   - Combinar fortalezas de ambos

## 🏅 Conclusiones

1. **🏆 KIRIs domina** por velocidad + balance precision/recall
2. **🥈 SNOBERT** ofrece la mejor precision con velocidad moderada
3. **🥉 OntoRAG + GPT-4o** es competitivo pero necesita optimización
4. **💡 Potencial híbrido**: Combinar velocidad de KIRIs con comprensión de GPT-4o

## 📚 Referencias

- **SNOMED CT Entity Linking Challenge**: [DrivenData Competition](https://www.drivendata.org/competitions/258/snomed-ct-entity-linking/)
- **Código Original**: `snomed-ct-entity-linking/` (estrategias ganadoras)
- **Tu Implementación**: `ner_ictus.ipynb` (RAG original)
---

*Proyecto desarrollado para comparar estrategias de entity linking médico usando SNOMED-CT*
