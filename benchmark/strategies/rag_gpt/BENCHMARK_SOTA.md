# Benchmark SOTA - Documentación

## Descripción

Este documento describe el nuevo **Benchmark SOTA (State-of-the-Art)** implementado para evaluar el sistema RAG+GPT con criterios estrictos.

## Diferencias entre Benchmarks

### Benchmark Antiguo (Permisivo)
- **Criterio**: Pares únicos (note_id, concept_id)
- **Total Anotaciones**: 64 pares únicos
- **Matching**: Solo requiere que exista el concepto en la nota (ignora localización)
- **Métrica**: F1-Score simple (micro-average)

### Benchmark SOTA (Estricto)
- **Criterio**: Match exacto con 3 condiciones:
  1. Mismo `note_id`
  2. Mismo `concept_id`
  3. **IoU (Intersection over Union) > 0.5** en los spans
- **Total Anotaciones**: 115 anotaciones completas del ground truth
- **Matching**: Requiere localización precisa (±50% de solapamiento)
- **Métrica**: **Macro-Average F1-Score** (promedio de F1 por nota)

## Fórmula del IoU

El **Intersection over Union (IoU)** mide el solapamiento entre dos spans:

$$
\text{IoU}(P, G) = \frac{|\text{span}_P \cap \text{span}_G|}{|\text{span}_P \cup \text{span}_G|}
$$

Donde:
- $\text{span}_P$ = span predicho (caracteres desde `start` hasta `end`)
- $\text{span}_G$ = span del ground truth
- $\cap$ = intersección (caracteres en común)
- $\cup$ = unión (caracteres totales cubiertos por ambos)

**Umbral**: IoU ≥ 0.5 para considerar un match válido

### Ejemplos de IoU:

1. **Match perfecto** (IoU = 1.0):
   ```
   Ground Truth: "CT angiography" [520-534]
   Predicción:   "CT angiography" [520-534]
   IoU = 14/14 = 1.0 ✓
   ```

2. **Match parcial válido** (IoU = 0.67):
   ```
   Ground Truth: "CT angiography" [520-534]
   Predicción:   "angiography"    [523-534]
   IoU = 11/14 ≈ 0.79 ✓
   ```

3. **Match insuficiente** (IoU = 0.33):
   ```
   Ground Truth: "CT angiography" [520-534]
   Predicción:   "CT"             [520-522]
   IoU = 2/14 ≈ 0.14 ✗
   ```

## Cálculo del Macro-Average F1

El **Macro-Average F1** se calcula en dos pasos:

### Paso 1: Calcular F1 por Nota

Para cada nota $i$ (del 1 al 5):

1. **Contar anotaciones**:
   - $GT_i$ = anotaciones del ground truth en la nota $i$
   - $Pred_i$ = predicciones en la nota $i$

2. **Hacer matching 1-a-1** (con IoU ≥ 0.5):
   - $TP_i$ = True Positives (predicciones que matchean con GT)
   - $FP_i = |Pred_i| - TP_i$ (predicciones sin match)
   - $FN_i = |GT_i| - TP_i$ (anotaciones GT no detectadas)

3. **Calcular métricas**:
   $$
   \text{Precision}_i = \frac{TP_i}{TP_i + FP_i}
   $$
   
   $$
   \text{Recall}_i = \frac{TP_i}{TP_i + FN_i}
   $$
   
   $$
   F1_i = 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
   $$

### Paso 2: Promediar F1 Scores

$$
F1_{\text{Macro}} = \frac{1}{N} \sum_{i=1}^{N} F1_i
$$

Donde $N = 5$ (número de notas).

**Ventaja del Macro-Average**: Da igual peso a cada nota, sin importar cuántas anotaciones tenga.

## Uso

### Ejecutar solo el Benchmark SOTA:

```bash
cd benchmark/strategies/rag_gpt
python scripts/benchmark_sota.py
```

### Ejecutar ambos benchmarks (antiguo + SOTA):

```bash
cd benchmark/strategies/rag_gpt
python scripts/run_rag_gpt.py
```

Esto ejecutará:
1. El pipeline RAG+GPT sobre las 5 notas
2. El benchmark antiguo (permisivo - 64 pares)
3. El benchmark SOTA (estricto - 115 anotaciones + IoU)

### Argumentos opcionales:

```bash
# Cambiar el umbral de IoU (default: 0.5)
python scripts/benchmark_sota.py --iou-threshold 0.7

# Especificar archivos de entrada
python scripts/benchmark_sota.py \
  --input ../../data/mimic-iv_notes_training_set.csv \
  --truth ../../data/train_annotations.csv

# Silenciar output detallado
python scripts/benchmark_sota.py --no-verbose
```

## Resultados Actuales

### Benchmark Antiguo (Permisivo):
- F1-Score: **0.7857**
- Precision: 0.9167
- Recall: 0.6875
- Pares matched: 44/64

### Benchmark SOTA (Estricto):
- **F1-Score Macro-Average: 0.2625** 
- Precision (Micro): 0.3710
- Recall (Micro): 0.2000
- Anotaciones matched: 23/115 (20%)

### Resultados por Nota (SOTA):

| Nota | GT Annotations | Predictions | TP | FP | FN | Precision | Recall | F1-Score |
|------|----------------|-------------|----|----|-------|-----------|--------|----------|
| 1    | 33             | 15          | 8  | 7  | 25    | 0.5333    | 0.2424 | 0.3333   |
| 2    | 16             | 12          | 6  | 6  | 10    | 0.5000    | 0.3750 | 0.4286   |
| 3    | 23             | 9           | 3  | 6  | 20    | 0.3333    | 0.1304 | 0.1875   |
| 4    | 22             | 9           | 4  | 5  | 18    | 0.4444    | 0.1818 | 0.2581   |
| 5    | 21             | 17          | 2  | 15 | 19    | 0.1176    | 0.0952 | 0.1053   |

**Macro-Average F1**: (0.3333 + 0.4286 + 0.1875 + 0.2581 + 0.1053) / 5 = **0.2625**

## Análisis de Resultados

### Problemas Identificados:

1. **Gran brecha entre benchmarks** (0.79 vs 0.26):
   - El benchmark antiguo no penaliza errores de localización
   - El sistema encuentra conceptos correctos pero en posiciones incorrectas

2. **Baja Recall (20%)**: 
   - 92/115 anotaciones del ground truth no son detectadas
   - Problema principal: **NER no extrae suficientes entidades**
   - O bien: **RAG/Coding mapea a conceptos incorrectos**

3. **Precision moderada (37%)**:
   - 39/62 predicciones no tienen match en el ground truth
   - Posibles causas:
     - Spans mal localizados (IoU < 0.5)
     - Conceptos SNOMED incorrectos
     - Entidades espurias extraídas por NER

4. **Notas 3 y 5 críticas** (F1 < 0.20):
   - Requieren análisis detallado para identificar patrones de fallo

### Próximos Pasos:

1. **Análisis de spans fallidos**: Ver qué IoU tienen las predicciones que fallan
2. **Diagnóstico NER**: Identificar qué entidades del GT no son extraídas
3. **Diagnóstico RAG/Coding**: Ver qué entidades NER son mal codificadas
4. **Mejora de localización**: Ajustar lógica de `find_span_in_text` para mejor precisión

## Archivos

- `scripts/benchmark_sota.py`: Script principal del benchmark SOTA
- `scripts/run_rag_gpt.py`: Script que ejecuta ambos benchmarks
- `scripts/analyze_missing_pairs.py`: Análisis de pares faltantes (benchmark antiguo)

## Referencias

- Train annotations: `benchmark/data/train_annotations.csv` (115 anotaciones)
- Input notes: `benchmark/data/mimic-iv_notes_training_set.csv` (5 notas)
- Metrics calculator: `benchmark/evaluation/metrics_calculator.py` (benchmark antiguo)
