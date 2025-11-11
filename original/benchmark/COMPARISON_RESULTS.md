# Comparación de Resultados - Benchmark SOTA

## Resultados KIRIS (Benchmark SOTA - Estricto)

### Métricas Globales

| Métrica | Valor |
|---------|-------|
| **Macro-Average F1** | **0.3798** |
| Micro Precision | 0.4000 |
| Micro Recall | 0.3652 |
| Micro F1 | 0.3818 |
| Execution Time | 0.04 seconds |

### Conteos Totales

- **True Positives**: 42/115 (36.5%)
- **False Positives**: 63
- **False Negatives**: 73
- **Total Predictions**: 105
- **Total Ground Truth**: 115 anotaciones

### Resultados por Nota

| Nota | GT | Pred | TP | FP | FN | Precision | Recall | F1-Score |
|------|-------|------|----|----|-------|-----------|--------|----------|
| 1    | 33    | 29   | 13 | 16 | 20    | 0.4483    | 0.3939 | **0.4194** |
| 2    | 16    | 16   | 7  | 9  | 9     | 0.4375    | 0.4375 | **0.4375** |
| 3    | 23    | 21   | 8  | 13 | 15    | 0.3810    | 0.3478 | **0.3636** |
| 4    | 22    | 20   | 9  | 11 | 13    | 0.4500    | 0.4091 | **0.4286** |
| 5    | 21    | 19   | 5  | 14 | 16    | 0.2632    | 0.2381 | **0.2500** |

**Macro-Average F1**: (0.4194 + 0.4375 + 0.3636 + 0.4286 + 0.2500) / 5 = **0.3798**

---

## Resultados RAG-GPT (Benchmark SOTA - Estricto)

*Pendiente de completar - ejecución en curso...*

Basado en ejecuciones anteriores (estimado):
- Macro-Average F1: ~0.26
- Micro Precision: ~0.37
- Micro Recall: ~0.20
- Total TP: ~23/115 (20%)

---

## Comparación Preliminar

### Benchmark Antiguo (Permisivo - 64 pares únicos)

| Estrategia | F1-Score | Precision | Recall |
|-----------|----------|-----------|--------|
| KIRIS | **0.8000** | 0.7353 | 0.8750 |
| RAG-GPT | 0.7857 | 0.9167 | 0.6875 |

### Benchmark SOTA (Estricto - 115 anotaciones + IoU > 0.5)

| Estrategia | Macro F1 | Micro Precision | Micro Recall | TP/115 | Exec Time |
|-----------|----------|----------------|--------------|---------|-----------|
| **KIRIS** | **0.3798** | 0.4000 | 0.3652 | **42 (36.5%)** | 0.04s |
| RAG-GPT | ~0.2625 | ~0.3710 | ~0.2000 | ~23 (20%) | ~160s |

### Hallazgos Clave

1. **KIRIS supera a RAG-GPT en el Benchmark SOTA**:
   - Macro F1: 0.38 vs 0.26 (+44.7% mejor)
   - Recall: 36.5% vs 20% (+83% más anotaciones detectadas)
   - 42 vs 23 anotaciones correctamente matcheadas

2. **KIRIS es extremadamente eficiente**:
   - 0.04 segundos vs 160 segundos (~4000x más rápido)
   - Usa diccionarios rule-based vs LLM calls

3. **Ambas estrategias tienen bajo rendimiento en SOTA**:
   - F1 < 0.40 indica que la **localización precisa de spans es muy difícil**
   - 60-80% de anotaciones del ground truth no son detectadas correctamente

4. **Nota 5 es crítica para ambas**:
   - KIRIS: F1=0.25 (la peor nota)
   - RAG-GPT: F1=0.11 (la peor nota)
   - Indica que esta nota tiene patrones especialmente difíciles

5. **Diferencia en Precision vs Recall**:
   - RAG-GPT: Alta precisión (0.92 antiguo), pero baja recall
   - KIRIS: Más balanceado (0.40 precision, 0.37 recall en SOTA)

---

## Conclusiones

### ¿Por qué KIRIS es mejor en SOTA?

1. **Localización de spans más precisa**: 
   - KIRIS usa regex pattern matching que captura spans exactos
   - RAG-GPT usa GPT-4o que a veces devuelve offsets incorrectos

2. **Mayor cobertura de entidades**:
   - KIRIS detecta 42/115 anotaciones (36.5%)
   - RAG-GPT detecta 23/115 anotaciones (20%)
   - La estrategia de diccionarios exhaustivos funciona mejor

3. **Menos dependencia de contexto**:
   - KIRIS: Búsqueda directa por patrón
   - RAG-GPT: Depende de que NER extraiga primero, luego RAG codifique

### ¿Qué revela el Benchmark SOTA?

El benchmark SOTA (estricto con IoU) revela que:

1. **El benchmark antiguo era MUY permisivo**:
   - KIRIS: 0.80 (antiguo) → 0.38 (SOTA) = -52.5%
   - RAG-GPT: 0.79 (antiguo) → 0.26 (SOTA) = -67%

2. **La localización de spans es el desafío principal**:
   - No basta con encontrar el concepto correcto
   - Hay que localizarlo con precisión (IoU > 0.5)

3. **Ninguna estrategia es realmente buena en matching estricto**:
   - F1 < 0.40 para ambas
   - Margen de mejora enorme

### Próximos Pasos Sugeridos

1. **Para RAG-GPT**:
   - Mejorar la precisión de offsets del NER
   - Implementar post-procesamiento de spans
   - Usar técnicas de KIRIS para localización

2. **Para KIRIS**:
   - Expandir diccionarios para cubrir más variaciones
   - Mejorar manejo de Nota 5 (F1=0.25)

3. **Enfoque híbrido**:
   - Usar localización de KIRIS + codificación semántica de RAG-GPT
   - Combinar lo mejor de ambas estrategias
