# 🔍 Guía de Debugging - RAG+GPT4o Strategy

## 🎯 Problema Actual

**Estado:** F1-Score = 0.0000 (0 exact matches de 67 predicciones)

**Análisis:**
- ✅ **NER funciona**: Extrae 67 entidades
- ✅ **Búsqueda de spans funciona**: 67 partial matches
- ❌ **Linking FALLA**: 0 exact matches → todos los `concept_id` son incorrectos

**Causa raíz probable:** GPT-4o no está devolviendo los códigos correctos del contexto RAG, o está usando los defaults (404684003, 12738006) que nunca coinciden con el ground truth.

---

## 🚀 Cambios Implementados

### 1. **Logging Detallado en `_execute_coding_step`**

Ahora imprime para CADA entidad:
```
[RAG+GPT4o] 🔍 Codificando entidad 1/67
[RAG+GPT4o]   📝 Span: 'ictus isquémico'
[RAG+GPT4o]   📍 Ubicación: 'Not specified'
[RAG+GPT4o]   ✓ Presencia: 'present'
[RAG+GPT4o]   🔎 Contexto RAG recuperado: 5 conceptos
[RAG+GPT4o]   🎯 Mejor match: 230690007 (dist: 0.234)
[RAG+GPT4o]   🤖 Consultando GPT-4o...
[RAG+GPT4o]   📥 Respuesta GPT-4o (primeros 200 chars):
[RAG+GPT4o]      {"entity_code": "230690007", "entity_description": "Stroke", ...
[RAG+GPT4o]   ✅ Códigos asignados:
[RAG+GPT4o]      • entity_code: 230690007
[RAG+GPT4o]      • anatomy_code: 12738006
[RAG+GPT4o]      • presence_code: 52101004
```

### 2. **Alertas de Códigos Genéricos**

```
[RAG+GPT4o]   ⚠️  ALERTA: Usando entity_code DEFAULT (Clinical finding)
[RAG+GPT4o]   ⚠️  ALERTA: Usando anatomy_code DEFAULT (Brain structure)
```

### 3. **Prompt Mejorado (v6)**

- Formato más claro con "CODE:" y "DESCRIPTION:"
- Ejemplos explícitos de cómo extraer códigos
- Instrucciones más estrictas
- Temperature reducida a 0.1 (más determinístico)

### 4. **Más Contexto RAG**

- Aumentado de k=3 a k=5 conceptos recuperados
- Narrativas truncadas a 200 chars para claridad

### 5. **Mejor Manejo de Errores**

- Códigos especiales "LINKING_FAILED" para identificar fallos
- JSON de error válido cuando GPT-4o falla
- Validación estricta de campos requeridos

### 6. **Script de Debug**

Nuevo archivo `debug_rag.py` para probar con una sola nota.

---

## 📋 Pasos para Debuggear

### Paso 1: Ejecutar con UNA nota

```bash
cd /c/Users/OFARRES/Desktop/RAG_ontologia
source .venv/Scripts/activate
cd benchmark
python debug_rag.py
```

Esto te mostrará **TODO** el proceso para una sola nota.

### Paso 2: Analizar los Logs

Busca estas señales:

#### ✅ **Señal BUENA:**
```
[RAG+GPT4o]   🎯 Mejor match: 230690007 (dist: 0.234)
[RAG+GPT4o]   📥 Respuesta GPT-4o: {"entity_code": "230690007", ...
[RAG+GPT4o]   ✅ Códigos asignados:
[RAG+GPT4o]      • entity_code: 230690007  ← CÓDIGO ESPECÍFICO
```

#### ❌ **Señal MALA (Problema #1 - GPT no sigue instrucciones):**
```
[RAG+GPT4o]   🎯 Mejor match: 230690007 (dist: 0.234)
[RAG+GPT4o]   📥 Respuesta GPT-4o: {"entity_code": "404684003", ...
[RAG+GPT4o]   ⚠️  ALERTA: Usando entity_code DEFAULT  ← GPT IGNORA CONTEXTO
```

#### ❌ **Señal MALA (Problema #2 - Respuesta JSON incorrecta):**
```
[RAG+GPT4o]   ❌ ERROR parseando respuesta: ...
[RAG+GPT4o]   ❌ Respuesta completa: {"code": "123", ...}  ← FORMATO INCORRECTO
[RAG+GPT4o]   ❌ Usando códigos FALLBACK
```

#### ❌ **Señal MALA (Problema #3 - RAG devuelve conceptos irrelevantes):**
```
[RAG+GPT4o]   🔎 Contexto RAG recuperado: 5 conceptos
[RAG+GPT4o]   🎯 Mejor match: 999999999 (dist: 5.678)  ← DISTANCIA ALTA = MAL MATCH
```

### Paso 3: Revisar la Respuesta de GPT-4o

En los logs, busca:
```
[RAG+GPT4o]   📥 Respuesta GPT-4o (primeros 200 chars):
```

**Pregunta:** ¿El JSON contiene los campos correctos?
- `entity_code`
- `entity_description`
- `anatomy_code`
- `anatomy_description`
- `presence_code`

**Pregunta:** ¿Los códigos vienen del contexto RAG o son inventados/default?

### Paso 4: Comparar con Ground Truth

El script `debug_rag.py` muestra:
```
Ground Truth: 
  - 230690007: 'ictus isquémico'
  
Predicciones:
  Span: 'ictus isquémico'
  Concept ID: 404684003  ← ❌ INCORRECTO (debería ser 230690007)
```

---

## 🔧 Soluciones según el Problema Detectado

### Solución 1: GPT-4o ignora el contexto

**Si ves:** Códigos DEFAULT a pesar de tener buenos matches en RAG

**Acción:** El prompt necesita ser aún más explícito.

**Modificar en `04_rag_gpt.py`:**

```python
# En _setup_prompts(), cambiar el prompt a algo como:

self.coding_prompt_template = """
You are a SNOMED-CT coder. You MUST extract codes from the context below.

Entity: "{entity}"
Location: "{location}"

Available Codes (YOU MUST CHOOSE FROM HERE):
{contexto_ontologico}

EXAMPLE:
If context has: "CODE: 230690007 | DESCRIPTION: stroke ictus..."
And entity is "ictus"
Then you MUST use: "entity_code": "230690007"

YOUR TASK: Find the best matching CODE from the context above.

OUTPUT (JSON only):
{{
  "entity_code": "THE CODE NUMBER FROM CONTEXT",
  "entity_description": "short description",
  "anatomy_code": "THE CODE NUMBER FROM CONTEXT or 12738006",
  "anatomy_description": "short description",
  "presence_code": "{presence_code}"
}}
"""
```

### Solución 2: Formato JSON incorrecto

**Si ves:** Errores de parsing

**Acción:** Verificar que `response_format={"type": "json_object"}` esté funcionando.

**Alternativa:** Usar regex más robusto para extraer JSON.

### Solución 3: RAG recupera conceptos irrelevantes

**Si ves:** Distancias altas (>2.0) o conceptos no relacionados

**Acciones:**
1. Verificar que el índice Faiss se construyó correctamente
2. Revisar la calidad de las narrativas en `conceptos_con_narrativas.csv`
3. Probar con diferentes k (ej: 7, 10)
4. Considerar normalizar los embeddings

### Solución 4: Verificar que el índice existe

```bash
ls -la benchmark/04_utils/assets/
# Debe mostrar:
# - ontology.index
# - ontology_concepts.pkl
# - ontology_narrativas.pkl
# - ontology_metadata.pkl
```

Si NO existen, ejecuta:
```bash
cd benchmark/04_utils
python build_ontology_index.py
```

---

## 📊 Interpretación de Resultados

Después de ejecutar `debug_rag.py`, verás:

```
📊 RESUMEN:
  Ground Truth: 10 entidades
  Predicciones: 8 entidades
  Exact Matches: 0  ← SI SIGUE EN 0, HAY PROBLEMA
  Precision: 0.00%
  Recall: 0.00%

📈 CÓDIGOS USADOS:
  ⚠️  404684003: 8 veces (CÓDIGO GENÉRICO/FALLBACK)  ← PROBLEMA!
```

**Objetivo:** Ver códigos específicos en lugar de 404684003:
```
📈 CÓDIGOS USADOS:
  ✓ 230690007: 3 veces
  ✓ 55342001: 2 veces
  ✓ 50582007: 2 veces
  ⚠️  404684003: 1 veces (CÓDIGO GENÉRICO/FALLBACK)
```

---

## 🎯 Checklist de Debugging

- [ ] Ejecutar `python debug_rag.py`
- [ ] Verificar que el índice Faiss existe y se carga correctamente
- [ ] Revisar logs de RAG: ¿recupera conceptos relevantes?
- [ ] Revisar logs de GPT-4o: ¿devuelve JSON correcto?
- [ ] Verificar códigos asignados: ¿son específicos o defaults?
- [ ] Comparar con ground truth: ¿los códigos coinciden?
- [ ] Si usa defaults: ¿por qué? (ver respuesta de GPT)
- [ ] Si GPT ignora contexto: mejorar prompt
- [ ] Si RAG falla: revisar embeddings/narrativas

---

## 🚀 Siguiente Paso

1. **Ejecuta:** `python debug_rag.py` en el benchmark
2. **Observa:** Los logs detallados
3. **Identifica:** Cuál de los 3 problemas es la causa (o combinación)
4. **Reporta:** Qué ves en los logs para ajustar la solución

---

**Versión:** v6.0 - Con Logging Detallado  
**Fecha:** Octubre 2025
