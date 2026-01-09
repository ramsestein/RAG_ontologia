### 1. Descodificación de las Claves (Metadatos)

Estas claves provienen de un sistema HL7 o DICOM estándar de gestión radiológica (RIS/PACS). Entenderlas es vital para filtrar "ruido":

* **`UO_PACN` (Unidad Organizativa Paciente):** El departamento solicitante o donde se realiza (ej. "RADIO" = Radiología). *Irrelevante para el NER clínico.*
* **`PRUEB` (Prueba):** El nombre del estudio solicitado (ej. "TC CODIGO ICTUS"). *Crucial* para saber el contexto (si es un TC de cráneo o una RM de rodilla).
* **`COD_TECN` (Código Técnica):** Código de facturación o procedimiento interno (ej. "9601A"). *Irrelevante.*
* **`CLIN` (Clínica):** El motivo de consulta o historia clínica breve (ej. "Infarto cerebral..."). *Muy relevante* para priors (probabilidad a priori), pero a veces está en catalán o muy escueto.
* **`OBSR` (Observaciones):** Suele contener el protocolo técnico usado. *Poco valor semántico* para el diagnóstico.
* **`RESL` (Resultados):** **El corazón de tu problema.** Aquí está la descripción radiológica completa.
* **`CONCL` (Conclusiones):** El resumen diagnóstico. *Alta densidad de información.*
* **`ACTI_ADICI` / `DKTXT` / `ICON` / `X00DIATXT` / `ZVASOS_D` / `EST_T_DES`:** Campos administrativos, firmas electrónicas, o códigos de actividad adicional. Generalmente son *ruido* para tu tarea de extracción de entidades.

---

### 2. Análisis Crítico: Problemas del Dataset

Este dataset, aunque ya convertido a JSON, presenta **5 problemas estructurales graves** que harán fallar a tu modelo si no se tratan antes de la fase de NER/Ontología.

#### A. Heterogeneidad Estructural (El problema del "Saco Roto")

Tu script de parseo (el paso anterior de `.txt` a `.json`) dividió el texto basándose en cabeceras en mayúsculas (`RESL:`, `CONCL:`). Sin embargo, los radiólogos no son consistentes.

* **Caso 1:** En algunos archivos, `RESL` contiene *todo* el texto mezclado (TC Basal, Perfusión, Angio).
* **Caso 2:** En otros (ej. `...58643591_cleaned.txt`), aparece una clave explícita `"ANGIOTC"` al mismo nivel que `RESL`.
* **Consecuencia:** Tu algoritmo de extracción no puede confiar en que `RESL` tenga siempre la misma estructura. Si buscas "oclusión", a veces estará en `RESL` y otras veces en una clave `ANGIOTC` que tu script anterior quizás capturó o quizás dejó dentro de `RESL`.

#### B. La Torre de Babel (Mezcla de Idiomas)

Tienes un problema severo de **Code-Switching**.

* Ejemplo en `CLIN`: *"Infart cerebral causat per oclusió..."* (Catalán).
* Ejemplo en `RESL`: *"No se observa hemorragia..."* (Español).
* **Consecuencia:** Tu taxonomía (`taxonomia.json`) está en español (`hemorragia`). Si usas un *matcher* exacto (DFA), fallará en "Infart" o "vòmits". Necesitas normalización lingüística o un modelo multilingüe/cross-lingual (como `xlm-roberta` o un LLM con prompt específico).

#### C. Ambigüedad de Abreviaturas y "Jerga"

El texto está lleno de siglas críticas que varían:

* `ACM`, `ACMs`, `ACM izquierda`, `ACMi`.
* `TICA`, `T-ICA`, `carótida interna terminal`.
* `AV`, `AVs`, `arteria vertebral`.
* **Consecuencia:** Si tu ontología espera `arteria cerebral media` y el texto dice `ACMi`, el enlace se rompe. Necesitas un paso de **Normalización de Entidades (Entity Linking)** muy robusto antes de intentar generar el árbol.

#### D. El Contexto Temporal (La trampa de la comparación)

Muchos informes empiezan con: *"Se compara con estudio previo del xx/xx/xxxx..."*

* El texto dice: *"Hipodensidad parietal derecha..."* (refiriéndose al estudio **antiguo** para compararlo).
* **Riesgo:** Tu NER extraerá "Hipodensidad" como un hallazgo *actual* del paciente, cuando en realidad podría ser una secuela antigua descrita para comparación. Tu árbol mostrará una enfermedad que el paciente *tuvo*, no necesariamente la que *tiene ahora* como aguda. Necesitas detectar si el hallazgo es "agudo", "crónico" o "secuela".

#### E. Negación y Especulación

El texto médico está plagado de: *"No se observa..."*, *"Sin signos de..."*, *"Dudosa imagen..."*.

* Si extraes "hemorragia" de la frase *"No se observa hemorragia"*, tu árbol marcará al paciente como crítico falsamente.
* **Solución:** Tu pipeline DEBE tener un módulo de detección de negación (`NegEx` o similar) que corra *sobre* los resultados del NER.

---
