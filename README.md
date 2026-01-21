# Proyecto RAG Ontología

Bienvenido al repositorio oficial del proyecto **RAG Ontología**. Este proyecto representa un esfuerzo de vanguardia en el desarrollo e implementación de soluciones basadas en **Generación Aumentada por Recuperación (RAG)** aplicadas a dominios ontológicos complejos, específicamente dentro del ámbito clínico y biomédico.

Desarrollado en el **Hospital Clínic de Barcelona**, este repositorio alberga la investigación y el código fuente necesarios para abordar desafíos críticos en la extracción de información y el procesamiento de lenguaje natural (PLN) clínico.

## 📋 Tabla de Contenidos

1.  [Descripción del Proyecto](#descripción-del-proyecto)
2.  [Estructura del Repositorio](#estructura-del-repositorio)
3.  [Soluciones Implementadas](#soluciones-implementadas)
4.  [Tecnologías Clave](#tecnologías-clave)
5.  [Autores y Reconocimientos](#autores-y-reconocimientos)
6.  [Licencia](#licencia)

## 🏥 Descripción del Proyecto

El objetivo principal de este proyecto es mejorar la precisión y la interpretabilidad de los sistemas de IA en entornos clínicos mediante el uso de ontologías estructuradas (como SNOMED CT) para guiar y validar la generación de respuestas. El proyecto se divide en fases iterativas de resolución de problemas, donde cada fase aborda una complejidad creciente en la vinculación de entidades y el razonamiento ontológico.

## 📂 Estructura del Repositorio

El código base está organizado meticulosamente para separar las diferentes iteraciones de investigación y desarrollo. A continuación, se detalla la estructura principal:

```
RAG_ontologia/
├── ofarres/             # Solución al Planteamiento del Primer Problema
├── new_ofarres/         # Solución al Planteamiento del Nuevo Problema (Actual)
├── snomed-ct-entity.../ # Recursos y herramientas de vinculación de entidades SNOMED
├── .venv/               # Entorno virtual (no incluido en control de versiones)
├── README.md            # Este archivo
├── SETUP.md             # Guía de instalación y configuración
├── LICENSE.md           # Términos de licencia y uso
├── CONTRIBUTING.md      # Guías para colaboradores
└── CHANGELOG.md         # Registro de cambios y versiones
```

## 🚀 Soluciones Implementadas

### 1. Solución Inicial (`ofarres/`)
Este directorio contiene la implementación original diseñada para el primer conjunto de desafíos planteados.
*   **Enfoque:** Métodos tradicionales de recuperación y alineación básica con ontologías.
*   **Estado:** Archivado/Referencia. Útil para entender la línea base del proyecto.

### 2. Solución Avanzada (`new_ofarres/`)
Este directorio alberga el desarrollo actual y más avanzado, respondiendo a los nuevos desafíos y complejidades descubiertas.
*   **Enfoque:** Uso de ensembles de modelos NER (Reconocimiento de Entidades Nombradas), algoritmos de grafos para linaje ontológico, y estrategias de RAG más sofisticadas.
*   **Estado:** Activo/En Desarrollo. Aquí reside el código fuente principal para la fase actual.

## 🛠️ Tecnologías Clave

Este proyecto hace uso extensivo de un stack tecnológico robusto en Python:

*   **Procesamiento de Lenguaje Natural:** `spaCy`, `scispacy` (modelos biomédicos), `transformers` (Hugging Face).
*   **Ontologías y Grafos:** `owlready2` para manipulación de OWL, algoritmos de grafos personalizados.
*   **Machine Learning/Deep Learning:** `torch`, `tensorflow` (según los modelos específicos utilizados).
*   **Infraestructura de Datos:** Gestión eficiente de datasets médicos y salidas JSON estructuradas.

## 👥 Autores y Reconocimientos

Este proyecto es el resultado del trabajo dedicado del **Departamento de Informática Clínica** del **Hospital Clínic de Barcelona**.

### Autor Principal
*   **Oriol Farrés**
    *   *Rol:* Desarrollador Principal e Investigador.
    *   *Contribución:* Implementación del código, diseño de algoritmos y experimentación.

### Supervisión y Dirección
*   **Santiago Frid**
    *   *Rol:* Supervisor del Proyecto.
    *   *Contribución:* Orientación estratégica y revisión técnica.
*   **Ramsés Marrero**
    *   *Rol:* Supervisor del Proyecto.
    *   *Contribución:* Definición de objetivos clínicos y validación de metodología.

## 📄 Licencia

Este software es propiedad intelectual del **Hospital Clínic de Barcelona**.

El uso, modificación y distribución de este software están regidos por las políticas internas del Hospital. Para más detalles, consulte el archivo [LICENSE.md](LICENSE.md).

---
© Hospital Clínic de Barcelona - Departamento de Informática Clínica
