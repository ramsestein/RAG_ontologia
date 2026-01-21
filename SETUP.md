# Guía de Configuración e Instalación

Esta documentación proporciona una guía paso a paso y exhaustiva para configurar el entorno de desarrollo necesario para el proyecto **RAG Ontología**.

Dado que este proyecto utiliza bibliotecas avanzadas de Procesamiento de Lenguaje Natural (PLN) y Deep Learning, es crucial seguir estos pasos meticulosamente para asegurar la compatibilidad y el funcionamiento correcto.

## 📋 Prerrequisitos del Sistema

Antes de comenzar, asegúrese de tener instalado el siguiente software en su sistema:

*   **Python 3.10+**: El proyecto ha sido probado y optimizado para versiones modernas de Python. [Descargar Python](https://www.python.org/downloads/)
*   **Git**: Sistema de control de versiones para clonar y gestionar el repositorio. [Descargar Git](https://git-scm.com/downloads)
*   **pip**: El instalador de paquetes de Python (generalmente incluido con Python).
*   **Herramientas de Compilación (C++ Build Tools)**: Algunas dependencias (como `nmslib` o `scipy`) pueden requerir compiladores de C++ instalados en el sistema (especialmente en Windows).

## 🔧 Pasos de Instalación

### 1. Clonar el Repositorio

Obtenga una copia local del código fuente ejecutando el siguiente comando en su terminal:

```bash
git clone https://github.com/ramsestein/RAG_ontologia.git
cd RAG_ontologia
```

### 2. Configuración del Entorno Virtual

Es **altamente recomendable** y considerado una mejor práctica utilizar un entorno virtual para aislar las dependencias de este proyecto y evitar conflictos con otras instalaciones de Python en su sistema.

#### Creación del Entorno
Ejecute el siguiente comando para crear un entorno virtual llamado `.venv`:

```bash
python -m venv .venv
```

#### Activación del Entorno

*   **En Windows (PowerShell/CMD):**
    ```bash
    .venv\Scripts\activate
    ```
    *Si encuentra errores de permisos en PowerShell, puede necesitar ejecutar `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.*

*   **En macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```

Una vez activado, verá `(.venv)` al principio de su línea de comandos.

### 3. Instalación de Dependencias

Con el entorno virtual activado, instale las bibliotecas necesarias listadas en el archivo `requirements.txt`. Este archivo incluye versiones específicas de bibliotecas como TensorFlow, PyTorch, SpaCy y otras utilidades científicas.

```bash
pip install -r requirements.txt
```

> ⚠️ **Nota Importante:** La instalación puede tardar varios minutos debido al tamaño de paquetes como `torch` o `tensorflow`. Asegúrese de tener una conexión a internet estable.

### 4. Descarga de Modelos de Lenguaje (SpaCy)

El proyecto depende de modelos de lenguaje específicos para el reconocimiento de entidades biomédicas. Aunque el archivo `requirements.txt` intenta instalar algunos directamente, es posible que deba descargarlos manualmente si la instalación automática falla.

Comandos para instalación manual si es necesario:

```bash
# Modelo base en inglés
python -m spacy download en_core_web_sm

# Modelos científicos (ScispaCy)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz
```

## 🚀 Ejecución del Proyecto

El proyecto tiene dos puntos de entrada principales según el problema que desee abordar:

### Solución Original (`ofarres`)
Para ejecutar los scripts correspondientes al primer planteamiento del problema:

```bash
cd ofarres
# Ejecute los scripts específicos, por ejemplo:
python main.py
```

### Solución Nueva (`new_ofarres`)
Para trabajar con la implementación más reciente:

```bash
cd new_ofarres
# Navegue a los directorios de código fuente (src)
cd src
# Ejecute los módulos necesarios
```

## ❓ Solución de Problemas Comunes

*   **Error: `Microsoft Visual C++ 14.0 is required`**: En Windows, instale las "Build Tools for Visual Studio" seleccionando la carga de trabajo "Desarrollo para el escritorio con C++".
*   **Conflictos de versión de `protobuf`**: Si encuentra errores relacionados con `protobuf`, asegúrese de no tener versiones conflictivas instaladas entre TensorFlow y otras librerías de Google. `pip install --upgrade protobuf` suele resolverlo.

---
Si tiene problemas que no se resuelven con esta guía, por favor contacte a los autores o abra un "Issue" en el repositorio.
