# Project Setup

This guide details the steps to set up the development environment for the RAG Ontologia project.

## Prerequisites

*   Python 3.10 or higher
*   pip (Python package installer)
*   Git

## Installation

1.  **Clone the Repository** (if not already done):
    ```bash
    git clone <repository_url>
    cd RAG_ontologia
    ```

2.  **Create a Virtual Environment**:
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**:

    *   **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    Install the required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

    *Note: This project has extensive dependencies including Spacy models, TensorFlow, and PyTorch. Ensure you have a stable internet connection.*

5.  **Download Spacy Models**:
    The requirements file lists direct URL dependencies for Spacy models like `en_core_sci_scibert`, `en_ner_bc5cdr_md`, etc. These should be installed automatically with the requirements file. If you encounter issues, you may need to install them manually using:
    ```bash
    python -m spacy download en_core_web_sm
    # And others as needed
    ```

## Usage

Navigate to the respective directory for the problem you are working on:

*   For the first problem solution:
    ```bash
    cd ofarres
    ```

*   For the new problem solution:
    ```bash
    cd new_ofarres
    ```

Refer to the specific documentation within those directories for detailed execution instructions.
