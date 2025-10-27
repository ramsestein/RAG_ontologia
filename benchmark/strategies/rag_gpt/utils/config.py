"""
Utilidades para configuración y carga de recursos
"""

import json
import os
from pathlib import Path
from typing import Dict
from openai import OpenAI


def load_prompt(prompt_name: str, prompts_dir: str = None) -> Dict:
    """
    Carga un prompt desde archivo JSON
    
    Args:
        prompt_name: Nombre del archivo (sin extensión .json)
        prompts_dir: Directorio de prompts (opcional)
        
    Returns:
        Diccionario con configuración del prompt
    """
    if prompts_dir is None:
        # Determinar automáticamente el directorio de prompts
        script_dir = Path(__file__).parent.parent
        prompts_dir = script_dir / "prompts"
    
    prompt_path = Path(prompts_dir) / f"{prompt_name}.json"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def setup_openai_client(project_root: str = None) -> OpenAI:
    """
    Configura cliente de OpenAI
    
    Args:
        project_root: Directorio raíz del proyecto
        
    Returns:
        Cliente OpenAI configurado
    """
    if project_root is None:
        # Determinar automáticamente el project root
        script_dir = Path(__file__).parent.parent
        project_root = script_dir.parent.parent.parent
    
    # Cargar API key
    api_key_path = Path(project_root) / 'api_keys'
    
    try:
        with open(api_key_path, "r") as f:
            for line in f:
                if line.startswith("chatGPT="):
                    api_key = line.split("=")[1].strip()
                    break
            else:
                raise ValueError("chatGPT key not found in api_keys")
    except Exception as e:
        print(f"[WARNING] Error cargando API key: {e}")
        api_key = "YOUR_API_KEY_HERE"
    
    return OpenAI(api_key=api_key)


def get_model_config() -> Dict:
    """
    Retorna configuración del modelo GPT-4o
    
    Returns:
        Diccionario con configuración del modelo
    """
    return {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 0.9
    }


def get_assets_dir() -> Path:
    """
    Retorna el directorio de assets (índice FAISS)
    
    Returns:
        Path absoluto al directorio de assets
    """
    script_dir = Path(__file__).parent.parent
    return script_dir / "04_utils" / "assets"
