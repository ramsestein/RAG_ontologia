import sys
import os
import pandas as pd
import argparse
import time
import json
import re
import importlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
STRATEGIES_DIR = os.path.join(SCRIPT_DIR, 'strategies')
EVALUATION_DIR = os.path.join(SCRIPT_DIR, 'evaluation')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Add to path
for path in [SCRIPT_DIR, STRATEGIES_DIR, EVALUATION_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import metrics calculator
from evaluation.metrics_calculator import MetricsCalculator


# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

STRATEGY_CONFIG = {
    1: {
        'id': 1,
        'name': '01_KIRIs',
        'display_name': 'KIRIs REAL (1st Place)',
        'description': 'Hybrid dictionary + linguistic rules',
        'module': '01_kiris',
        'class_name': 'RealKIRIsStrategy',
        'active': True
    },
    2: {
        'id': 2,
        'name': '02_SNOBERT',
        'display_name': 'SNOBERT REAL (2nd Place)',
        'description': 'BERT + SapBERT + embeddings',
        'module': '02_snobert',
        'class_name': 'RealSNOBERTStrategy',
        'active': True
    },
    3: {
        'id': 3,
        'name': '03_Ollama',
        'display_name': 'MITEL REAL (3rd Place)',
        'description': 'Mistral 7B via Ollama + RAG',
        'module': '03_ollama',
        'class_name': 'RealMITELOllamaStrategy',
        'active': False  # Currently disabled
    },
    4: {
        'id': 4,
        'name': '04_RAG_GPT',
        'display_name': 'RAG + GPT-4o',
        'description': 'Custom RAG system with GPT-4o',
        'module': '04_rag_gpt',
        'class_name': 'RAGWithGPT4oStrategy',
        'active': True
    }
}

# Active strategies (IDs)
ACTIVE_STRATEGIES = [sid for sid, config in STRATEGY_CONFIG.items() if config['active']]


# ============================================================================
# STRATEGY LOADING
# ============================================================================

def load_strategy(strategy_id: int) -> Tuple[Optional[object], Dict]:
    """
    Dynamically load and instantiate a strategy.
    
    Args:
        strategy_id: ID of the strategy to load
        
    Returns:
        Tuple of (strategy_instance, config_dict)
    """
    
    if strategy_id not in STRATEGY_CONFIG:
        print(f"[ERROR] Invalid strategy ID: {strategy_id}")
        return None, {}
    
    config = STRATEGY_CONFIG[strategy_id]
    
    if not config['active']:
        print(f"[WARNING] Strategy {strategy_id} ({config['display_name']}) is currently disabled")
        return None, config
    
    print(f"\n{'='*80}")
    print(f"INITIALIZING: {config['display_name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    try:
        # Dynamic import using importlib.util for modules with numeric names
        import importlib.util
        
        module_name = config['module']
        module_path = os.path.join(STRATEGIES_DIR, f"{module_name}.py")
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Strategy module not found: {module_path}")
        
        # Load module from file path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get strategy class
        strategy_class = getattr(module, config['class_name'])
        strategy_instance = strategy_class()
        
        print(f"[SUCCESS] {config['display_name']} initialized successfully")
        return strategy_instance, config
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize {config['display_name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, config

# ejecuta 1 estrategia
def run_strategy(strategy_id: int, 
                notes_df: pd.DataFrame, 
                annotations_df: pd.DataFrame,
                metrics_calc: MetricsCalculator) -> Dict:
    # Load strategy
    strategy, config = load_strategy(strategy_id)
    
    if strategy is None:
        return {
            'config': config,
            'metrics': None,
            'predictions': None,
            'execution_time': 0.0,
            'error': 'Failed to load strategy'
        }
    
    strategy_name = config['name']
    
    print(f"\n{'='*100}")
    print(f"EXECUTING: {config['display_name']}")
    print(f"{'='*100}")
    
    start_time = time.time()
    
    try:
        # Execute prediction
        print(f"[INFO] Running predictions on {len(notes_df)} notes...")
        predictions = strategy.predict(notes_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"[INFO] Generated {len(predictions)} predictions in {execution_time:.2f} seconds")
        
        # Calculate metrics
        metrics = metrics_calc.calculate_metrics(predictions, annotations_df, strategy_name)
        
        # Print single report
        report = metrics_calc.format_single_report(metrics, execution_time, config['display_name'])
        print(report)
        
        return {
            'config': config,
            'metrics': metrics,
            'predictions': predictions,
            'execution_time': execution_time,
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"[ERROR] Failed to execute {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'config': config,
            'metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'predictions': 0,
                'matches': 0,
                'partial_matches': 0,
                'ground_truth': len(annotations_df),
                'coverage': 0.0
            },
            'predictions': None,
            'execution_time': execution_time,
            'error': str(e)
        }


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def create_execution_directory() -> Tuple[str, str]:
    # Find next execution number
    exec_num = 1
    dir_pattern = re.compile(r"^(\d+)_execution_.*$")
    existing_nums = []
    
    for dirname in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, dirname)
        if os.path.isdir(full_path):
            match = dir_pattern.match(dirname)
            if match:
                existing_nums.append(int(match.group(1)))
    
    if existing_nums:
        exec_num = max(existing_nums) + 1
    
    # Create directory name with timestamp
    timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
    exec_num_str = f"{exec_num:02d}"
    dir_name = f"{exec_num_str}_execution_{timestamp_str}"
    
    # Create directory
    full_path = os.path.join(RESULTS_DIR, dir_name)
    os.makedirs(full_path, exist_ok=True)
    
    return dir_name, full_path


def save_results(results_dict: Dict, 
                notes_df: pd.DataFrame, 
                annotations_df: pd.DataFrame,
                execution_dir: str) -> None:
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    # Prepare summary data
    summary = {
        'timestamp': datetime.now().strftime("%m_%d_%Y_%H_%M"),
        'execution_folder': os.path.basename(execution_dir),
        'strategies_evaluated': [],
        'dataset_info': {
            'notes_count': len(notes_df),
            'annotations_count': len(annotations_df)
        },
        'results': {},
        'ranking': []
    }
    
    # Process each strategy result
    for strategy_id, result in results_dict.items():
        config = result['config']
        metrics = result['metrics']
        exec_time = result['execution_time']
        predictions = result['predictions']
        
        strategy_name = config['name']
        summary['strategies_evaluated'].append(strategy_name)
        
        # Add to summary
        summary['results'][strategy_name] = {
            'display_name': config['display_name'],
            'description': config['description'],
            'metrics': metrics,
            'execution_time': exec_time,
            'error': result.get('error')
        }
        
        # Save predictions CSV
        if predictions is not None and len(predictions) > 0:
            pred_filename = os.path.join(execution_dir, f"predictions_{strategy_name}.csv")
            predictions.to_csv(pred_filename, index=False, encoding="utf-8")
            print(f"[SAVED] Predictions: {pred_filename}")
    
    # Create ranking
    ranked = sorted(
        [(name, data['metrics']['f1']) for name, data in summary['results'].items()],
        key=lambda x: x[1],
        reverse=True
    )
    summary['ranking'] = [{'strategy': name, 'f1_score': f1} for name, f1 in ranked]
    
    # Save JSON report
    report_filename = os.path.join(execution_dir, "evaluation_report.json")
    with open(report_filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"[SAVED] Summary report: {report_filename}")
    print(f"\n[INFO] All results saved to: {execution_dir}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SNOMED-CT entity linking strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_strategy.py                 # Run all active strategies (1, 2, 4)
  python evaluate_strategy.py -s 1            # Run only strategy 1 (KIRIs)
  python evaluate_strategy.py -s 2            # Run only strategy 2 (SNOBERT)
  python evaluate_strategy.py -s 4            # Run only strategy 4 (RAG GPT-4o)
  python evaluate_strategy.py -s 1 4          # Run strategies 1 and 4, then compare

Active Strategies:
  1: KIRIs REAL - Hybrid dictionary + linguistic rules
  2: SNOBERT REAL - BERT + SapBERT + embeddings
  4: RAG + GPT-4o - Custom RAG system with GPT-4o
  
Note: Strategy 3 (Ollama) is currently disabled.
        """
    )
    
    parser.add_argument(
        '-s', '--strategy-id',
        nargs='+',
        type=int,
        choices=[1, 2, 3, 4],
        help='Strategy ID(s) to run. If not specified, runs all active strategies.'
    )
    
    args = parser.parse_args()
    
    # Determine which strategies to run
    if args.strategy_id:
        strategies_to_run = args.strategy_id
        # Filter out inactive strategies
        strategies_to_run = [sid for sid in strategies_to_run 
                            if STRATEGY_CONFIG[sid]['active']]
        
        if not strategies_to_run:
            print("[ERROR] All specified strategies are inactive.")
            return
    else:
        # Default: run all active strategies
        strategies_to_run = ACTIVE_STRATEGIES
    
    # Print header
    print("\n" + "="*100)
    print("SNOMED-CT ENTITY LINKING - STRATEGY EVALUATION")
    print("="*100)
    print(f"\nStrategies to evaluate: {', '.join([STRATEGY_CONFIG[sid]['display_name'] for sid in strategies_to_run])}")
    print("="*100)
    
    # Load datasets
    print("\n[LOADING] Loading datasets...")
    try:
        notes_path = os.path.join(DATA_DIR, "mimic-iv_notes_training_set.csv")
        annotations_path = os.path.join(DATA_DIR, "train_annotations.csv")
        
        notes_df = pd.read_csv(notes_path)
        annotations_df = pd.read_csv(annotations_path)
        
        print(f"[SUCCESS] Loaded {len(notes_df)} notes")
        print(f"[SUCCESS] Loaded {len(annotations_df)} annotations")
        
    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        return
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Run strategies
    results = {}
    
    for strategy_id in strategies_to_run:
        result = run_strategy(strategy_id, notes_df, annotations_df, metrics_calc)
        results[strategy_id] = result
    
    # Print comparison report if multiple strategies
    if len(results) > 1:
        comparison_data = {
            result['config']['name']: {
                'metrics': result['metrics'],
                'execution_time': result['execution_time']
            }
            for result in results.values()
            if result['metrics'] is not None
        }
        
        comparison_report = metrics_calc.format_comparison_report(comparison_data)
        print(comparison_report)
    
    # Create execution directory and save results
    dir_name, exec_path = create_execution_directory()
    save_results(results, notes_df, annotations_df, exec_path)
    
    # Final message
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETED")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
