/**
 * Benchmark Service
 * 
 * Handles API calls for NER and RAG benchmarking.
 * Supports modes: all, assembly, single (like diagnose_NER.py CLI)
 */

import api from './api';

// ==============================================================================
// TYPES
// ==============================================================================

export type BenchmarkMode = 'all' | 'assembly' | 'single';

export interface NERModelInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
}

export interface NoteMetrics {
  note_id: string;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
}

export interface SequentialContribution {
  model_id: string;
  incremental_recall: number;
  cumulative_recall: number;
}

export interface ModelBenchmarkResult {
  model_id: string;
  precision: number;
  recall: number;
  f1_micro: number;
  f1_macro: number;
  f1_harmonic: number;
  f1_arithmetic: number;
  total_tp: number;
  total_fp: number;
  total_fn: number;
  processing_time_s: number;
  per_note_metrics: NoteMetrics[];
}

export interface NERBenchmarkResponse {
  mode: BenchmarkMode;
  iou_threshold: number;
  results: ModelBenchmarkResult[];
  sequential_contribution: SequentialContribution[] | null;
  total_processing_time_ms: number;
  notes_processed: number;
  total_entities: number;
}

export interface RAGBenchmarkResponse {
  status: string;
  message: string;
  metrics: null | {
    accuracy: number;
    latency_ms: number;
  };
}

// ==============================================================================
// API FUNCTIONS
// ==============================================================================

/**
 * Get available NER models for benchmarking.
 */
export async function getAvailableModels(): Promise<NERModelInfo[]> {
  const response = await api.get<NERModelInfo[]>('/benchmark/models');
  return response.data;
}

/**
 * Run NER benchmark with specified mode.
 * 
 * @param mode - 'all' | 'assembly' | 'single'
 * @param modelId - Required for 'single' mode
 * @param iouThreshold - IoU threshold for matching (0.0 - 1.0)
 */
export async function runNERBenchmark(
  mode: BenchmarkMode = 'all',
  modelId?: string,
  iouThreshold: number = 0.25
): Promise<NERBenchmarkResponse> {
  const params: Record<string, string | number> = {
    mode,
    iou_threshold: iouThreshold,
  };
  
  if (modelId) {
    params.model_id = modelId;
  }
  
  const response = await api.get<NERBenchmarkResponse>('/benchmark/ner', { params });
  return response.data;
}

/**
 * Get RAG benchmark status.
 */
export async function getRAGBenchmarkStatus(): Promise<RAGBenchmarkResponse> {
  const response = await api.get<RAGBenchmarkResponse>('/benchmark/rag');
  return response.data;
}
