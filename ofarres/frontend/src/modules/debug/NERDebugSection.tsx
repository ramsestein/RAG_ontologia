/**
 * NER Debug Section
 * 
 * Displays NER evaluation metrics and benchmarking controls.
 * Supports modes: all, assembly, single (like diagnose_NER.py CLI)
 * Uses Server-Sent Events (SSE) for real-time progress updates with accurate ETA.
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  RefreshCw, 
  CheckCircle2, 
  XCircle, 
  AlertCircle,
  TrendingUp,
  Target,
  Crosshair,
  Clock,
  FileText,
  ChevronDown,
  ChevronUp,
  Layers,
  List,
  Cpu,
  Wifi
} from 'lucide-react';
import { 
  getAvailableModels, 
  runNERBenchmark,
  type NERModelInfo,
  type NERBenchmarkResponse,
  type ModelBenchmarkResult,
  type BenchmarkMode
} from '../../services/benchmark';

interface ProgressState {
  percentage: number;
  message: string;
  etaSeconds: number | null;
  currentModel?: string;
}

export const NERDebugSection: React.FC = () => {
  const [models, setModels] = useState<NERModelInfo[]>([]);
  const [selectedMode, setSelectedMode] = useState<BenchmarkMode>('all');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [iouThreshold, setIouThreshold] = useState<number>(0.25);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<NERBenchmarkResponse | null>(null);
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  
  // Progress tracking from SSE
  const [progress, setProgress] = useState<ProgressState>({
    percentage: 0,
    message: 'Starting...',
    etaSeconds: null
  });
  const [elapsedTime, setElapsedTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Load available models on mount
  useEffect(() => {
    loadModels();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const loadModels = async () => {
    try {
      const availableModels = await getAvailableModels();
      setModels(availableModels);
      // Set default model for single mode
      const firstAvailable = availableModels.find(m => m.available);
      if (firstAvailable) {
        setSelectedModel(firstAvailable.id);
      }
    } catch (err) {
      console.error('Failed to load models:', err);
      setModels([]);
    }
  };

  const startProgressTimer = () => {
    startTimeRef.current = Date.now();
    setElapsedTime(0);
    
    timerRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
      setElapsedTime(elapsed);
    }, 100);
  };

  const stopProgressTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const runBenchmark = async () => {
    setIsLoading(true);
    setError(null);
    setBenchmarkResult(null);
    setProgress({ percentage: 0, message: 'Initializing...', etaSeconds: null });
    startProgressTimer();
    
    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    try {
      // Build SSE URL
      const params = new URLSearchParams({
        mode: selectedMode,
        iou_threshold: iouThreshold.toString()
      });
      if (selectedMode === 'single' && selectedModel) {
        params.set('model_id', selectedModel);
      }
      
      const sseUrl = `http://localhost:8000/api/benchmark/ner/stream?${params.toString()}`;
      console.log('Connecting to SSE:', sseUrl);
      
      const eventSource = new EventSource(sseUrl);
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('SSE event:', data);
          
          if (data.type === 'error') {
            setError(data.message);
            stopProgressTimer();
            setIsLoading(false);
            eventSource.close();
            return;
          }
          
          // Update progress
          setProgress({
            percentage: data.percentage || 0,
            message: data.message || 'Processing...',
            etaSeconds: data.eta_seconds,
            currentModel: data.current_model
          });
          
          // Check if complete
          if (data.type === 'complete' && data.data) {
            setBenchmarkResult(data.data);
            stopProgressTimer();
            setIsLoading(false);
            eventSource.close();
          }
        } catch (e) {
          console.error('Failed to parse SSE data:', e);
        }
      };
      
      eventSource.onerror = (err) => {
        console.error('SSE error:', err);
        
        // Only set error if we were still loading
        if (isLoading && !benchmarkResult) {
          setError('Connection lost. Please try again.');
        }
        stopProgressTimer();
        setIsLoading(false);
        eventSource.close();
      };
      
    } catch (err: any) {
      setError(err.message || 'Failed to start benchmark');
      stopProgressTimer();
      setIsLoading(false);
    }
  };

  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds}s`;
    }
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  const formatEta = (seconds: number | null): string => {
    if (seconds === null) return 'Calculating...';
    if (seconds <= 0) return 'Almost done...';
    return `~${formatTime(Math.round(seconds))} remaining`;
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getScoreColor = (value: number): string => {
    if (value >= 0.9) return 'text-green-600';
    if (value >= 0.7) return 'text-yellow-600';
    if (value >= 0.5) return 'text-orange-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (value: number): string => {
    if (value >= 0.9) return 'bg-green-100';
    if (value >= 0.7) return 'bg-yellow-100';
    if (value >= 0.5) return 'bg-orange-100';
    return 'bg-red-100';
  };

  const getModeIcon = (mode: BenchmarkMode) => {
    switch (mode) {
      case 'all': return <List className="w-4 h-4" />;
      case 'assembly': return <Layers className="w-4 h-4" />;
      case 'single': return <Cpu className="w-4 h-4" />;
    }
  };

  const getModeDescription = (mode: BenchmarkMode) => {
    switch (mode) {
      case 'all': return 'Run each NER model individually and compare results';
      case 'assembly': return 'Run all models together (ensemble) with sequential contribution analysis';
      case 'single': return 'Run a specific model only';
    }
  };

  return (
    <div className="p-6">
      {/* Controls */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">NER Benchmark Configuration</h2>
        
        {/* Mode Selection */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Benchmark Mode
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(['all', 'assembly', 'single'] as BenchmarkMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setSelectedMode(mode)}
                disabled={isLoading}
                className={`flex items-center justify-center gap-2 px-4 py-3 rounded-lg border-2 transition-all ${
                  selectedMode === mode
                    ? 'border-indigo-600 bg-indigo-50 text-indigo-700'
                    : 'border-gray-200 hover:border-gray-300 text-gray-600'
                }`}
              >
                {getModeIcon(mode)}
                <span className="font-medium capitalize">{mode}</span>
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {getModeDescription(selectedMode)}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          {/* Model Selection (only for single mode) */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              NER Model {selectedMode !== 'single' && '(N/A)'}
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 disabled:text-gray-500"
              disabled={isLoading || selectedMode !== 'single'}
            >
              {models.filter(m => m.available).map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>

          {/* IoU Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              IoU Threshold
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={iouThreshold}
              onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Minimum overlap required for a match (0.0 - 1.0)
            </p>
          </div>

          {/* Run Button */}
          <div className="flex items-end">
            <button
              onClick={runBenchmark}
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Benchmark
                </>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Progress Bar - Real-time from SSE */}
        {isLoading && (
          <div className="mt-4 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Wifi className="w-4 h-4 text-green-600 animate-pulse" />
                <span className="font-medium text-indigo-800">{progress.message}</span>
              </div>
              <div className="text-sm text-indigo-600 font-bold">
                {progress.percentage.toFixed(0)}%
              </div>
            </div>
            
            {/* Progress bar */}
            <div className="w-full bg-indigo-200 rounded-full h-3 mb-2 overflow-hidden">
              <div 
                className="bg-indigo-600 h-3 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress.percentage}%` }}
              />
            </div>
            
            {/* Time info */}
            <div className="flex justify-between text-xs text-indigo-600">
              <span>Elapsed: {formatTime(elapsedTime)}</span>
              <span>{formatEta(progress.etaSeconds)}</span>
            </div>
            
            {/* Current model indicator */}
            {progress.currentModel && (
              <div className="mt-2 text-xs text-indigo-700">
                <span className="font-medium">Current model:</span> {progress.currentModel}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results */}
      {benchmarkResult && (
        <div className="space-y-6">
          {/* Summary Info */}
          <div className="flex flex-wrap gap-4 text-sm text-gray-600 bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center gap-2">
              <span className="font-medium">Mode:</span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded uppercase text-xs font-bold">
                {getModeIcon(benchmarkResult.mode)}
                {benchmarkResult.mode}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              <span>Notes: <strong>{benchmarkResult.notes_processed}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4" />
              <span>Total Entities: <strong>{benchmarkResult.total_entities}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>Time: <strong>{benchmarkResult.total_processing_time_ms}ms</strong></span>
            </div>
          </div>

          {/* Results Table */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Results {benchmarkResult.results.length > 1 && `(${benchmarkResult.results.length} models)`}
            </h3>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-gray-100">
                    <th className="px-4 py-3 text-left font-medium text-gray-700">Model</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">F1-Harmonic</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">F1-Arithmetic</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">Precision</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">Recall</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">TP</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">FP</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">FN</th>
                    <th className="px-4 py-3 text-right font-medium text-gray-700">Time</th>
                    <th className="px-4 py-3 text-center font-medium text-gray-700"></th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkResult.results.map((result, idx) => (
                    <React.Fragment key={result.model_id}>
                      <tr className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-4 py-3 font-medium text-gray-900">
                          {result.model_id}
                        </td>
                        <td className={`px-4 py-3 text-right font-bold ${getScoreColor(result.f1_harmonic)}`}>
                          {formatPercentage(result.f1_harmonic)}
                        </td>
                        <td className={`px-4 py-3 text-right ${getScoreColor(result.f1_arithmetic)}`}>
                          {formatPercentage(result.f1_arithmetic)}
                        </td>
                        <td className={`px-4 py-3 text-right ${getScoreColor(result.precision)}`}>
                          {formatPercentage(result.precision)}
                        </td>
                        <td className={`px-4 py-3 text-right ${getScoreColor(result.recall)}`}>
                          {formatPercentage(result.recall)}
                        </td>
                        <td className="px-4 py-3 text-right text-green-600">{result.total_tp}</td>
                        <td className="px-4 py-3 text-right text-red-600">{result.total_fp}</td>
                        <td className="px-4 py-3 text-right text-orange-600">{result.total_fn}</td>
                        <td className="px-4 py-3 text-right text-gray-500">
                          {result.processing_time_s.toFixed(2)}s
                        </td>
                        <td className="px-4 py-3 text-center">
                          <button
                            onClick={() => setExpandedModel(
                              expandedModel === result.model_id ? null : result.model_id
                            )}
                            className="text-gray-400 hover:text-gray-600"
                          >
                            {expandedModel === result.model_id ? (
                              <ChevronUp className="w-4 h-4" />
                            ) : (
                              <ChevronDown className="w-4 h-4" />
                            )}
                          </button>
                        </td>
                      </tr>
                      
                      {/* Expanded per-note metrics */}
                      {expandedModel === result.model_id && (
                        <tr>
                          <td colSpan={10} className="px-4 py-2 bg-gray-100">
                            <div className="text-xs">
                              <div className="font-medium text-gray-700 mb-2">Per-Note Breakdown:</div>
                              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                                {result.per_note_metrics.map((note) => (
                                  <div key={note.note_id} className="bg-white p-2 rounded border">
                                    <div className="font-mono text-gray-600">{note.note_id}</div>
                                    <div className={`font-bold ${getScoreColor(note.f1)}`}>
                                      F1: {formatPercentage(note.f1)}
                                    </div>
                                    <div className="text-gray-500">
                                      TP:{note.tp} FP:{note.fp} FN:{note.fn}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Sequential Contribution (Assembly mode only) */}
          {benchmarkResult.sequential_contribution && benchmarkResult.sequential_contribution.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                🎯 Sequential Contribution Analysis (Non-Redundant)
              </h3>
              <p className="text-sm text-gray-500 mb-4">
                Shows how much NEW recall each model adds that previous models didn't find.
              </p>
              
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="px-4 py-3 text-left font-medium text-gray-700">Model</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">+ Recall</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Cumulative</th>
                      <th className="px-4 py-3 text-left font-medium text-gray-700">Progress</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResult.sequential_contribution.map((contrib, idx) => (
                      <tr key={contrib.model_id} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-4 py-3 font-medium text-gray-900">{contrib.model_id}</td>
                        <td className="px-4 py-3 text-right text-green-600 font-medium">
                          +{formatPercentage(contrib.incremental_recall)}
                        </td>
                        <td className="px-4 py-3 text-right font-bold text-indigo-600">
                          {formatPercentage(contrib.cumulative_recall)}
                        </td>
                        <td className="px-4 py-3">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-indigo-600 h-2 rounded-full transition-all"
                              style={{ width: `${contrib.cumulative_recall * 100}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!benchmarkResult && !isLoading && !error && (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <TrendingUp className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Benchmark Results Yet</h3>
          <p className="text-gray-500 max-w-md mx-auto">
            Select a benchmark mode and click "Run Benchmark" to evaluate NER model performance
            using the ground truth data.
          </p>
        </div>
      )}
    </div>
  );
};
