/**
 * RAG Debug Section
 * 
 * Placeholder for RAG evaluation metrics.
 */

import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Clock, 
  AlertCircle, 
  RefreshCw,
  Construction
} from 'lucide-react';
import { getRAGBenchmarkStatus, type RAGBenchmarkResponse } from '../../services/benchmark';

export const RAGDebugSection: React.FC = () => {
  const [status, setStatus] = useState<RAGBenchmarkResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await getRAGBenchmarkStatus();
      setStatus(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to get RAG status');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">RAG Benchmark Configuration</h2>
        
        {/* Status */}
        {isLoading && (
          <div className="flex items-center gap-2 text-gray-600">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Loading status...</span>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Coming Soon Message */}
      <div className="text-center py-16">
        <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <Construction className="w-10 h-10 text-indigo-600" />
        </div>
        <h3 className="text-2xl font-bold text-gray-900 mb-3">RAG Evaluation Coming Soon</h3>
        <p className="text-gray-500 max-w-lg mx-auto mb-8">
          This section will allow you to evaluate Retrieval-Augmented Generation performance, 
          measuring accuracy, relevance, and latency of medical entity linking.
        </p>

        {/* Planned Features */}
        <div className="bg-gray-50 rounded-lg p-6 max-w-2xl mx-auto">
          <h4 className="font-semibold text-gray-900 mb-4 text-left">Planned Features</h4>
          <ul className="space-y-3 text-left text-gray-600">
            <li className="flex items-start gap-3">
              <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-xs font-bold text-indigo-600">1</span>
              </div>
              <span>
                <strong className="text-gray-900">Retrieval Accuracy</strong> - Measure how accurately 
                the system retrieves relevant SNOMED CT concepts for given text spans.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-xs font-bold text-indigo-600">2</span>
              </div>
              <span>
                <strong className="text-gray-900">Concept Linking Precision</strong> - Evaluate the 
                precision of linking entities to the correct SNOMED CT concept IDs.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-xs font-bold text-indigo-600">3</span>
              </div>
              <span>
                <strong className="text-gray-900">Response Latency</strong> - Track end-to-end 
                latency from entity detection to concept retrieval.
              </span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-xs font-bold text-indigo-600">4</span>
              </div>
              <span>
                <strong className="text-gray-900">Embedding Quality</strong> - Analyze vector 
                similarity scores and retrieval rankings.
              </span>
            </li>
          </ul>
        </div>

        {status && status.message && (
          <div className="mt-8 text-sm text-gray-500">
            <span className="inline-flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-full">
              <Clock className="w-4 h-4" />
              {status.message}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
