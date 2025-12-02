/**
 * Debug Page - Main Component
 * 
 * Provides debugging tools for NER and RAG evaluation.
 * Displays precision, recall, F1 scores and other metrics.
 */

import React, { useState } from 'react';
import { Bug, Cpu, Database, RefreshCw } from 'lucide-react';
import { NERDebugSection } from './NERDebugSection';
import { RAGDebugSection } from './RAGDebugSection';

type DebugTab = 'ner' | 'rag';

export const DebugPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<DebugTab>('ner');

  return (
    <div className="h-[calc(100vh-4rem)] overflow-y-auto bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-16">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <Bug className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-900">Debug Console</h1>
          </div>
          <p className="text-gray-600">
            Evaluate and benchmark NER and RAG model performance using ground truth data.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('ner')}
                className={`flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'ner'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Cpu className="w-4 h-4" />
                NER Evaluation
              </button>
              <button
                onClick={() => setActiveTab('rag')}
                className={`flex items-center gap-2 px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === 'rag'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Database className="w-4 h-4" />
                RAG Evaluation
              </button>
            </nav>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-sm">
          {activeTab === 'ner' && <NERDebugSection />}
          {activeTab === 'rag' && <RAGDebugSection />}
        </div>
      </div>
    </div>
  );
};
