import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Search, Book, Menu, X, ChevronRight } from 'lucide-react';

const MOCK_DOC_CONTENT = `
# System Architecture

## Overview
The Medical Entity RAG Workbench uses a **Retrieval-Augmented Generation** approach to ground Large Language Model outputs in verified clinical terminologies like SNOMED CT and ICD-10.

## Key Components

### 1. The Vector Store
We utilize a high-performance vector database to index over 300,000 clinical concepts. This allows for semantic search rather than just keyword matching.

### 2. The Inference Engine
The backend runs a fine-tuned transformer model optimized for Named Entity Recognition (NER) on clinical text.

## API Integration

To connect a new frontend client, ensure you authenticate via the \`Authorization\` header.

\`\`\`javascript
const response = await api.get('/health');
\`\`\`

## Data Privacy
All PII is redacted before processing in the cloud pipeline according to HIPAA Safe Harbor guidelines.
`;

const DocLink = ({ children, active = false }: { children?: React.ReactNode, active?: boolean }) => (
  <button 
    onClick={(e) => e.preventDefault()}
    className={`block w-full text-left pl-4 py-2 text-sm border-l-2 transition-all duration-200 ${
      active 
        ? 'text-primary-700 border-primary-600 font-bold bg-primary-50 rounded-r-md' 
        : 'text-slate-500 border-transparent hover:text-slate-900 hover:border-slate-300'
    }`}
  >
    {children}
  </button>
);

export const DocLayout: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <div className="flex flex-col lg:flex-row w-full min-h-screen bg-canvas">
      
      {/* Mobile Header for Menu */}
      <div className="lg:hidden bg-white border-b border-slate-200 p-4 sticky top-16 z-30 flex items-center justify-between shadow-sm">
        <span className="font-bold text-slate-700 flex items-center gap-2">
          <Book className="h-4 w-4 text-primary-600" />
          Documentation
        </span>
        <button 
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="p-2 text-slate-500 hover:bg-slate-100 rounded-md"
        >
          {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Sidebar Overlay (Mobile) */}
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/50 z-40 lg:hidden backdrop-blur-sm"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 w-72 bg-white border-r border-slate-200 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:h-[calc(100vh-4rem)] lg:sticky lg:top-16 lg:bg-transparent lg:border-none lg:shadow-none
        ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="h-full lg:h-auto lg:m-6 lg:bg-white lg:rounded-xl lg:shadow-sm lg:border lg:border-slate-200 overflow-hidden flex flex-col">
          <div className="p-5 border-b border-slate-100 bg-slate-50/50">
            <h2 className="font-bold text-slate-900 flex items-center gap-2 mb-4">
              <Book className="h-5 w-5 text-primary-600" />
              <span className="lg:hidden xl:inline">Doc Explorer</span>
            </h2>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input 
                type="text" 
                placeholder="Filter topics..." 
                className="w-full pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
              />
            </div>
          </div>

          <nav className="flex-1 overflow-y-auto p-5 space-y-8 bg-white">
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Getting Started</h3>
              <ul className="space-y-1">
                <li><DocLink active>Installation</DocLink></li>
                <li><DocLink>Configuration</DocLink></li>
                <li><DocLink>Authentication</DocLink></li>
              </ul>
            </div>
            <div>
              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 px-2">Core Concepts</h3>
              <ul className="space-y-1">
                <li><DocLink>Entity Extraction</DocLink></li>
                <li><DocLink>RAG Pipeline</DocLink></li>
                <li><DocLink>Vector Search</DocLink></li>
              </ul>
            </div>
          </nav>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 min-w-0 p-4 lg:p-6 xl:p-8">
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 md:p-10 lg:p-12 mx-auto max-w-4xl">
          <div className="mb-6 pb-6 border-b border-slate-100 lg:hidden">
             <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
               <span>Docs</span>
               <ChevronRight className="h-3 w-3" />
               <span>Getting Started</span>
             </div>
          </div>

          <article className="prose prose-slate prose-headings:font-bold prose-headings:text-slate-900 prose-p:text-slate-600 prose-a:text-primary-600 hover:prose-a:text-primary-500 max-w-none">
            <ReactMarkdown>{MOCK_DOC_CONTENT}</ReactMarkdown>
          </article>
        </div>
      </main>

      {/* Right Table of Contents (Desktop Only) */}
      <aside className="hidden xl:block w-64 shrink-0 h-[calc(100vh-4rem)] sticky top-16 pt-8 pr-8">
        <h5 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-primary-500"></span>
          On this page
        </h5>
        <ul className="space-y-3 text-sm border-l border-slate-200 ml-1">
          <li><button onClick={(e) => e.preventDefault()} className="block pl-4 text-slate-700 hover:text-primary-600 font-medium text-left transition-colors">Overview</button></li>
          <li><button onClick={(e) => e.preventDefault()} className="block pl-4 text-slate-500 hover:text-primary-600 text-left transition-colors">Key Components</button></li>
          <li>
            <ul className="space-y-2 mt-2">
              <li><button onClick={(e) => e.preventDefault()} className="block pl-8 text-slate-400 hover:text-primary-600 text-left transition-colors">The Vector Store</button></li>
              <li><button onClick={(e) => e.preventDefault()} className="block pl-8 text-slate-400 hover:text-primary-600 text-left transition-colors">The Inference Engine</button></li>
            </ul>
          </li>
          <li><button onClick={(e) => e.preventDefault()} className="block pl-4 text-slate-500 hover:text-primary-600 text-left transition-colors">API Integration</button></li>
          <li><button onClick={(e) => e.preventDefault()} className="block pl-4 text-slate-500 hover:text-primary-600 text-left transition-colors">Data Privacy</button></li>
        </ul>
      </aside>

    </div>
  );
};