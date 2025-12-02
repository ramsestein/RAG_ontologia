import React, { useRef } from 'react';
import { UploadCloud, ArrowRight, Keyboard, Sparkles, Database } from 'lucide-react';
import { Button } from '../../components/ui/Button';

interface IngestionViewProps {
  inputText: string;
  setInputText: (text: string) => void;
  onAnalyze: () => void;
  onLoadFromBackend: () => void;
  isLoading?: boolean;
}

export const IngestionView: React.FC<IngestionViewProps> = ({ 
  inputText, 
  setInputText, 
  onAnalyze,
  onLoadFromBackend,
  isLoading = false
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => setInputText(event.target?.result as string || '');
    reader.readAsText(file);
  };

  return (
    <div className="flex-1 flex flex-col p-8 max-w-7xl mx-auto w-full gap-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary-600" />
            Clinical Notes Analysis
          </h1>
          <p className="text-slate-500 mt-1">Import clinical notes or load from backend to begin entity extraction.</p>
        </div>
        <div className="flex items-center gap-3">
           <Button 
             onClick={onLoadFromBackend}
             variant="secondary"
             disabled={isLoading}
             className="flex items-center gap-2"
           >
             <Database className="h-4 w-4" />
             {isLoading ? 'Loading...' : 'Load from Backend'}
           </Button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-8 min-h-0">
        
        {/* Left: Manual Input (2 cols) */}
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
              <Keyboard className="h-4 w-4 text-slate-400" />
              Manual Entry
            </div>
            <span className="text-xs text-slate-400 font-mono">
              {inputText.length} chars
            </span>
          </div>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            className="flex-1 w-full p-6 resize-none focus:outline-none text-slate-700 text-base font-mono leading-relaxed placeholder:text-slate-300"
            placeholder="Paste clinical text here (e.g., 'Patient ID: 123...')"
          />
          <div className="p-4 border-t border-slate-100 flex justify-end">
            <Button 
                onClick={onAnalyze} 
                disabled={!inputText.trim()}
                className="w-full sm:w-auto shadow-md shadow-primary-900/10"
            >
                Start Processing
                <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </div>
        </div>

        {/* Right: Upload (1 col) */}
        <div className="bg-slate-50 rounded-xl border-2 border-dashed border-slate-300 flex flex-col items-center justify-center p-8 text-center gap-4 hover:border-primary-400 hover:bg-slate-50/80 transition-all cursor-pointer group"
             onClick={() => fileInputRef.current?.click()}>
          
          <div className="h-16 w-16 bg-white rounded-full shadow-sm flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
             <UploadCloud className="h-8 w-8 text-slate-400 group-hover:text-primary-500 transition-colors" />
          </div>
          
          <div>
            <h3 className="font-semibold text-slate-900">Upload Documents</h3>
            <p className="text-sm text-slate-500 mt-1">
              Drag & drop .txt, .csv, or .md files
            </p>
          </div>

          <Button variant="secondary" size="sm" className="mt-2 pointer-events-none">
            Browse Files
          </Button>

          <input 
            type="file" 
            ref={fileInputRef} 
            className="hidden" 
            onChange={handleFileUpload} 
            accept=".txt,.md,.csv" 
          />
        </div>

      </div>
    </div>
  );
};