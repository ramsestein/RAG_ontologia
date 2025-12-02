import React from 'react';
import { Sparkles, Timer } from 'lucide-react';

interface LoadingViewProps {
  progress: number;
  eta: number;
}

export const LoadingView: React.FC<LoadingViewProps> = ({ progress, eta }) => {
  return (
      <div className="min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center p-6 animate-in fade-in duration-500">
        <div className="w-full max-w-md bg-white rounded-2xl shadow-xl border border-slate-200 p-10 text-center space-y-8 relative overflow-hidden">
          
          {/* Background decoration */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent opacity-50"></div>

          <div className="relative inline-flex items-center justify-center">
            {/* Double Spinner Ring */}
            <div className="w-24 h-24 border-4 border-slate-100 rounded-full"></div>
            <div 
              className="w-24 h-24 border-4 border-primary-500 rounded-full absolute border-t-transparent animate-spin" 
              style={{ animationDuration: '1.2s' }}
            ></div>
            <div 
              className="w-16 h-16 border-4 border-slate-200 rounded-full absolute border-b-transparent animate-spin" 
              style={{ animationDuration: '2s', animationDirection: 'reverse' }}
            ></div>
            <Sparkles className="h-8 w-8 text-primary-600 absolute" />
          </div>

          <div className="space-y-3">
            <h2 className="text-2xl font-bold text-slate-900">Analysis in Progress</h2>
            <p className="text-slate-500 text-sm leading-relaxed">
              Our AI is currently segmenting notes, extracting entities, and querying the SNOMED CT vector database.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs font-bold text-slate-400 uppercase tracking-widest px-1">
              <span>Processing</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden shadow-inner">
              <div 
                className="bg-gradient-to-r from-primary-500 to-primary-600 h-full transition-all duration-300 ease-out rounded-full" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>

          <div className="inline-flex items-center gap-2 px-4 py-2 bg-slate-50 rounded-lg text-xs font-medium text-slate-600 border border-slate-200/60 shadow-sm">
            <Timer className="h-3.5 w-3.5 text-primary-500" />
            <span>Estimated completion: {eta}s</span>
          </div>

        </div>
      </div>
  );
};