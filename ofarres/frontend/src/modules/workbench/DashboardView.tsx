import React from 'react';
import { ChevronLeft, CheckCircle2, Search, User, Sparkles, AlertCircle } from 'lucide-react';
import { Note, Entity } from '../../types';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { ENTITY_COLORS } from '../../config/constants';

interface DashboardViewProps {
  notes: Note[];
  selectedNoteId: string | null;
  onSelectNote: (id: string) => void;
  entityMap: Record<string, Entity[]>;
  onBack: () => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  notes,
  selectedNoteId,
  onSelectNote,
  entityMap,
  onBack
}) => {
  const activeNote = notes.find(n => n.id === selectedNoteId);
  const activeEntities = activeNote ? (entityMap[activeNote.id] || []) : [];

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col">
      {/* Dashboard Toolbar */}
      <header className="h-14 bg-white border-b border-slate-200 px-4 flex items-center justify-between shrink-0 shadow-sm z-10">
         <div className="flex items-center gap-4">
            <button 
              onClick={onBack}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors text-sm font-medium"
            >
              <ChevronLeft className="h-4 w-4" />
              New Analysis
            </button>
            <div className="h-6 w-px bg-slate-200" />
            <h2 className="font-semibold text-slate-800 text-sm flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-500" />
              Analysis Complete
            </h2>
            <Badge className="bg-slate-100 text-slate-600">{notes.length} Records</Badge>
         </div>
      </header>

      <div className="flex-1 flex min-h-0">
        
        {/* COL 1: Patient List */}
        <aside className="w-72 bg-white border-r border-slate-200 flex flex-col z-0">
          <div className="p-3 border-b border-slate-100 bg-slate-50/50">
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
              <input 
                placeholder="Search Patient ID..." 
                className="w-full bg-white border border-slate-200 rounded-md py-1.5 pl-9 pr-3 text-xs focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-all"
              />
            </div>
          </div>
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {notes.map(note => {
               const entityCount = entityMap[note.id]?.length || 0;
               const isSelected = selectedNoteId === note.id;
               
               return (
                <button
                  key={note.id}
                  onClick={() => onSelectNote(note.id)}
                  className={`w-full text-left p-4 border-b border-slate-100 transition-all duration-200 relative group ${
                    isSelected ? 'bg-primary-50/40' : 'hover:bg-slate-50'
                  }`}
                >
                  {isSelected && <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary-500" />}
                  
                  <div className="flex justify-between items-center mb-1.5">
                    <span className={`font-mono font-bold text-sm ${isSelected ? 'text-primary-700' : 'text-slate-700'}`}>
                      {note.anonymousId}
                    </span>
                    <span className="text-[10px] text-slate-400">{note.timestamp}</span>
                  </div>
                  
                  <div className="flex items-center gap-2 mb-2">
                    <Badge colorClass={entityCount > 0 ? "bg-white border border-slate-200 text-slate-600 shadow-sm" : "bg-slate-100 text-slate-400"}>
                      {entityCount} Entities
                    </Badge>
                  </div>
                  
                  <p className="text-xs text-slate-400 line-clamp-2 font-mono leading-relaxed opacity-80 group-hover:opacity-100">
                    {note.content}
                  </p>
                </button>
               );
            })}
          </div>
        </aside>

        {/* COL 2: Note Content */}
        <main className="flex-1 bg-white flex flex-col min-w-0">
          {activeNote ? (
            <div className="flex-1 flex flex-col h-full">
              <div className="p-6 border-b border-slate-100 flex justify-between items-start bg-slate-50/30">
                <div>
                   <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                     <div className="p-1.5 bg-white border border-slate-200 rounded shadow-sm">
                        <User className="h-5 w-5 text-slate-500" />
                     </div>
                     {activeNote.anonymousId}
                   </h3>
                   <p className="text-sm text-slate-500 mt-2 ml-1">
                     Raw Clinical Text Segment
                   </p>
                </div>
                <div className="flex gap-2">
                  <Button variant="ghost" size="sm" className="bg-white border border-slate-200 shadow-sm text-slate-600">
                    Export JSON
                  </Button>
                </div>
              </div>
              
              <div className="flex-1 p-8 overflow-y-auto bg-slate-50/20">
                <div className="max-w-3xl mx-auto bg-white border border-slate-200 shadow-sm p-8 rounded-xl min-h-[500px] relative">
                   <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-slate-200 via-slate-100 to-slate-200"></div>
                   <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-slate-700">
                     {activeNote.content}
                   </pre>
                </div>
              </div>
            </div>
          ) : (
             <div className="flex-1 flex items-center justify-center flex-col gap-3 text-slate-300">
               <User className="h-12 w-12 opacity-20" />
               <p>Select a patient record to view details</p>
             </div>
          )}
        </main>

        {/* COL 3: Entity Inspector */}
        <aside className="w-96 bg-slate-50 border-l border-slate-200 flex flex-col overflow-hidden shrink-0">
          <div className="p-4 border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-10">
            <h3 className="font-bold text-slate-800 text-sm flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary-600" />
              Extracted Entities
            </h3>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
             {activeEntities.length === 0 ? (
               <div className="text-center py-16 px-6 border-2 border-dashed border-slate-200 rounded-xl mt-4 mx-2">
                 <AlertCircle className="h-8 w-8 text-slate-300 mx-auto mb-3" />
                 <p className="text-sm text-slate-500 font-medium">No entities found</p>
                 <p className="text-xs text-slate-400 mt-1">Try adjusting the text or checking the input quality.</p>
               </div>
             ) : (
               activeEntities.map((entity, idx) => (
                 <div key={idx} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm hover:shadow-md hover:border-primary-200 transition-all group animate-in slide-in-from-right-4 duration-500" style={{animationDelay: `${idx * 50}ms`}}>
                    <div className="flex justify-between items-start gap-2 mb-2">
                      <span className="font-semibold text-slate-800 text-sm leading-snug group-hover:text-primary-700 transition-colors">
                        {entity.text}
                      </span>
                      <Badge colorClass={`${ENTITY_COLORS[entity.type]} text-[10px] px-2 py-0.5 whitespace-nowrap`}>
                        {entity.type}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-[10px] text-slate-500 mt-3 pt-3 border-t border-slate-50">
                      <div>
                        <span className="block text-slate-400 uppercase tracking-wider text-[9px] font-bold mb-0.5">SNOMED CT</span>
                        <span className="font-mono bg-slate-50 px-1.5 py-0.5 rounded text-slate-600">
                          {entity.snomedCode || "N/A"}
                        </span>
                      </div>
                      <div className="text-right">
                         <span className="block text-slate-400 uppercase tracking-wider text-[9px] font-bold mb-0.5">Confidence</span>
                         <span className={`inline-block px-1.5 py-0.5 rounded ${entity.confidence > 0.9 ? 'bg-green-50 text-green-700 font-bold' : 'bg-yellow-50 text-yellow-700'}`}>
                           {(entity.confidence * 100).toFixed(0)}%
                         </span>
                      </div>
                    </div>
                 </div>
               ))
             )}
          </div>
        </aside>

      </div>
    </div>
  );
};