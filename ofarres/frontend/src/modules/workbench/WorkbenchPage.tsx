import React, { useState } from 'react';
import { 
  analyzeNote, 
  segmentClinicalNotes, 
  fetchAllNotes, 
  fetchNoteWithEntities,
  transformBackendNotes 
} from '../../services/medical';
import type { Entity, Note } from '../../types';
import { IngestionView } from './IngestionView';
import { LoadingView } from './LoadingView';
import { DashboardView } from './DashboardView';

export const WorkbenchPage: React.FC = () => {
  // --- State ---
  const [viewMode, setViewMode] = useState<'ingest' | 'loading' | 'analysis'>('ingest');
  const [inputText, setInputText] = useState<string>('');
  const [isLoadingBackend, setIsLoadingBackend] = useState(false);
  
  // Loading State
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState(5);

  // Analysis State
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [entityMap, setEntityMap] = useState<Record<string, Entity[]>>({});

  // --- Load Notes from Backend ---
  const loadFromBackend = async () => {
    setIsLoadingBackend(true);
    
    try {
      // Fetch all notes from backend
      const backendNotes = await fetchAllNotes();
      
      if (backendNotes.length === 0) {
        alert('No notes found in backend');
        setIsLoadingBackend(false);
        return;
      }

      // Transform to frontend format
      const frontendNotes = transformBackendNotes(backendNotes);
      setNotes(frontendNotes);
      setSelectedNoteId(frontendNotes[0]?.id || null);

      // Transition to loading view
      setViewMode('loading');
      setProgress(0);
      setEta(backendNotes.length);

      // Fetch entities for each note
      const newEntityMap: Record<string, Entity[]> = {};
      const totalNotes = backendNotes.length;
      
      for (let i = 0; i < backendNotes.length; i++) {
        const note = backendNotes[i];
        try {
          const result = await fetchNoteWithEntities(note.note_id);
          newEntityMap[note.note_id] = result.entities;
        } catch (error) {
          console.error(`Failed to fetch entities for note ${note.note_id}:`, error);
          newEntityMap[note.note_id] = [];
        }
        
        // Update progress
        setProgress(((i + 1) / totalNotes) * 100);
        setEta(Math.max(0, totalNotes - i - 1));
      }

      setEntityMap(newEntityMap);
      setViewMode('analysis');
    } catch (error) {
      console.error('Failed to load from backend:', error);
      alert('Failed to connect to backend. Make sure the API server is running on http://localhost:8000');
    } finally {
      setIsLoadingBackend(false);
    }
  };

  // --- Process Manual Input ---
  const processAndSegment = async () => {
    if (!inputText.trim()) return;
    
    // 1. Transition to Loading
    setViewMode('loading');
    setProgress(0);
    setEta(5);

    // 2. Progress Simulation
    const duration = 3000;
    const intervalTime = 50;
    const steps = duration / intervalTime;
    let currentStep = 0;

    await new Promise<void>((resolve) => {
      const timer = setInterval(() => {
        currentStep++;
        const newProgress = Math.min((currentStep / steps) * 100, 100);
        setProgress(newProgress);
        
        if (currentStep % (1000 / intervalTime) === 0) {
          setEta(prev => Math.max(0, prev - 1));
        }

        if (currentStep >= steps) {
          clearInterval(timer);
          resolve();
        }
      }, intervalTime);
    });

    // 3. Segment text into notes
    const generatedNotes = segmentClinicalNotes(inputText);
    setNotes(generatedNotes);
    setSelectedNoteId(generatedNotes[0]?.id || null);

    // 4. Analyze each note
    const newEntityMap: Record<string, Entity[]> = {};
    await Promise.all(generatedNotes.map(async (note) => {
      const result = await analyzeNote(note.content);
      newEntityMap[note.id] = result.entities;
    }));

    setEntityMap(newEntityMap);
    
    // 5. Show Dashboard
    setViewMode('analysis');
  };

  // --- Render Controller ---
  if (viewMode === 'ingest') {
    return (
      <IngestionView 
        inputText={inputText}
        setInputText={setInputText}
        onAnalyze={processAndSegment}
        onLoadFromBackend={loadFromBackend}
        isLoading={isLoadingBackend}
      />
    );
  }

  if (viewMode === 'loading') {
    return <LoadingView progress={progress} eta={eta} />;
  }

  return (
    <DashboardView 
      notes={notes}
      selectedNoteId={selectedNoteId}
      onSelectNote={setSelectedNoteId}
      entityMap={entityMap}
      onBack={() => setViewMode('ingest')}
    />
  );
};