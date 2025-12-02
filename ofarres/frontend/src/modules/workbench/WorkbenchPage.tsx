import React, { useState } from 'react';
import { analyzeNote, segmentClinicalNotes } from '../../services/medical';
import { Entity, Note } from '../../types';
import { IngestionView } from './IngestionView';
import { LoadingView } from './LoadingView';
import { DashboardView } from './DashboardView';

const SAMPLE_BULK_TEXT = `Patient ID: 8821
Reason: Follow-up
Subjective: 65M w/ hx of CAD and Type 2 Diabetes. Complains of mild chest tightness upon exertion.
Assessment: Stable Angina.
Plan: Continue Metformin 500mg and Aspirin.

---

Patient ID: 9940
Reason: Emergency
History: Patient fell from standing height. Pain in left wrist.
Radiology: X-ray confirms distal radius fracture.
Plan: Splint application and ortho referral.

---

Patient ID: 1023
Reason: Routine
Notes: 45F for annual physical. BP 120/80. No complaints. 
Screening: Mammogram scheduled.`;

export const WorkbenchPage: React.FC = () => {
  // --- State ---
  const [viewMode, setViewMode] = useState<'ingest' | 'loading' | 'analysis'>('ingest');
  const [inputText, setInputText] = useState<string>('');
  
  // Loading State
  const [progress, setProgress] = useState(0);
  const [eta, setEta] = useState(5);

  // Analysis State
  const [notes, setNotes] = useState<Note[]>([]);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [entityMap, setEntityMap] = useState<Record<string, Entity[]>>({});

  // --- Logic ---
  const processAndSegment = async () => {
    if (!inputText.trim()) return;
    
    // 1. Transition to Loading
    setViewMode('loading');
    setProgress(0);
    setEta(5);

    // 2. Mock Progress Simulation (5 seconds)
    const duration = 5000;
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

    // 3. Business Logic (Segmentation)
    const generatedNotes = segmentClinicalNotes(inputText);
    setNotes(generatedNotes);
    setSelectedNoteId(generatedNotes[0]?.id || null);

    // 4. Batch Analysis (Mock)
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
        onLoadSample={() => setInputText(SAMPLE_BULK_TEXT)}
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