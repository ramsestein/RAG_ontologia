import api from './api';
import type { AnalysisResponse, Entity, EntityType, SnomedDetail, Note, BackendNote } from '../types';

/**
 * Medical Service - API integration layer.
 * Following Single Responsibility Principle.
 */

/**
 * Fetches all clinical notes from the backend.
 */
export const fetchAllNotes = async (): Promise<BackendNote[]> => {
  const response = await api.get<BackendNote[]>('/notes');
  return response.data;
};

/**
 * Fetches a specific note with its entities from the backend.
 */
export const fetchNoteWithEntities = async (noteId: string): Promise<{
  note_id: string;
  text: string;
  entities: Entity[];
}> => {
  const response = await api.get(`/notes/${noteId}`);
  
  // Transform backend response to frontend format
  const entities: Entity[] = response.data.entities.map((e: any) => ({
    id: e.id,
    text: e.text,
    type: e.type as EntityType,
    start: e.start,
    end: e.end,
    snomedCode: e.snomed_code,
    confidence: e.confidence,
  }));
  
  return {
    note_id: response.data.note_id,
    text: response.data.text,
    entities,
  };
};

/**
 * Sends a clinical note to the backend for entity extraction and analysis.
 */
export const analyzeNote = async (text: string): Promise<AnalysisResponse> => {
  try {
    const response = await api.post('/notes/analyze', { text });
    
    // Transform backend response to frontend format
    const entities: Entity[] = response.data.entities.map((e: any) => ({
      id: e.id,
      text: e.text,
      type: e.type as EntityType,
      start: e.start,
      end: e.end,
      snomedCode: e.snomed_code,
      confidence: e.confidence,
    }));
    
    return {
      entities,
      processingTimeMs: response.data.processing_time_ms,
      modelVersion: response.data.model_version,
    };
  } catch (error) {
    // Fallback to mock if backend is unavailable
    console.warn('Backend unavailable, using mock analysis');
    return mockAnalyzeNote(text);
  }
};

/**
 * Mock analysis function for when backend is unavailable.
 */
const mockAnalyzeNote = async (text: string): Promise<AnalysisResponse> => {
  const entities: Entity[] = [];
  let idCounter = 1;
  const textLower = text.toLowerCase();

  const keywords: Record<string, { type: EntityType; code: string }> = {
    'hypertension': { type: 'Disorder' as EntityType, code: '38341003' },
    'diabetes': { type: 'Disorder' as EntityType, code: '73211009' },
    'stroke': { type: 'Disorder' as EntityType, code: '230690007' },
    'hemorrhage': { type: 'Disorder' as EntityType, code: '50960005' },
    'weakness': { type: 'Disorder' as EntityType, code: '13791008' },
    'headache': { type: 'Disorder' as EntityType, code: '25064002' },
    'ct': { type: 'Procedure' as EntityType, code: '77477000' },
    'mri': { type: 'Procedure' as EntityType, code: '113091000' },
    'thrombectomy': { type: 'Procedure' as EntityType, code: '433112001' },
  };

  for (const [keyword, { type, code }] of Object.entries(keywords)) {
    if (textLower.includes(keyword)) {
      entities.push({
        id: String(idCounter++),
        text: keyword,
        type,
        start: textLower.indexOf(keyword),
        end: textLower.indexOf(keyword) + keyword.length,
        snomedCode: code,
        confidence: 0.85 + Math.random() * 0.1,
      });
    }
  }

  return {
    entities,
    processingTimeMs: 50,
    modelVersion: 'v1.0.0-mock',
  };
};

/**
 * Retrieves detailed SNOMED CT information for a specific code.
 */
export const getSnomedDetails = async (code: string): Promise<SnomedDetail> => {
  try {
    const response = await api.get<any>(`/snomed/${code}`);
    return {
      code: response.data.code,
      preferredTerm: response.data.preferred_term,
      description: response.data.description,
      parents: response.data.parents,
    };
  } catch (error) {
    return {
      code,
      preferredTerm: "SNOMED Concept",
      description: "Clinical concept from SNOMED CT terminology",
      parents: ["Clinical finding"]
    };
  }
};

/**
 * Converts backend notes to frontend Note format.
 */
export const transformBackendNotes = (backendNotes: BackendNote[]): Note[] => {
  return backendNotes.map((note, index) => ({
    id: note.note_id,
    anonymousId: `NOTE-${note.note_id.padStart(4, '0')}`,
    content: note.text,
    timestamp: new Date().toLocaleTimeString(),
    status: 'analyzed' as const,
  }));
};

/**
 * Segments a bulk text string into individual notes.
 */
export const segmentClinicalNotes = (bulkText: string): Note[] => {
  const generateId = () => Math.random().toString(36).substring(2, 11);
  
  const rawSegments = bulkText
    .split(/(?:-{3,}|Patient ID:|Case:|Note ID:)/i)
    .map(s => s.trim())
    .filter(s => s.length > 10);

  if (rawSegments.length === 0) {
    return bulkText.trim() ? [{
      id: generateId(),
      anonymousId: `NOTE-${Math.floor(Math.random() * 9000) + 1000}`,
      content: bulkText,
      timestamp: new Date().toLocaleTimeString(),
      status: 'analyzed'
    }] : [];
  }

  return rawSegments.map((segment) => ({
    id: generateId(),
    anonymousId: `NOTE-${Math.floor(Math.random() * 9000) + 1000}`,
    content: segment,
    timestamp: new Date().toLocaleTimeString(),
    status: 'analyzed'
  }));
};