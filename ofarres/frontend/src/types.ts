export enum EntityType {
  DISORDER = 'Disorder',
  ANATOMY = 'Anatomy',
  PROCEDURE = 'Procedure',
  MEDICATION = 'Medication',
  OBSERVATION = 'Observation'
}

export interface Entity {
  id: string;
  text: string;
  type: EntityType;
  start: number;
  end: number;
  snomedCode?: string;
  confidence: number;
}

export interface AnalysisResponse {
  entities: Entity[];
  processingTimeMs: number;
  modelVersion: string;
}

export interface SnomedDetail {
  code: string;
  preferredTerm: string;
  description: string;
  parents: string[];
}

export interface Note {
  id: string;
  anonymousId: string;
  content: string;
  timestamp: string;
  status: 'pending' | 'analyzed';
}

/**
 * Backend note format - matches API response.
 */
export interface BackendNote {
  note_id: string;
  text: string;
}

/**
 * Backend entity format - matches API response.
 */
export interface BackendEntity {
  id: string;
  text: string;
  type: string;
  start: number;
  end: number;
  snomed_code?: string;
  confidence: number;
}