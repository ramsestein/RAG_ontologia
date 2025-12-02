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
  anonymousId: string; // e.g. "PATIENT-8821"
  content: string;
  timestamp: string;
  status: 'pending' | 'analyzed';
}