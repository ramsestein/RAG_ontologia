// In a real Vite app, this would be import.meta.env.VITE_API_BASE_URL
// Defaulting to localhost for development
export const API_BASE_URL = 'http://localhost:8000/api/v1';

export const APP_NAME = 'MedRAG Workbench';

// Semantic medical colors (Disorders = Red)
// We keep Disorders red even if the UI is red, as it is standard.
export const ENTITY_COLORS: Record<string, string> = {
  Disorder: 'bg-red-100 text-red-800 border-red-200',
  Anatomy: 'bg-blue-100 text-blue-800 border-blue-200',
  Procedure: 'bg-green-100 text-green-800 border-green-200',
  Medication: 'bg-purple-100 text-purple-800 border-purple-200',
  Observation: 'bg-yellow-100 text-yellow-800 border-yellow-200',
};