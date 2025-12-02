/**
 * Documentation content configuration.
 * Following Open/Closed Principle: Add new docs without modifying components.
 */

export interface DocPage {
  id: string;
  title: string;
  content: string;
  tocItems: TocItem[];
}

export interface TocItem {
  id: string;
  label: string;
  level: number;
  children?: TocItem[];
}

export interface DocNavSection {
  id: string;
  title: string;
  items: DocNavItem[];
}

export interface DocNavItem {
  id: string;
  label: string;
}

/**
 * Navigation sections for the sidebar.
 */
export const NAV_SECTIONS: DocNavSection[] = [
  {
    id: 'getting-started',
    title: 'Getting Started',
    items: [
      { id: 'installation', label: 'Installation' },
      { id: 'configuration', label: 'Configuration' },
      { id: 'authentication', label: 'Authentication' },
    ],
  },
  {
    id: 'core-concepts',
    title: 'Core Concepts',
    items: [
      { id: 'entity-extraction', label: 'Entity Extraction' },
      { id: 'rag-pipeline', label: 'RAG Pipeline' },
      { id: 'vector-search', label: 'Vector Search' },
    ],
  },
];

/**
 * Documentation pages content.
 */
export const DOC_PAGES: Record<string, DocPage> = {
  installation: {
    id: 'installation',
    title: 'Installation',
    content: `
# Installation

## Overview
The Medical Entity RAG Workbench uses a **Retrieval-Augmented Generation** approach to ground Large Language Model outputs in verified clinical terminologies like SNOMED CT and ICD-10.

## System Requirements

Before installing, ensure your system meets these requirements:

- **Node.js** 18.x or higher
- **Python** 3.10 or higher
- **Docker** (optional, for containerized deployment)

## Quick Start

### 1. Clone the Repository

\`\`\`bash
git clone https://github.com/your-org/medical-rag-workbench.git
cd medical-rag-workbench
\`\`\`

### 2. Install Dependencies

\`\`\`bash
npm install
pip install -r requirements.txt
\`\`\`

### 3. Start the Application

\`\`\`bash
npm run dev
\`\`\`

## Next Steps
After installation, proceed to [Configuration](#configuration) to set up your environment.
`,
    tocItems: [
      { id: 'overview', label: 'Overview', level: 0 },
      { id: 'system-requirements', label: 'System Requirements', level: 0 },
      {
        id: 'quick-start',
        label: 'Quick Start',
        level: 0,
        children: [
          { id: 'clone-repo', label: 'Clone the Repository', level: 1 },
          { id: 'install-deps', label: 'Install Dependencies', level: 1 },
          { id: 'start-app', label: 'Start the Application', level: 1 },
        ],
      },
      { id: 'next-steps', label: 'Next Steps', level: 0 },
    ],
  },
  configuration: {
    id: 'configuration',
    title: 'Configuration',
    content: `
# Configuration

## Environment Variables

Create a \`.env\` file in the project root with the following variables:

\`\`\`bash
# API Configuration
API_BASE_URL=http://localhost:8000
API_TIMEOUT=30000

# Model Settings
MODEL_NAME=sapbert-medical
EMBEDDING_DIM=768

# Database
VECTOR_DB_PATH=./data/vectors
\`\`\`

## Configuration Options

### API Settings

| Variable | Description | Default |
|----------|-------------|---------|
| API_BASE_URL | Backend API endpoint | http://localhost:8000 |
| API_TIMEOUT | Request timeout in ms | 30000 |

### Model Settings

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_NAME | NER model identifier | sapbert-medical |
| EMBEDDING_DIM | Vector embedding dimension | 768 |

## Advanced Configuration

For production deployments, consider:

1. **SSL/TLS Configuration** - Enable HTTPS for secure communication
2. **Load Balancing** - Configure multiple backend instances
3. **Caching** - Set up Redis for response caching
`,
    tocItems: [
      { id: 'environment-variables', label: 'Environment Variables', level: 0 },
      { id: 'configuration-options', label: 'Configuration Options', level: 0 },
      { id: 'advanced-configuration', label: 'Advanced Configuration', level: 0 },
    ],
  },
  authentication: {
    id: 'authentication',
    title: 'Authentication',
    content: `
# Authentication

## Overview

The workbench supports multiple authentication methods for secure API access.

## API Key Authentication

The simplest method for development and testing:

\`\`\`javascript
const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY'
  }
});
\`\`\`

## OAuth 2.0

For production environments, we recommend OAuth 2.0:

### Configuration

\`\`\`javascript
const authConfig = {
  clientId: 'your-client-id',
  authorizationEndpoint: '/oauth/authorize',
  tokenEndpoint: '/oauth/token',
  scopes: ['read', 'write', 'analyze']
};
\`\`\`

## Role-Based Access Control

The system supports granular permissions:

- **Reader** - View results only
- **Analyst** - Run analyses and view results
- **Admin** - Full system access

## Security Best Practices

1. Never expose API keys in client-side code
2. Use environment variables for sensitive data
3. Rotate credentials regularly
4. Enable audit logging
`,
    tocItems: [
      { id: 'overview', label: 'Overview', level: 0 },
      { id: 'api-key-authentication', label: 'API Key Authentication', level: 0 },
      { id: 'oauth-2', label: 'OAuth 2.0', level: 0 },
      { id: 'role-based-access', label: 'Role-Based Access Control', level: 0 },
      { id: 'security-best-practices', label: 'Security Best Practices', level: 0 },
    ],
  },
  'entity-extraction': {
    id: 'entity-extraction',
    title: 'Entity Extraction',
    content: `
# Entity Extraction

## How It Works

The entity extraction pipeline uses a fine-tuned transformer model to identify medical entities in clinical text.

## Supported Entity Types

| Type | Description | Example |
|------|-------------|---------|
| DISORDER | Medical conditions | Diabetes, Hypertension |
| MEDICATION | Drugs and treatments | Metformin, Aspirin |
| PROCEDURE | Medical procedures | MRI, Blood test |
| ANATOMY | Body parts | Heart, Liver |
| OBSERVATION | Clinical findings | Elevated BP, Fever |

## API Usage

\`\`\`javascript
const response = await api.post('/analyze', {
  text: 'Patient presents with Type 2 Diabetes and takes Metformin daily.'
});

console.log(response.data.entities);
// [
//   { text: 'Type 2 Diabetes', type: 'DISORDER', confidence: 0.95 },
//   { text: 'Metformin', type: 'MEDICATION', confidence: 0.98 }
// ]
\`\`\`

## Confidence Scores

Each entity includes a confidence score (0-1):
- **> 0.9** - High confidence
- **0.7-0.9** - Medium confidence  
- **< 0.7** - Low confidence (review recommended)
`,
    tocItems: [
      { id: 'how-it-works', label: 'How It Works', level: 0 },
      { id: 'supported-entity-types', label: 'Supported Entity Types', level: 0 },
      { id: 'api-usage', label: 'API Usage', level: 0 },
      { id: 'confidence-scores', label: 'Confidence Scores', level: 0 },
    ],
  },
  'rag-pipeline': {
    id: 'rag-pipeline',
    title: 'RAG Pipeline',
    content: `
# RAG Pipeline

## Architecture

The Retrieval-Augmented Generation pipeline grounds LLM outputs in verified medical terminologies.

## Pipeline Stages

### 1. Text Preprocessing
Raw clinical text is normalized and tokenized for analysis.

### 2. Entity Recognition
The NER model identifies medical entities in the text.

### 3. Vector Search
Identified entities are matched against the SNOMED CT vector database.

### 4. Context Enrichment
Retrieved terminology provides context for accurate classification.

### 5. Output Generation
Final structured output with entity codes and confidence scores.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Average Latency | 45ms |
| Precision | 0.94 |
| Recall | 0.91 |
| F1 Score | 0.92 |

## Optimization Tips

- Batch multiple texts for better throughput
- Use caching for frequently analyzed terms
- Pre-warm the model for production use
`,
    tocItems: [
      { id: 'architecture', label: 'Architecture', level: 0 },
      {
        id: 'pipeline-stages',
        label: 'Pipeline Stages',
        level: 0,
        children: [
          { id: 'text-preprocessing', label: 'Text Preprocessing', level: 1 },
          { id: 'entity-recognition', label: 'Entity Recognition', level: 1 },
          { id: 'vector-search', label: 'Vector Search', level: 1 },
        ],
      },
      { id: 'performance-metrics', label: 'Performance Metrics', level: 0 },
      { id: 'optimization-tips', label: 'Optimization Tips', level: 0 },
    ],
  },
  'vector-search': {
    id: 'vector-search',
    title: 'Vector Search',
    content: `
# Vector Search

## Overview

The vector search system enables semantic matching of clinical concepts against the SNOMED CT terminology.

## How Vector Search Works

1. **Embedding Generation** - Text is converted to dense vector representations
2. **Similarity Search** - Vectors are compared using cosine similarity
3. **Ranking** - Results are ranked by relevance score

## Index Statistics

| Metric | Value |
|--------|-------|
| Total Concepts | 350,000+ |
| Embedding Dimension | 768 |
| Index Type | HNSW |
| Average Query Time | 5ms |

## Query Examples

\`\`\`javascript
// Search for similar concepts
const results = await api.post('/search', {
  query: 'chest pain',
  limit: 10
});

// Returns ranked SNOMED CT concepts
results.data.forEach(concept => {
  console.log(\`\${concept.term} (score: \${concept.score})\`);
});
\`\`\`

## Tuning Parameters

- **k** - Number of neighbors to retrieve
- **ef_search** - Controls accuracy vs speed tradeoff
- **threshold** - Minimum similarity score to include
`,
    tocItems: [
      { id: 'overview', label: 'Overview', level: 0 },
      { id: 'how-vector-search-works', label: 'How Vector Search Works', level: 0 },
      { id: 'index-statistics', label: 'Index Statistics', level: 0 },
      { id: 'query-examples', label: 'Query Examples', level: 0 },
      { id: 'tuning-parameters', label: 'Tuning Parameters', level: 0 },
    ],
  },
};

/**
 * Get all searchable items for the search functionality.
 */
export const getSearchableItems = (): Array<{ id: string; label: string; section: string }> => {
  const items: Array<{ id: string; label: string; section: string }> = [];
  
  NAV_SECTIONS.forEach(section => {
    section.items.forEach(item => {
      items.push({
        id: item.id,
        label: item.label,
        section: section.title,
      });
    });
  });
  
  return items;
};
