# 🏥 Medical Entity RAG System

> **A two-stage pipeline for high-recall medical entity extraction and knowledge enrichment**

This system combines state-of-the-art Named Entity Recognition (NER) with Retrieval-Augmented Generation (RAG) to extract and contextualize medical entities from clinical notes using SNOMED-CT ontology.

---

## 📋 Quick Start

### Prerequisites

- **Python:** 3.12 or higher
- **Node.js:** 18 or higher
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 10GB for models and data

### Installation

1. **Clone the repository** (if not already done)
   ```bash
   git clone <repository-url>
   cd RAG_ontologia
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows (bash)
   source .venv/Scripts/activate
   
   # Windows (cmd)
   .venv\Scripts\activate.bat
   
   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Install Python dependencies**
   ```bash
   # Install backend dependencies
   pip install -r requirements.txt
   
   # Install API dependencies
   pip install -r ofarres/api/requirements.txt
   ```

5. **Install Frontend dependencies**
   ```bash
   cd ofarres/frontend
   npm install
   cd ../..
   ```

---

## 🚀 Running the Application

### Complete Setup (All Components)

Open **3 separate terminals** and run the following commands:

#### Terminal 1: API Server
```bash
cd RAG_ontologia/ofarres
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Terminal 2: Frontend Development Server
```bash
cd RAG_ontologia/ofarres/frontend
npm run dev
```

#### Terminal 3: Backend NER Pipeline (Optional - for testing)
```bash
cd RAG_ontologia/ofarres/backend
python src/NER/A_pipeline_orchestrator.py
```

---

## 🌐 Access Points

Once running, you can access:

- **Frontend UI:** http://localhost:3000
- **API Documentation:** http://localhost:8000/api/docs
- **API ReDoc:** http://localhost:8000/api/redoc
- **API Root:** http://localhost:8000

---

## 📦 Quick Commands Reference

### API Server

```bash
# Development mode (auto-reload on code changes)
cd ofarres
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
cd ofarres
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Different port
cd ofarres
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend

```bash
# Development mode
cd ofarres/frontend
npm run dev

# Build for production
cd ofarres/frontend
npm run build

# Preview production build
cd ofarres/frontend
npm run preview
```

### Backend NER Pipeline

```bash
# Run complete pipeline with dashboard
cd ofarres/backend
python src/NER/A_pipeline_orchestrator.py

# Run individual benchmark
cd ofarres/backend
python benchmarks/diagnose_NER.py all

# Run specific worker benchmark
cd ofarres/backend
python benchmarks/diagnose_NER.py OntologyExact --iou 0.25 -v
```

---

## 🏗️ Architecture Overview

### Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: NER (Named Entity Recognition)               │
│  ─────────────────────────────────────────────────────  │
│  • Extract medical entities from clinical notes         │
│  • Multi-worker approach (Ontology, ML, Acronyms)      │
│  • 5-step post-processing pipeline                     │
│  • Achieves 100% Recall on test dataset                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: RAG (Retrieval-Augmented Generation)         │
│  ─────────────────────────────────────────────────────  │
│  • Retrieval: Link entities to SNOMED-CT concepts      │
│  • Augmentation: Enrich with medical knowledge         │
│  • Generation: Format for consumption                  │
└─────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 📁 Project Structure

```
ofarres/
├── api/                      # FastAPI REST API
│   ├── main.py              # API entry point
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   └── models/              # Data schemas
│
├── backend/                  # NER & RAG processing
│   ├── src/
│   │   ├── NER/             # Stage 1: Entity extraction
│   │   │   ├── A_pipeline_orchestrator.py  # Main orchestrator
│   │   │   ├── ontology_ner.py             # Dictionary-based NER
│   │   │   ├── spacy_ner.py                # ML-based NER
│   │   │   ├── acronym_ner.py              # Acronym specialist
│   │   │   └── postprocessor/              # 5-step pipeline
│   │   │       ├── 01_gather_assembly.py
│   │   │       ├── 02_assign_ranks.py
│   │   │       ├── 03_safe_deduplication.py
│   │   │       ├── 04_linguistic_filter.py
│   │   │       └── 05_semantic_judge.py
│   │   │
│   │   ├── RAG/             # Stage 2: Knowledge enrichment
│   │   │   ├── 01_Retrieval/
│   │   │   ├── 02_Augmentation/
│   │   │   └── 03_Generation/
│   │   │
│   │   └── utils/           # Helper functions
│   │
│   ├── config/              # Configuration files
│   ├── data/                # Input/output data
│   ├── ontology/            # SNOMED-CT ontology
│   └── benchmarks/          # Evaluation scripts
│
├── frontend/                 # React UI
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── modules/         # Feature modules
│   │   ├── routes/          # Page routes
│   │   └── services/        # API clients
│   └── package.json
│
├── ARCHITECTURE.md           # Detailed architecture docs
└── README.md                 # This file
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `ofarres/` directory:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: OpenAI API key (for future RAG generation)
OPENAI_API_KEY=your_api_key_here

# Model Paths
ONTOLOGY_PATH=ontology/multilingual_ontology.json
FAISS_INDEX_PATH=assets/ontology/ontology_rag.index
NER_REGISTRY_PATH=config/ner_registry.json
```

### NER Workers Configuration

Edit `ofarres/backend/config/ner_registry.json` to configure NER workers:

```json
{
  "OntologyExact": {
    "module": "src.NER.ontology_ner",
    "class": "OntologyNER",
    "ontology_path": "ontology/multilingual_ontology.json",
    "min_length": 1
  },
  "SBert": {
    "module": "src.NER.spacy_ner",
    "class": "ScispaCyNER",
    "model_name": "en_core_sci_scibert"
  },
  "Acronyms": {
    "module": "src.NER.acronym_ner",
    "class": "AcronymNER",
    "ontology_path": "ontology/multilingual_ontology.json",
    "max_acronym_len": 8
  }
}
```

---

## 🔬 Testing & Benchmarks

### Run NER Pipeline Benchmark

```bash
cd ofarres/backend
python src/NER/A_pipeline_orchestrator.py
```

**Expected Output:**
```
====================================================================================================
 PIPELINE PERFORMANCE DASHBOARD (RAG-Friendly Metrics)
====================================================================================================
Step Name                           | Entities   | Recall     | Precision  | F1
----------------------------------------------------------------------------------------------------
01_gather_assembly                  | 394        | 100.00%    | 24.62%     | 0.3951
02_assign_ranks                     | 394        | 100.00%    | 24.62%     | 0.3951
03_safe_deduplication               | 389        | 100.00%    | 24.94%     | 0.3992
04_linguistic_filter                | 361        | 100.00%    | 26.87%     | 0.4236
05_semantic_judge                   | 360        | 100.00%    | 26.94%     | 0.4245
====================================================================================================
```

### Run Individual Worker Benchmarks

```bash
cd ofarres/backend

# Test all workers
python benchmarks/diagnose_NER.py all

# Test specific worker with verbose output
python benchmarks/diagnose_NER.py OntologyExact --iou 0.25 -v

# Test assembly (all workers combined)
python benchmarks/diagnose_NER.py assembly -v
```

### Cross-Validation

```bash
cd ofarres/backend
python benchmarks/cross_validation.py
```

---

## 📊 API Endpoints

### Health Check
```http
GET /api/v1/health
```

### Notes Management
```http
# Get all notes
GET /api/v1/notes

# Get specific note with entities
GET /api/v1/notes/{note_id}

# Analyze new text
POST /api/v1/notes/analyze
Content-Type: application/json

{
  "text": "Patient presents with acute ischemic stroke..."
}
```

### Entity Extraction
```http
# Get entities for a note
GET /api/v1/entities/{note_id}
```

### Benchmarks
```http
# Run benchmark
POST /api/v1/benchmark/run
```

**Interactive API Documentation:** http://localhost:8000/api/docs

---

## 🛠️ Development

### Hot Reload

Both the API and frontend support hot reload during development:

- **API:** Uvicorn with `--reload` flag watches for Python file changes
- **Frontend:** Vite dev server watches for TypeScript/React changes

### Adding New NER Workers

1. Create new worker class in `ofarres/backend/src/NER/`
2. Implement `extract_entities(text: str) -> List[Dict]` method
3. Register in `config/ner_registry.json`
4. Test with `benchmarks/diagnose_NER.py`

### Adding API Endpoints

1. Create router in `ofarres/api/routers/`
2. Create service in `ofarres/api/services/`
3. Define schemas in `ofarres/api/models/schemas.py`
4. Register router in `ofarres/api/main.py`

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000 (API)
netstat -ano | findstr :8000

# Find process using port 3000 (Frontend)
netstat -ano | findstr :3000

# Kill process by PID
taskkill /PID <PID> /F
```

### Python Module Not Found

```bash
# Ensure virtual environment is activated
source .venv/Scripts/activate  # Windows bash
.venv\Scripts\activate.bat     # Windows cmd

# Reinstall dependencies
pip install -r requirements.txt
pip install -r ofarres/api/requirements.txt
```

### Frontend Build Errors

```bash
cd ofarres/frontend

# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

### ScispaCy Model Not Found

```bash
# Download and install SciBERT model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
```

---

## 📈 Performance

### Current Benchmarks

- **NER Latency:** ~500ms per clinical note (500 words)
- **API Response Time:** ~100ms (cached) to ~600ms (cold)
- **Recall:** 100% on test dataset (97 ground truth entities)
- **Precision:** 26.94% (with high recall optimization)
- **F1 Score:** 0.4245

### Optimization Tips

1. **GPU Acceleration:** Use CUDA-enabled PyTorch for faster SciBERT inference
2. **Batch Processing:** Process multiple notes in parallel
3. **Caching:** Implement Redis for frequently accessed ontology data
4. **Index Optimization:** Use quantized FAISS indices for faster retrieval

---

## 🔒 Security

### API Security
- CORS configured for `http://localhost:3000`
- Input validation via Pydantic models
- No data persistence (stateless processing)

### Production Recommendations
- Add API key authentication
- Implement rate limiting
- Enable HTTPS/TLS
- Set up proper CORS origins
- Add request logging and monitoring

---

## 📚 Documentation

- **Architecture Details:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Backend NER Pipeline:** [backend/README.md](./backend/README.md)
- **API Documentation:** http://localhost:8000/api/docs (when running)

---

## 🤝 Contributing

### Code Style
- **Python:** Follow PEP 8, use type hints
- **TypeScript:** Use ESLint configuration
- **Commits:** Conventional Commits format

### Testing
```bash
# Run Python tests (when available)
cd ofarres
pytest tests/

# Run frontend tests (when available)
cd ofarres/frontend
npm test
```

---

## 📝 System Requirements

### Minimum
- **OS:** Windows 10+, macOS 11+, Linux (Ubuntu 20.04+)
- **CPU:** 4 cores
- **RAM:** 16GB
- **Storage:** 10GB free space

### Recommended
- **CPU:** 8+ cores
- **RAM:** 32GB
- **GPU:** NVIDIA with 8GB+ VRAM (for faster ML inference)
- **Storage:** SSD with 20GB+ free space

---

## 📄 License

Internal Project - All Rights Reserved

---

## 👤 Author

**Oscar Farrés** - NLP Engineer
- NER Pipeline Development
- RAG Integration
- System Architecture

---

## 🙏 Acknowledgments

- **ScispaCy** - Biomedical NLP models
- **SNOMED International** - Medical ontology
- **Allen Institute for AI** - SciBERT model
- **Hugging Face** - Transformer models

---

## 📞 Support

For issues or questions:
1. Check [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed documentation
2. Review [Troubleshooting](#-troubleshooting) section
3. Contact the development team

---

**Last Updated:** December 2024
**Version:** 1.0.0
