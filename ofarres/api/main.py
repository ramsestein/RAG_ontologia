"""
Medical Entity RAG API - FastAPI Application

RESTful API for clinical note analysis and entity extraction.
Following SOLID principles and clean architecture.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import notes, entities, health

# Create FastAPI application
app = FastAPI(
    title="Medical Entity RAG API",
    description="RESTful API for clinical note analysis and SNOMED CT entity extraction",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(notes.router, prefix="/api/v1", tags=["Notes"])
app.include_router(entities.router, prefix="/api/v1", tags=["Entities"])


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "Medical Entity RAG API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }
