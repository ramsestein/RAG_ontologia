"""
Health Check Router

Provides endpoints for API health monitoring.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "service": "medical-entity-rag-api",
        "version": "1.0.0"
    }
