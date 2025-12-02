"""
Benchmark Router

Handles NER and RAG evaluation endpoints for debugging.
Supports modes: all, assembly, single (like diagnose_NER.py CLI)
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.schemas import (
    NERBenchmarkResponse,
    NERModelInfo,
    RAGBenchmarkResponse,
    BenchmarkMode
)
from ..services.benchmark_service import BenchmarkService

router = APIRouter()

benchmark_service = BenchmarkService()

# Thread pool for running CPU-bound benchmark in background
executor = ThreadPoolExecutor(max_workers=2)


@router.get("/benchmark/models", response_model=List[NERModelInfo])
async def get_available_models():
    """
    Get list of available NER models for benchmarking.
    """
    try:
        return benchmark_service.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/ner/stream")
async def run_ner_benchmark_stream(
    mode: str = Query(default="all"),
    model_id: Optional[str] = Query(default=None),
    iou_threshold: float = Query(default=0.25, ge=0.0, le=1.0)
):
    """
    Run NER benchmark with real-time progress updates via Server-Sent Events.
    
    Returns progress updates as SSE events, with the final result as the last event.
    """
    import queue
    import threading
    
    # Queue to pass events from sync generator to async generator
    event_queue = queue.Queue()
    
    def run_benchmark_in_thread():
        """Run benchmark in thread and push events to queue."""
        try:
            for update in benchmark_service.run_benchmark_stream(
                mode=mode,
                model_id=model_id,
                iou_threshold=iou_threshold
            ):
                event_queue.put(update)
            event_queue.put(None)  # Signal completion
        except Exception as e:
            event_queue.put({"type": "error", "message": str(e)})
            event_queue.put(None)
    
    async def generate():
        # Start benchmark in background thread
        thread = threading.Thread(target=run_benchmark_in_thread)
        thread.start()
        
        try:
            while True:
                # Poll queue with small timeout to allow async operations
                try:
                    update = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: event_queue.get(timeout=0.1)
                    )
                    
                    if update is None:
                        break
                    
                    yield f"data: {json.dumps(update)}\n\n"
                    
                except queue.Empty:
                    # Keep connection alive
                    continue
        finally:
            thread.join(timeout=1.0)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/benchmark/ner", response_model=NERBenchmarkResponse)
async def run_ner_benchmark(
    mode: str = Query(
        default="all",
        description="Benchmark mode: 'all' (each model individually), 'assembly' (all models combined), 'single' (specific model)"
    ),
    model_id: Optional[str] = Query(
        default=None,
        description="Model ID for 'single' mode (e.g., 'OntologyExact', 'SBert')"
    ),
    iou_threshold: float = Query(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="IoU threshold for matching (0.0 - 1.0)"
    )
):
    """
    Run NER benchmark with specified mode (non-streaming version).
    """
    try:
        if mode not in ["all", "assembly", "single"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid mode: {mode}. Use 'all', 'assembly', or 'single'"
            )
        
        if mode == "single" and not model_id:
            raise HTTPException(
                status_code=400,
                detail="model_id is required for 'single' mode"
            )
        
        result = benchmark_service.run_benchmark(
            mode=mode,
            model_id=model_id,
            iou_threshold=iou_threshold
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/rag", response_model=RAGBenchmarkResponse)
async def get_rag_benchmark_status():
    """
    Get RAG benchmark status.
    """
    return RAGBenchmarkResponse(
        status="not_implemented",
        message="RAG benchmarking is not yet available. Coming soon!",
        metrics=None
    )
