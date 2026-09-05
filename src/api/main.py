from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import SearchRequest, SearchResponse, ChatRequest, ChatResponse
from src.api.services import RAGService
import os
import glob
import pandas as pd
import numpy as np

rag_service: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    print("[INFO] Starting up FastAPI Application & Initializing RAG Services...")
    rag_service = RAGService()
    yield
    print("[INFO] Shutting down FastAPI Application & Cleaning up RAG Services...")


app = FastAPI(
    title="TU Delft Academic RAG Engine API",
    description="Enterprise Async RAG Service for TU Delft Publications Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Streamlit / Frontend UI Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_service_ready():
    """Helper method to verify service readiness before processing requests."""
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Service initialization in progress. Please try again shortly."
        )


@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Service health and readiness check probe."""
    is_ready = rag_service is not None
    return {
        "status": "online" if is_ready else "starting",
        "service": "TU Delft Academic RAG API",
        "version": "1.0.0",
        "ready": is_ready
    }


@app.post("/api/v1/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_publications(req: SearchRequest):
    """Executes asynchronous dense vector similarity search against Qdrant Vector DB."""
    _ensure_service_ready()
    try:
        results = await rag_service.search_similar_chunks(
            query=req.query,
            top_k=req.top_k,
            score_threshold=req.score_threshold
        )
        return SearchResponse(
            query=req.query,
            total_results=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )


@app.post("/api/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def generate_chat_response(req: ChatRequest):
    """Executes full RAG workflow: Vector Retrieval + Augmented LLM Answer Generation."""
    _ensure_service_ready()
    try:
        answer, citations, retrieved_count = await rag_service.generate_rag_response(
            query=req.query,
            top_k=req.top_k,
            temperature=req.temperature
        )
        return ChatResponse(
            query=req.query,
            answer=answer,
            citations=citations,
            retrieved_chunks_count=retrieved_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG Chat service error: {str(e)}"
        )


@app.get("/api/v1/publications", status_code=status.HTTP_200_OK)
async def get_publications_catalog():
    """Reads publications catalog directly from Gold Parquet Lakehouse or Silver JSON fallback."""
    gold_dir = "/app/data/gold/publications"
    silver_file = "/app/data/silver/cleaned_publications.json"
    
    publications = []
    try:
        # Priority 1: Check Gold Layer Parquet files
        parquet_files = glob.glob(f"{gold_dir}/**/*.parquet", recursive=True)
        if parquet_files:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            if "title" in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=["title"])
            
            # Replace NaNs safely at DataFrame level
            combined_df = combined_df.replace({np.nan: None})
            publications = combined_df.to_dict(orient="records")
            
        # Priority 2: Fallback to Silver Layer JSON
        elif os.path.exists(silver_file):
            silver_df = pd.read_json(silver_file)
            silver_df = silver_df.replace({np.nan: None})
            publications = silver_df.to_dict(orient="records")

        # Type-safe cleanup for JSON response
        clean_pubs = []
        for pub in publications:
            item = {}
            for k, v in pub.items():
                if isinstance(v, (np.integer, int)):
                    item[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    item[k] = None if np.isnan(v) else float(v)
                elif isinstance(v, np.ndarray):
                    item[k] = v.tolist()
                else:
                    item[k] = v
            clean_pubs.append(item)
                    
        return clean_pubs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch publications catalog: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
