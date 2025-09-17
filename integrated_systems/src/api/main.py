"""
FastAPI Web Interface for RAG System
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import logging

from ..rag.core import RAGSystem, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DeepBlue RAG System",
    description="Cutting-edge modular RAG system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
config = RAGConfig()
rag_system = RAGSystem(config)

class QueryRequest(BaseModel):
    question: str
    use_reranking: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class DocumentUpload(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

@app.get("/")
async def root():
    return {"message": "DeepBlue RAG System API", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = await rag_system.query(request.question, request.use_reranking)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_documents(documents: List[DocumentUpload]):
    """Add documents to the knowledge base."""
    try:
        contents = [doc.content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        
        success = await rag_system.add_documents(contents, metadata)
        if success:
            return {"message": f"Added {len(documents)} documents successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add documents")
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-system"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
