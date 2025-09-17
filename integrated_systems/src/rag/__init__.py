"""
🌊 DeepBlue RAG System - Cutting Edge Modular RAG
The most advanced, modular, and scalable RAG system available.
"""

from .core import RAGSystem
from .retrievers import HybridRetriever, DenseRetriever, SparseRetriever
from .generators import LLMGenerator, StreamingGenerator
from .embeddings import EmbeddingManager
from .vectorstores import VectorStoreManager
from .rerankers import RerankerManager
from .chunkers import ChunkingManager

__version__ = "1.0.0"
__all__ = [
    "RAGSystem",
    "HybridRetriever", 
    "DenseRetriever",
    "SparseRetriever",
    "LLMGenerator",
    "StreamingGenerator",
    "EmbeddingManager",
    "VectorStoreManager", 
    "RerankerManager",
    "ChunkingManager"
]
