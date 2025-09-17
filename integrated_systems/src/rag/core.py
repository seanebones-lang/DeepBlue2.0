"""
Core RAG System Implementation
"""

from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import asyncio
import logging
from dataclasses import dataclass

from .retrievers import HybridRetriever
from .generators import LLMGenerator
from .embeddings import EmbeddingManager
from .vectorstores import VectorStoreManager
from .rerankers import RerankerManager
from .chunkers import ChunkingManager

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chroma"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    reranker: str = "cross-encoder"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    rerank_top_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000

class RAGSystem:
    """Modular RAG System with cutting-edge features."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_manager = EmbeddingManager(config.embedding_model)
        self.vector_store = VectorStoreManager(config.vector_store)
        self.retriever = HybridRetriever(
            embedding_manager=self.embedding_manager,
            vector_store=self.vector_store,
            config=config
        )
        self.generator = LLMGenerator(config)
        self.reranker = RerankerManager(config.reranker)
        self.chunker = ChunkingManager(config)
        
    async def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the knowledge base."""
        try:
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            
            # Generate embeddings
            embeddings = await self.embedding_manager.embed_documents(chunks)
            
            # Store in vector database
            await self.vector_store.add_documents(chunks, embeddings, metadata)
            
            logger.info(f"Added {len(chunks)} chunks to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def query(self, question: str, use_reranking: bool = True) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            # Retrieve relevant documents
            retrieved_docs = await self.retriever.retrieve(question, top_k=self.config.top_k)
            
            # Rerank if enabled
            if use_reranking and len(retrieved_docs) > self.config.rerank_top_k:
                retrieved_docs = await self.reranker.rerank(
                    question, retrieved_docs, top_k=self.config.rerank_top_k
                )
            
            # Generate response
            context = "\n".join([doc["content"] for doc in retrieved_docs])
            response = await self.generator.generate(question, context)
            
            return {
                "answer": response,
                "sources": retrieved_docs,
                "metadata": {
                    "retrieval_count": len(retrieved_docs),
                    "reranked": use_reranking
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {"error": str(e)}
    
    async def stream_query(self, question: str, use_reranking: bool = True):
        """Stream query results."""
        try:
            # Retrieve and rerank (same as query)
            retrieved_docs = await self.retriever.retrieve(question, top_k=self.config.top_k)
            
            if use_reranking and len(retrieved_docs) > self.config.rerank_top_k:
                retrieved_docs = await self.reranker.rerank(
                    question, retrieved_docs, top_k=self.config.rerank_top_k
                )
            
            # Stream generation
            context = "\n".join([doc["content"] for doc in retrieved_docs])
            async for chunk in self.generator.stream_generate(question, context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            yield {"error": str(e)}
