#!/usr/bin/env python3
"""
🔍 ENHANCED RAG SYSTEM - DEEPBLUE 2.0 ULTIMATE UPGRADE
Integrates advanced RAG capabilities from original DeepBlue with new features
"""

import asyncio
import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.llms import OpenAI, Anthropic, GoogleGenerativeAI, Ollama
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, Weaviate, Qdrant, FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document

logger = structlog.get_logger()

@dataclass
class RAGConfig:
    """Configuration for enhanced RAG system."""
    # Model configurations
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4-turbo"
    temperature: float = 0.7
    max_tokens: int = 4000
    
    # Vector store configurations
    vector_store_type: str = "chroma"  # chroma, pinecone, weaviate, qdrant, faiss
    collection_name: str = "deepblue2_rag"
    persist_directory: str = "./vector_db"
    
    # Retrieval configurations
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_docs: int = 5
    similarity_threshold: float = 0.7
    
    # Advanced features
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    enable_context_compression: bool = True
    enable_multi_modal: bool = True
    enable_conversation_memory: bool = True

class EnhancedRAGSystem:
    """Enhanced RAG system with advanced capabilities."""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedding_model = None
        self.llm = None
        self.vector_store = None
        self.memory = None
        self.conversation_chain = None
        self.query_expander = None
        self.reranker = None
        
        logger.info("🔍 Enhanced RAG System initializing...")
    
    async def initialize(self) -> bool:
        """Initialize the enhanced RAG system."""
        try:
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize LLM
            await self._initialize_llm()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Initialize memory
            await self._initialize_memory()
            
            # Initialize advanced components
            await self._initialize_advanced_components()
            
            # Create conversation chain
            await self._create_conversation_chain()
            
            logger.info("✅ Enhanced RAG System initialized")
            return True
            
        except Exception as e:
            logger.error("❌ Enhanced RAG System initialization failed", error=str(e))
            return False
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model."""
        logger.info("🔤 Initializing embedding model...")
        
        if self.config.embedding_model.startswith("text-embedding-"):
            self.embedding_model = OpenAIEmbeddings(model=self.config.embedding_model)
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
        
        logger.info("✅ Embedding model initialized")
    
    async def _initialize_llm(self):
        """Initialize LLM."""
        logger.info("🤖 Initializing LLM...")
        
        if self.config.llm_model.startswith("gpt-"):
            self.llm = OpenAI(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.llm_model.startswith("claude-"):
            self.llm = Anthropic(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.llm_model.startswith("gemini-"):
            self.llm = GoogleGenerativeAI(
                model=self.config.llm_model,
                temperature=self.config.temperature
            )
        else:
            # Default to OpenAI
            self.llm = OpenAI(
                model_name="gpt-4-turbo",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        
        logger.info("✅ LLM initialized")
    
    async def _initialize_vector_store(self):
        """Initialize vector store."""
        logger.info("🗄️ Initializing vector store...")
        
        if self.config.vector_store_type == "chroma":
            self.vector_store = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.config.persist_directory
            )
        elif self.config.vector_store_type == "pinecone":
            import pinecone
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
            self.vector_store = Pinecone(
                index_name=self.config.collection_name,
                embedding_function=self.embedding_model
            )
        elif self.config.vector_store_type == "weaviate":
            self.vector_store = Weaviate(
                client=weaviate.Client("http://localhost:8080"),
                index_name=self.config.collection_name,
                text_key="text",
                embedding=self.embedding_model
            )
        elif self.config.vector_store_type == "qdrant":
            self.vector_store = Qdrant(
                client=QdrantClient("localhost", port=6333),
                collection_name=self.config.collection_name,
                embeddings=self.embedding_model
            )
        elif self.config.vector_store_type == "faiss":
            # Initialize with empty documents
            self.vector_store = FAISS.from_texts(
                ["DeepBlue 2.0 RAG System initialized"],
                self.embedding_model
            )
        
        logger.info("✅ Vector store initialized")
    
    async def _initialize_memory(self):
        """Initialize conversation memory."""
        logger.info("🧠 Initializing conversation memory...")
        
        if self.config.enable_conversation_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            self.memory = None
        
        logger.info("✅ Memory initialized")
    
    async def _initialize_advanced_components(self):
        """Initialize advanced RAG components."""
        logger.info("⚡ Initializing advanced components...")
        
        # Initialize query expander
        if self.config.enable_query_expansion:
            self.query_expander = QueryExpander()
        
        # Initialize reranker
        if self.config.enable_reranking:
            self.reranker = Reranker()
        
        logger.info("✅ Advanced components initialized")
    
    async def _create_conversation_chain(self):
        """Create conversation chain."""
        logger.info("🔗 Creating conversation chain...")
        
        # Create retrieval chain
        if self.memory:
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.max_docs}
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        else:
            self.conversation_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.max_docs}
                ),
                return_source_documents=True
            )
        
        logger.info("✅ Conversation chain created")
    
    async def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector store."""
        try:
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            # Add documents to vector store
            self.vector_store.add_texts(
                texts=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    async def query(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None,
        use_advanced_features: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system."""
        start_time = time.time()
        
        try:
            # Expand query if enabled
            if use_advanced_features and self.query_expander:
                expanded_query = await self.query_expander.expand_query(question)
                logger.info(f"Query expanded: {expanded_query}")
            else:
                expanded_query = question
            
            # Perform retrieval
            if self.memory:
                # Use conversational retrieval
                result = await self.conversation_chain.acall({
                    "question": expanded_query,
                    "chat_history": []
                })
                
                answer = result["answer"]
                source_docs = result.get("source_documents", [])
            else:
                # Use standard retrieval
                result = self.conversation_chain({"query": expanded_query})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
            
            # Rerank results if enabled
            if use_advanced_features and self.reranker and source_docs:
                source_docs = await self.reranker.rerank(expanded_query, source_docs)
            
            # Compress context if enabled
            if use_advanced_features and self.config.enable_context_compression:
                answer = await self._compress_context(answer, source_docs)
            
            processing_time = time.time() - start_time
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', 0.0)
                    }
                    for doc in source_docs
                ],
                "query": question,
                "expanded_query": expanded_query if expanded_query != question else None,
                "processing_time": processing_time,
                "model_used": self.config.llm_model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "query": question,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _compress_context(self, answer: str, source_docs: List[Document]) -> str:
        """Compress context to reduce token usage."""
        # Simple context compression - in production, use more sophisticated methods
        if len(answer) > 2000:
            # Truncate answer if too long
            answer = answer[:2000] + "..."
        
        return answer
    
    async def get_similar_documents(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar documents without generating answer."""
        try:
            docs = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in docs
            ]
            
        except Exception as e:
            logger.error(f"Error getting similar documents: {e}")
            return []
    
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        if not self.memory:
            return []
        
        try:
            history = self.memory.chat_memory.messages
            return [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in history
            ]
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")

class QueryExpander:
    """Query expansion for better retrieval."""
    
    def __init__(self):
        self.expansion_patterns = [
            "What is",
            "How does",
            "Explain",
            "Describe",
            "Tell me about",
            "What are the benefits of",
            "What are the drawbacks of",
            "How to implement",
            "Best practices for",
            "Common issues with"
        ]
    
    async def expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        # Simple query expansion - in production, use more sophisticated methods
        expanded_terms = []
        
        # Add expansion patterns
        for pattern in self.expansion_patterns:
            if pattern.lower() not in query.lower():
                expanded_terms.append(f"{pattern} {query}")
        
        # Combine original query with expansions
        expanded_query = f"{query} {' '.join(expanded_terms[:3])}"
        
        return expanded_query

class Reranker:
    """Document reranking for better relevance."""
    
    def __init__(self):
        self.reranking_model = None
    
    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents by relevance."""
        # Simple reranking - in production, use more sophisticated methods
        # For now, just return documents as-is
        return documents

# Global enhanced RAG system
enhanced_rag = EnhancedRAGSystem()

async def main():
    """Main function for testing."""
    if await enhanced_rag.initialize():
        logger.info("🔍 Enhanced RAG System is ready!")
        
        # Add some test documents
        test_docs = [
            "DeepBlue 2.0 is an advanced AI system with maximum capabilities.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks."
        ]
        
        await enhanced_rag.add_documents(test_docs)
        
        # Test query
        result = await enhanced_rag.query("What is DeepBlue 2.0?")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['source_documents'])}")
    else:
        logger.error("❌ Enhanced RAG System failed to initialize")

if __name__ == "__main__":
    asyncio.run(main())

