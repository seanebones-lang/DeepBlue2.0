#!/usr/bin/env python3
"""
📚 KNOWLEDGE INTEGRATION SYSTEM - DEEPBLUE 2.0 ULTIMATE UPGRADE
Integrates all knowledge from original DeepBlue system with advanced capabilities
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
from pathlib import Path
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

logger = structlog.get_logger()

@dataclass
class KnowledgeItem:
    """Represents a knowledge item."""
    id: str
    content: str
    category: str
    source: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgeIntegrationConfig:
    """Configuration for knowledge integration."""
    knowledge_base_path: str = "./knowledge_base"
    converted_kb_path: str = "./converted_kb"
    vector_db_path: str = "./vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7

class KnowledgeIntegrationSystem:
    """Advanced knowledge integration system."""
    
    def __init__(self, config: KnowledgeIntegrationConfig = None):
        self.config = config or KnowledgeIntegrationConfig()
        self.knowledge_items = []
        self.vector_store = None
        self.embedding_model = None
        self.categories = {
            'machine_learning': [],
            'neural_networks': [],
            'ai_components': [],
            'build_systems': [],
            'troubleshooting': [],
            'hacks': [],
            'agi_consciousness': [],
            'trusted_sources': [],
            'rag_systems': [],
            'monitoring': []
        }
        
        logger.info("📚 Knowledge Integration System initializing...")
    
    async def initialize(self) -> bool:
        """Initialize the knowledge integration system."""
        try:
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize vector store
            await self._initialize_vector_store()
            
            # Load knowledge from original DeepBlue
            await self._load_original_knowledge()
            
            # Process and index knowledge
            await self._process_knowledge()
            
            # Create knowledge database
            await self._create_knowledge_database()
            
            logger.info("✅ Knowledge Integration System initialized")
            return True
            
        except Exception as e:
            logger.error("❌ Knowledge Integration System initialization failed", error=str(e))
            return False
    
    async def _initialize_embedding_model(self):
        """Initialize embedding model."""
        logger.info("🔤 Initializing embedding model...")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        logger.info("✅ Embedding model initialized")
    
    async def _initialize_vector_store(self):
        """Initialize vector store."""
        logger.info("🗄️ Initializing vector store...")
        self.vector_store = chromadb.PersistentClient(path=self.config.vector_db_path)
        
        # Create or get collection
        try:
            self.collection = self.vector_store.get_collection("deepblue2_knowledge")
        except:
            self.collection = self.vector_store.create_collection("deepblue2_knowledge")
        
        logger.info("✅ Vector store initialized")
    
    async def _load_original_knowledge(self):
        """Load knowledge from original DeepBlue system."""
        logger.info("📖 Loading knowledge from original DeepBlue...")
        
        # Load knowledge base files
        knowledge_base_path = Path(self.config.knowledge_base_path)
        if knowledge_base_path.exists():
            for file_path in knowledge_base_path.glob("*.txt"):
                await self._load_text_file(file_path)
        
        # Load converted knowledge base
        converted_kb_path = Path(self.config.converted_kb_path)
        if converted_kb_path.exists():
            for file_path in converted_kb_path.glob("*.json"):
                await self._load_json_file(file_path)
        
        # Load additional knowledge from original system files
        await self._load_system_knowledge()
        
        logger.info(f"✅ Loaded {len(self.knowledge_items)} knowledge items")
    
    async def _load_text_file(self, file_path: Path):
        """Load knowledge from text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine category from filename
            category = self._determine_category(file_path.stem)
            
            # Create knowledge item
            knowledge_item = KnowledgeItem(
                id=f"text_{file_path.stem}_{int(time.time())}",
                content=content,
                category=category,
                source=f"knowledge_base/{file_path.name}",
                confidence=0.9,
                timestamp=datetime.now(),
                metadata={"file_type": "text", "file_size": len(content)}
            )
            
            self.knowledge_items.append(knowledge_item)
            self.categories[category].append(knowledge_item)
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
    
    async def _load_json_file(self, file_path: Path):
        """Load knowledge from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process JSON data
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 50:
                        category = self._determine_category(key)
                        
                        knowledge_item = KnowledgeItem(
                            id=f"json_{key}_{int(time.time())}",
                            content=value,
                            category=category,
                            source=f"converted_kb/{file_path.name}",
                            confidence=0.8,
                            timestamp=datetime.now(),
                            metadata={"file_type": "json", "key": key}
                        )
                        
                        self.knowledge_items.append(knowledge_item)
                        self.categories[category].append(knowledge_item)
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
    
    async def _load_system_knowledge(self):
        """Load knowledge from original DeepBlue system files."""
        # Load from original DeepBlue directory
        original_deepblue_path = "/Users/seanmcdonnell/Desktop/DeepBlue"
        
        # Load from key system files
        system_files = [
            "ultimate_deepblue_system.py",
            "comprehensive_2025_knowledge_aggregator.py",
            "ultimate_2025_knowledge_rag.py",
            "rag_100_percent.py",
            "trusted_source_aggregator.py",
            "ultimate_hack_system.py",
            "working_agi_system.py",
            "hallucination_safeguard_system.py",
            "build_diagnosis_system.py",
            "system_builder.py"
        ]
        
        for filename in system_files:
            file_path = Path(original_deepblue_path) / filename
            if file_path.exists():
                await self._load_python_file(file_path)
    
    async def _load_python_file(self, file_path: Path):
        """Load knowledge from Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract docstrings and comments
            docstrings = self._extract_docstrings(content)
            comments = self._extract_comments(content)
            
            # Create knowledge items for docstrings
            for i, docstring in enumerate(docstrings):
                if len(docstring.strip()) > 100:
                    knowledge_item = KnowledgeItem(
                        id=f"python_doc_{file_path.stem}_{i}_{int(time.time())}",
                        content=docstring,
                        category="system_documentation",
                        source=f"original_deepblue/{file_path.name}",
                        confidence=0.95,
                        timestamp=datetime.now(),
                        metadata={"file_type": "python", "extraction_type": "docstring"}
                    )
                    
                    self.knowledge_items.append(knowledge_item)
                    self.categories["system_documentation"].append(knowledge_item)
            
            # Create knowledge items for comments
            for i, comment in enumerate(comments):
                if len(comment.strip()) > 50:
                    knowledge_item = KnowledgeItem(
                        id=f"python_comment_{file_path.stem}_{i}_{int(time.time())}",
                        content=comment,
                        category="system_implementation",
                        source=f"original_deepblue/{file_path.name}",
                        confidence=0.8,
                        timestamp=datetime.now(),
                        metadata={"file_type": "python", "extraction_type": "comment"}
                    )
                    
                    self.knowledge_items.append(knowledge_item)
                    self.categories["system_implementation"].append(knowledge_item)
            
        except Exception as e:
            logger.error(f"Error loading Python file {file_path}: {e}")
    
    def _extract_docstrings(self, content: str) -> List[str]:
        """Extract docstrings from Python code."""
        import ast
        
        try:
            tree = ast.parse(content)
            docstrings = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if ast.get_docstring(node):
                        docstrings.append(ast.get_docstring(node))
            
            return docstrings
        except:
            return []
    
    def _extract_comments(self, content: str) -> List[str]:
        """Extract meaningful comments from Python code."""
        lines = content.split('\n')
        comments = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#') and len(line) > 10:
                comment = line[1:].strip()
                if len(comment) > 20:  # Only meaningful comments
                    comments.append(comment)
        
        return comments
    
    def _determine_category(self, filename: str) -> str:
        """Determine category based on filename."""
        filename_lower = filename.lower()
        
        if 'machine_learning' in filename_lower or 'ml' in filename_lower:
            return 'machine_learning'
        elif 'neural' in filename_lower or 'network' in filename_lower:
            return 'neural_networks'
        elif 'ai' in filename_lower or 'artificial' in filename_lower:
            return 'ai_components'
        elif 'build' in filename_lower or 'system' in filename_lower:
            return 'build_systems'
        elif 'hack' in filename_lower or 'bypass' in filename_lower:
            return 'hacks'
        elif 'agi' in filename_lower or 'consciousness' in filename_lower:
            return 'agi_consciousness'
        elif 'rag' in filename_lower or 'retrieval' in filename_lower:
            return 'rag_systems'
        elif 'trusted' in filename_lower or 'source' in filename_lower:
            return 'trusted_sources'
        elif 'monitor' in filename_lower or 'log' in filename_lower:
            return 'monitoring'
        else:
            return 'general'
    
    async def _process_knowledge(self):
        """Process and chunk knowledge items."""
        logger.info("🔄 Processing knowledge items...")
        
        for item in self.knowledge_items:
            # Chunk large content
            chunks = self._chunk_content(item.content)
            
            for i, chunk in enumerate(chunks):
                # Create chunk item
                chunk_item = KnowledgeItem(
                    id=f"{item.id}_chunk_{i}",
                    content=chunk,
                    category=item.category,
                    source=item.source,
                    confidence=item.confidence,
                    timestamp=item.timestamp,
                    metadata={
                        **item.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_id": item.id
                    }
                )
                
                # Add to vector store
                await self._add_to_vector_store(chunk_item)
        
        logger.info("✅ Knowledge processing completed")
    
    def _chunk_content(self, content: str) -> List[str]:
        """Chunk content into smaller pieces."""
        if len(content) <= self.config.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + self.config.chunk_size
            
            if end >= len(content):
                chunks.append(content[start:])
                break
            
            # Try to break at sentence boundary
            last_period = content.rfind('.', start, end)
            last_newline = content.rfind('\n', start, end)
            
            if last_period > start and last_period - start > self.config.chunk_size * 0.5:
                end = last_period + 1
            elif last_newline > start and last_newline - start > self.config.chunk_size * 0.5:
                end = last_newline + 1
            
            chunks.append(content[start:end])
            start = end - self.config.chunk_overlap
        
        return chunks
    
    async def _add_to_vector_store(self, item: KnowledgeItem):
        """Add knowledge item to vector store."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(item.content).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                ids=[item.id],
                embeddings=[embedding],
                documents=[item.content],
                metadatas=[{
                    "category": item.category,
                    "source": item.source,
                    "confidence": item.confidence,
                    "timestamp": item.timestamp.isoformat(),
                    **item.metadata
                }]
            )
            
        except Exception as e:
            logger.error(f"Error adding item to vector store: {e}")
    
    async def _create_knowledge_database(self):
        """Create SQLite database for knowledge management."""
        logger.info("🗃️ Creating knowledge database...")
        
        conn = sqlite3.connect("deepblue2_knowledge.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                name TEXT PRIMARY KEY,
                item_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        ''')
        
        # Insert knowledge items
        for item in self.knowledge_items:
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_items 
                (id, content, category, source, confidence, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.id,
                item.content,
                item.category,
                item.source,
                item.confidence,
                item.timestamp.isoformat(),
                json.dumps(item.metadata) if item.metadata else None
            ))
        
        # Insert category statistics
        for category, items in self.categories.items():
            if items:
                cursor.execute('''
                    INSERT OR REPLACE INTO categories 
                    (name, item_count, last_updated)
                    VALUES (?, ?, ?)
                ''', (
                    category,
                    len(items),
                    datetime.now().isoformat()
                ))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Knowledge database created")
    
    async def search_knowledge(
        self, 
        query: str, 
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search vector store
            where_clause = {"category": category} if category else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause
            )
            
            # Format results
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                search_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "rank": i + 1
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "total_items": len(self.knowledge_items),
            "categories": {cat: len(items) for cat, items in self.categories.items()},
            "vector_store_size": self.collection.count(),
            "embedding_model": self.config.embedding_model,
            "last_updated": datetime.now().isoformat()
        }
    
    async def add_knowledge(
        self, 
        content: str, 
        category: str, 
        source: str,
        confidence: float = 0.8
    ) -> str:
        """Add new knowledge to the system."""
        knowledge_id = f"new_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        knowledge_item = KnowledgeItem(
            id=knowledge_id,
            content=content,
            category=category,
            source=source,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.knowledge_items.append(knowledge_item)
        self.categories[category].append(knowledge_item)
        
        # Add to vector store
        await self._add_to_vector_store(knowledge_item)
        
        logger.info(f"Added new knowledge: {knowledge_id}")
        return knowledge_id

# Global knowledge integration system
knowledge_integration = KnowledgeIntegrationSystem()

async def main():
    """Main function for testing."""
    if await knowledge_integration.initialize():
        logger.info("📚 Knowledge Integration System is ready!")
        
        # Test search
        results = await knowledge_integration.search_knowledge("machine learning", limit=5)
        print(f"Found {len(results)} results")
        
        # Get stats
        stats = await knowledge_integration.get_knowledge_stats()
        print(f"Knowledge stats: {stats}")
    else:
        logger.error("❌ Knowledge Integration System failed to initialize")

if __name__ == "__main__":
    asyncio.run(main())

