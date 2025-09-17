# 🌊 DeepBlue RAG System

The most cutting-edge, modular, and scalable RAG (Retrieval-Augmented Generation) system available.

## 🚀 Features

### Core Capabilities
- **Hybrid Retrieval**: Combines dense and sparse search methods
- **Multiple Vector Stores**: Pinecone, Weaviate, Qdrant, Chroma, FAISS
- **Multiple LLM Providers**: OpenAI, Anthropic, Ollama, Hugging Face
- **Advanced Reranking**: Cross-encoder and specialized rerankers
- **Smart Chunking**: Recursive, semantic, and fixed strategies
- **Streaming Support**: Real-time response streaming
- **Modular Architecture**: Easy to extend and customize

### Advanced Features
- **Multi-modal Support**: Text, PDF, DOCX, Markdown processing
- **Semantic Search**: Advanced embedding models
- **Query Expansion**: Automatic query enhancement
- **Context Compression**: Intelligent context management
- **Caching**: Redis-based response caching
- **Monitoring**: Prometheus metrics and structured logging
- **Docker Support**: Containerized deployment
- **Kubernetes Ready**: Production-scale orchestration

## 🛠️ Quick Start

### Installation

```bash
# Clone and setup
git clone <your-repo>
cd <project-name>

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export PINECONE_API_KEY="your-key"

# Run the system
python -m uvicorn src.api.main:app --reload
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# The API will be available at http://localhost:8000
```

### Basic Usage

```python
from src.rag.core import RAGSystem, RAGConfig

# Initialize
config = RAGConfig()
rag = RAGSystem(config)

# Add documents
await rag.add_documents(["Your document content here"])

# Query
result = await rag.query("Your question here")
print(result["answer"])
```

### API Usage

```bash
# Query the system
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?", "use_reranking": true}'

# Add documents
curl -X POST "http://localhost:8000/documents" \
     -H "Content-Type: application/json" \
     -d '[{"content": "Document content", "metadata": {"source": "test"}}]'
```

## 🏗️ Architecture

```
src/
├── rag/
│   ├── core.py              # Main RAG system
│   ├── retrievers/          # Retrieval methods
│   ├── generators/          # LLM generators
│   ├── embeddings/          # Embedding management
│   ├── vectorstores/        # Vector database interfaces
│   ├── rerankers/           # Reranking methods
│   └── chunkers/            # Document chunking
├── api/                     # FastAPI web interface
└── web/                     # Frontend interface
```

## 🔧 Configuration

Edit `config/settings.yaml` to customize:
- Embedding models
- Vector stores
- LLM providers
- Chunking strategies
- Retrieval parameters

## 📊 Monitoring

- **Metrics**: Prometheus-compatible metrics
- **Logging**: Structured logging with context
- **Health Checks**: Built-in health monitoring
- **Tracing**: Request tracing and performance monitoring

## 🚀 Production Deployment

### Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key

# Optional
PINECONE_API_KEY=your-key
WEAVIATE_URL=http://weaviate:8080
REDIS_URL=redis://redis:6379
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

*Built with DeepBlue 🌊 - We're gonna need a bigger boat!*
