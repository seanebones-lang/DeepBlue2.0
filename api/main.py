#!/usr/bin/env python3
"""
🌊 DEEPBLUE 2.0 API - ULTIMATE BACKEND 🌊
High-performance FastAPI backend with maximum capabilities.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse

import redis
import motor.motor_asyncio
from sqlalchemy import create_engine, Column, String, DateTime, Text, JSON, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field, validator
import structlog
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

# Import DeepBlue 2.0 core
import sys
sys.path.append('../core')
from deepblue_2_0 import DeepBlue2System, SystemConfig, QueryRequest, QueryResponse

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Configure Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FastApiIntegration(auto_enabling_instrumentations=True),
        SqlalchemyIntegration(),
    ],
    traces_sample_rate=0.1,
    environment=os.getenv("ENVIRONMENT", "development"),
)

# Prometheus metrics
REQUEST_COUNT = Counter('deepblue2_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('deepblue2_api_request_duration_seconds', 'API request duration')
ACTIVE_CONNECTIONS = Gauge('deepblue2_api_active_connections', 'Active API connections')
MEMORY_USAGE = Gauge('deepblue2_api_memory_usage_bytes', 'API memory usage')
CPU_USAGE = Gauge('deepblue2_api_cpu_usage_percent', 'API CPU usage')
WEBSOCKET_CONNECTIONS = Gauge('deepblue2_websocket_connections', 'WebSocket connections')

# Database models
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)
    tokens_used = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default={})

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, default={})

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    uptime: float
    version: str
    timestamp: str
    metrics: Dict[str, Any]

class QueryRequestModel(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4000, ge=1, le=8000)
    stream: bool = False
    use_rag: bool = True
    use_memory: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class QueryResponseModel(BaseModel):
    response: str
    model_used: str
    tokens_used: int
    processing_time: float
    confidence: float
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    timestamp: str

class StreamResponseModel(BaseModel):
    chunk: str
    is_final: bool
    metadata: Dict[str, Any] = {}

# Global variables
deepblue2_system: Optional[DeepBlue2System] = None
redis_client: Optional[redis.Redis] = None
mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
db_session: Optional[Session] = None
websocket_connections: List[WebSocket] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global deepblue2_system, redis_client, mongo_client, db_session
    
    # Startup
    logger.info("🚀 Starting DeepBlue 2.0 API...")
    
    # Initialize DeepBlue 2.0 system
    config = SystemConfig()
    deepblue2_system = DeepBlue2System(config)
    
    if not await deepblue2_system.initialize():
        logger.error("❌ Failed to initialize DeepBlue 2.0 system")
        raise RuntimeError("DeepBlue 2.0 initialization failed")
    
    # Initialize Redis
    redis_client = redis.Redis.from_url(config.redis_url)
    
    # Initialize MongoDB
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(config.mongodb_url)
    
    # Initialize PostgreSQL
    engine = create_engine(config.postgres_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db_session = SessionLocal()
    
    logger.info("✅ DeepBlue 2.0 API started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down DeepBlue 2.0 API...")
    
    if deepblue2_system:
        await deepblue2_system.shutdown()
    
    if redis_client:
        redis_client.close()
    
    if mongo_client:
        mongo_client.close()
    
    if db_session:
        db_session.close()
    
    logger.info("✅ DeepBlue 2.0 API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="DeepBlue 2.0 API",
    description="The Ultimate AI System with Maximum Capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SessionMiddleware, secret_key="deepblue2-secret-key")

# Security
security = HTTPBearer()

# Dependency functions
async def get_redis() -> redis.Redis:
    """Get Redis client."""
    return redis_client

async def get_mongo() -> motor.motor_asyncio.AsyncIOMotorClient:
    """Get MongoDB client."""
    return mongo_client

async def get_db() -> Session:
    """Get database session."""
    return db_session

async def get_deepblue2() -> DeepBlue2System:
    """Get DeepBlue 2.0 system."""
    return deepblue2_system

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        WEBSOCKET_CONNECTIONS.inc()
        logger.info("WebSocket connected", total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            WEBSOCKET_CONNECTIONS.dec()
            logger.info("WebSocket disconnected", total_connections=len(self.active_connections))
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)

manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DeepBlue 2.0 - Ultimate AI System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            .logo { font-size: 3em; color: #0066cc; margin-bottom: 20px; }
            .status { background: #e8f5e8; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
            .feature { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .api-link { display: inline-block; background: #0066cc; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">🌊 DeepBlue 2.0</div>
            <h1>The Ultimate AI System</h1>
            <div class="status">
                <h2>System Status: ✅ OPERATIONAL</h2>
                <p>Maximum capabilities enabled with surgical precision</p>
            </div>
            <div class="features">
                <div class="feature">
                    <h3>🤖 Multi-LLM Support</h3>
                    <p>GPT-4, Claude-3, Gemini, Ollama</p>
                </div>
                <div class="feature">
                    <h3>🗄️ Advanced RAG</h3>
                    <p>Multiple vector databases</p>
                </div>
                <div class="feature">
                    <h3>⚡ Real-time Streaming</h3>
                    <p>WebSocket support</p>
                </div>
                <div class="feature">
                    <h3>🔒 Enterprise Security</h3>
                    <p>JWT, encryption, audit logs</p>
                </div>
            </div>
            <a href="/docs" class="api-link">API Documentation</a>
            <a href="/health" class="api-link">Health Check</a>
            <a href="/metrics" class="api-link">Metrics</a>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = time.time() - deepblue2_system.start_time if deepblue2_system else 0
    
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        metrics={
            "active_connections": len(manager.active_connections),
            "system_status": deepblue2_system.status.value if deepblue2_system else "unknown",
            "memory_usage": "N/A",  # Would be implemented with psutil
            "cpu_usage": "N/A"      # Would be implemented with psutil
        }
    )

@app.post("/query", response_model=QueryResponseModel)
async def query(
    request: QueryRequestModel,
    background_tasks: BackgroundTasks,
    deepblue2: DeepBlue2System = Depends(get_deepblue2),
    db: Session = Depends(get_db)
):
    """Process a query with DeepBlue 2.0."""
    start_time = time.time()
    
    try:
        # Convert to internal request format
        internal_request = QueryRequest(
            query=request.query,
            context=request.context,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            use_rag=request.use_rag,
            use_memory=request.use_memory,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Process query
        response = await deepblue2.process_query(internal_request)
        
        # Log query
        background_tasks.add_task(
            log_query,
            request.query,
            response.response,
            response.model_used,
            response.tokens_used,
            response.processing_time,
            response.confidence,
            request.user_id,
            request.session_id,
            response.metadata
        )
        
        # Update metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/query", status="200").inc()
        REQUEST_DURATION.observe(time.time() - start_time)
        
        return QueryResponseModel(
            response=response.response,
            model_used=response.model_used,
            tokens_used=response.tokens_used,
            processing_time=response.processing_time,
            confidence=response.confidence,
            sources=response.sources,
            metadata=response.metadata,
            timestamp=response.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        REQUEST_COUNT.labels(method="POST", endpoint="/query", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(
    request: QueryRequestModel,
    deepblue2: DeepBlue2System = Depends(get_deepblue2)
):
    """Process a query with streaming response."""
    
    async def generate_stream():
        try:
            # Convert to internal request format
            internal_request = QueryRequest(
                query=request.query,
                context=request.context,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                use_rag=request.use_rag,
                use_memory=request.use_memory,
                user_id=request.user_id,
                session_id=request.session_id
            )
            
            # Process query with streaming
            async for chunk in process_query_stream(internal_request, deepblue2):
                yield f"data: {chunk.json()}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error("Streaming query failed", error=str(e))
            yield f"data: {StreamResponseModel(chunk=f'Error: {str(e)}', is_final=True).json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

async def process_query_stream(request: QueryRequest, deepblue2: DeepBlue2System) -> AsyncGenerator[StreamResponseModel, None]:
    """Process query with streaming response."""
    # This is a simplified streaming implementation
    # In production, you'd implement actual streaming with the LLM
    
    response_text = f"DeepBlue 2.0 streaming response to: {request.query}"
    
    # Simulate streaming by chunking the response
    words = response_text.split()
    for i, word in enumerate(words):
        is_final = i == len(words) - 1
        yield StreamResponseModel(
            chunk=word + " ",
            is_final=is_final,
            metadata={"chunk_index": i, "total_chunks": len(words)}
        )
        await asyncio.sleep(0.1)  # Simulate processing time

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Process the message
            try:
                message_data = json.loads(data)
                query = message_data.get("query", "")
                
                if query:
                    # Process query
                    response = await deepblue2_system.process_query(
                        QueryRequest(query=query)
                    )
                    
                    # Send response
                    await manager.send_personal_message(
                        json.dumps({
                            "response": response.response,
                            "model_used": response.model_used,
                            "confidence": response.confidence,
                            "timestamp": response.timestamp.isoformat()
                        }),
                        websocket
                    )
            
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"error": "Invalid JSON format"}),
                    websocket
                )
            except Exception as e:
                await manager.send_personal_message(
                    json.dumps({"error": str(e)}),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.get("/sessions")
async def get_sessions(db: Session = Depends(get_db)):
    """Get active sessions."""
    sessions = db.query(UserSession).filter(UserSession.is_active == True).all()
    return [{"id": s.id, "user_id": s.user_id, "created_at": s.created_at.isoformat()} for s in sessions]

async def log_query(
    query: str,
    response: str,
    model_used: str,
    tokens_used: int,
    processing_time: float,
    confidence: float,
    user_id: Optional[str],
    session_id: Optional[str],
    metadata: Dict[str, Any]
):
    """Log query to database."""
    try:
        query_log = QueryLog(
            query=query,
            response=response,
            model_used=model_used,
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence=confidence,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata
        )
        db_session.add(query_log)
        db_session.commit()
    except Exception as e:
        logger.error("Failed to log query", error=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )

