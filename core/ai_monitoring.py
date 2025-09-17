#!/usr/bin/env python3
"""
📊 AI-POWERED MONITORING - DEEPBLUE 2.0 ULTIMATE UPGRADE
Intelligent monitoring with predictive analytics
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logger = structlog.get_logger()

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    response_time: float
    error_rate: float
    throughput: float
    active_connections: int
    cache_hit_rate: float

@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    alert_id: str
    alert_type: str
    severity: str
    description: str
    metrics: Dict[str, float]
    timestamp: datetime
    confidence: float

class AIMonitoringSystem:
    """AI-powered monitoring and analytics system."""
    
    def __init__(self):
        self.metrics_history = []
        self.anomaly_detector = None
        self.predictive_model = None
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
        # Prometheus metrics
        self.cpu_gauge = Gauge('deepblue2_cpu_usage_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('deepblue2_memory_usage_percent', 'Memory usage percentage')
        self.response_time_histogram = Histogram('deepblue2_response_time_seconds', 'Response time in seconds')
        self.error_counter = Counter('deepblue2_errors_total', 'Total number of errors')
        self.anomaly_counter = Counter('deepblue2_anomalies_total', 'Total number of anomalies detected')
        
        logger.info("📊 AI Monitoring System initializing...")
    
    async def initialize(self) -> bool:
        """Initialize the AI monitoring system."""
        try:
            # Initialize anomaly detection
            await self._initialize_anomaly_detection()
            
            # Initialize predictive analytics
            await self._initialize_predictive_analytics()
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            # Start anomaly detection loop
            asyncio.create_task(self._anomaly_detection_loop())
            
            logger.info("✅ AI Monitoring System initialized")
            return True
            
        except Exception as e:
            logger.error("❌ AI Monitoring System initialization failed", error=str(e))
            return False
    
    async def _initialize_anomaly_detection(self):
        """Initialize anomaly detection models."""
        # Isolation Forest for outlier detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Standard scaler for feature normalization
        self.scaler = StandardScaler()
        
        logger.info("Anomaly detection models initialized")
    
    async def _initialize_predictive_analytics(self):
        """Initialize predictive analytics models."""
        # This would initialize ML models for prediction
        # For now, we'll use simple statistical methods
        self.predictive_model = "statistical"
        
        logger.info("Predictive analytics models initialized")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                await self._store_metrics(metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Keep only last 10000 metrics
                if len(self.metrics_history) > 10000:
                    self.metrics_history = self.metrics_history[-10000:]
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_io=self._get_disk_io(),
            network_io=self._get_network_io(),
            response_time=self._get_avg_response_time(),
            error_rate=self._get_error_rate(),
            throughput=self._get_throughput(),
            active_connections=self._get_active_connections(),
            cache_hit_rate=self._get_cache_hit_rate()
        )
    
    def _get_disk_io(self) -> float:
        """Get disk I/O usage."""
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters:
                return io_counters.read_bytes + io_counters.write_bytes
            return 0.0
        except:
            return 0.0
    
    def _get_network_io(self) -> float:
        """Get network I/O usage."""
        try:
            io_counters = psutil.net_io_counters()
            if io_counters:
                return io_counters.bytes_sent + io_counters.bytes_recv
            return 0.0
        except:
            return 0.0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time."""
        # This would be calculated from actual request logs
        return 0.1  # Simulated
    
    def _get_error_rate(self) -> float:
        """Get current error rate."""
        # This would be calculated from actual error logs
        return 0.01  # Simulated 1% error rate
    
    def _get_throughput(self) -> float:
        """Get current throughput."""
        # This would be calculated from actual request logs
        return 100.0  # Simulated 100 requests/second
    
    def _get_active_connections(self) -> int:
        """Get active connections count."""
        # This would be calculated from actual connection logs
        return 50  # Simulated
    
    def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        # This would be calculated from actual cache logs
        return 0.85  # Simulated 85% hit rate
    
    async def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in Redis for real-time access."""
        key = f"metrics:{int(metrics.timestamp.timestamp())}"
        data = {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "disk_io": metrics.disk_io,
            "network_io": metrics.network_io,
            "response_time": metrics.response_time,
            "error_rate": metrics.error_rate,
            "throughput": metrics.throughput,
            "active_connections": metrics.active_connections,
            "cache_hit_rate": metrics.cache_hit_rate
        }
        
        self.redis_client.setex(key, 3600, json.dumps(data))  # Store for 1 hour
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics."""
        self.cpu_gauge.set(metrics.cpu_usage)
        self.memory_gauge.set(metrics.memory_usage)
        self.response_time_histogram.observe(metrics.response_time)
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection loop."""
        while True:
            try:
                if len(self.metrics_history) > 100:  # Need enough data
                    await self._detect_anomalies()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("Anomaly detection error", error=str(e))
                await asyncio.sleep(300)
    
    async def _detect_anomalies(self):
        """Detect anomalies in system metrics."""
        if len(self.metrics_history) < 100:
            return
        
        # Prepare data for anomaly detection
        data = []
        for metrics in self.metrics_history[-1000:]:  # Use last 1000 data points
            data.append([
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_io,
                metrics.network_io,
                metrics.response_time,
                metrics.error_rate,
                metrics.throughput,
                metrics.active_connections,
                metrics.cache_hit_rate
            ])
        
        data = np.array(data)
        
        # Normalize data
        data_scaled = self.scaler.fit_transform(data)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(data_scaled)
        
        # Process anomalies
        for i, label in enumerate(anomaly_labels):
            if label == -1:  # Anomaly detected
                await self._handle_anomaly(self.metrics_history[-1000:][i])
    
    async def _handle_anomaly(self, metrics: SystemMetrics):
        """Handle detected anomaly."""
        alert = AnomalyAlert(
            alert_id=f"anomaly_{int(time.time())}",
            alert_type="performance_anomaly",
            severity=self._calculate_severity(metrics),
            description=f"Anomaly detected in system metrics at {metrics.timestamp}",
            metrics={
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "response_time": metrics.response_time,
                "error_rate": metrics.error_rate
            },
            timestamp=metrics.timestamp,
            confidence=0.85
        )
        
        # Log anomaly
        logger.warning("Anomaly detected", alert=alert.__dict__)
        
        # Update Prometheus counter
        self.anomaly_counter.inc()
        
        # Store alert
        await self._store_alert(alert)
    
    def _calculate_severity(self, metrics: SystemMetrics) -> str:
        """Calculate alert severity."""
        if (metrics.cpu_usage > 90 or 
            metrics.memory_usage > 90 or 
            metrics.error_rate > 0.1):
            return "critical"
        elif (metrics.cpu_usage > 80 or 
              metrics.memory_usage > 80 or 
              metrics.error_rate > 0.05):
            return "warning"
        else:
            return "info"
    
    async def _store_alert(self, alert: AnomalyAlert):
        """Store alert in Redis."""
        key = f"alert:{alert.alert_id}"
        self.redis_client.setex(key, 86400, json.dumps(alert.__dict__, default=str))  # Store for 24 hours
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
        
        latest = self.metrics_history[-1]
        recent_metrics = self.metrics_history[-60:]  # Last 60 data points
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        response_time_trend = self._calculate_trend([m.response_time for m in recent_metrics])
        
        # Calculate health score
        health_score = self._calculate_health_score(latest)
        
        return {
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "critical",
            "health_score": health_score,
            "current_metrics": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "response_time": latest.response_time,
                "error_rate": latest.error_rate,
                "throughput": latest.throughput,
                "active_connections": latest.active_connections,
                "cache_hit_rate": latest.cache_hit_rate
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend,
                "response_time_trend": response_time_trend
            },
            "recommendations": self._generate_recommendations(latest),
            "timestamp": latest.timestamp.isoformat()
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_health_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall health score (0-1)."""
        # Weighted health score calculation
        cpu_score = max(0, 1 - metrics.cpu_usage / 100)
        memory_score = max(0, 1 - metrics.memory_usage / 100)
        response_score = max(0, 1 - metrics.response_time / 1.0)  # Assuming 1s is max acceptable
        error_score = max(0, 1 - metrics.error_rate / 0.1)  # Assuming 10% is max acceptable
        
        # Weighted average
        health_score = (
            cpu_score * 0.3 +
            memory_score * 0.3 +
            response_score * 0.2 +
            error_score * 0.2
        )
        
        return min(1.0, max(0.0, health_score))
    
    def _generate_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Generate system optimization recommendations."""
        recommendations = []
        
        if metrics.cpu_usage > 80:
            recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        if metrics.memory_usage > 80:
            recommendations.append("Consider increasing memory allocation or optimizing memory usage")
        
        if metrics.response_time > 0.5:
            recommendations.append("Consider optimizing database queries or implementing caching")
        
        if metrics.error_rate > 0.05:
            recommendations.append("Investigate error sources and implement better error handling")
        
        if metrics.cache_hit_rate < 0.7:
            recommendations.append("Consider optimizing cache strategy or increasing cache size")
        
        return recommendations
    
    async def get_predictive_analytics(self) -> Dict[str, Any]:
        """Get predictive analytics and forecasts."""
        if len(self.metrics_history) < 100:
            return {"error": "Insufficient data for predictions"}
        
        # Simple forecasting using moving averages
        recent_data = self.metrics_history[-100:]
        
        # Predict next hour
        cpu_forecast = self._forecast_metric([m.cpu_usage for m in recent_data], 12)  # 12 * 5min = 1hour
        memory_forecast = self._forecast_metric([m.memory_usage for m in recent_data], 12)
        response_time_forecast = self._forecast_metric([m.response_time for m in recent_data], 12)
        
        return {
            "forecasts": {
                "cpu_usage_1h": cpu_forecast,
                "memory_usage_1h": memory_forecast,
                "response_time_1h": response_time_forecast
            },
            "confidence": 0.75,  # Simulated confidence
            "timestamp": datetime.now().isoformat()
        }
    
    def _forecast_metric(self, values: List[float], periods: int) -> List[float]:
        """Simple forecasting using moving average."""
        if len(values) < 10:
            return values[-1:] * periods
        
        # Use exponential moving average
        alpha = 0.3
        forecast = []
        last_value = values[-1]
        
        for _ in range(periods):
            # Simple trend-based forecast
            trend = np.mean(np.diff(values[-10:])) if len(values) >= 10 else 0
            last_value = last_value + trend
            forecast.append(max(0, min(100, last_value)))  # Clamp between 0 and 100
        
        return forecast

# Global AI monitoring system
ai_monitoring = AIMonitoringSystem()

async def main():
    """Main function for testing."""
    if await ai_monitoring.initialize():
        logger.info("📊 AI Monitoring System is ready!")
        
        # Wait for some data to be collected
        await asyncio.sleep(10)
        
        # Get system health
        health = await ai_monitoring.get_system_health()
        print(f"System Health: {health}")
        
        # Get predictive analytics
        predictions = await ai_monitoring.get_predictive_analytics()
        print(f"Predictions: {predictions}")
    else:
        logger.error("❌ AI Monitoring System failed to initialize")

if __name__ == "__main__":
    asyncio.run(main())

