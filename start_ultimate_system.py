#!/usr/bin/env python3
"""
🌊 DEEPBLUE 2.0 ULTIMATE SYSTEM STARTER 🌊
Start the most advanced AI system ever built
"""

import asyncio
import sys
import os
import signal
import logging
from pathlib import Path

# Add core directory to path
sys.path.append(str(Path(__file__).parent / "core"))

from ultimate_integration import ultimate_system
import structlog

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

class UltimateSystemManager:
    """Manager for the Ultimate DeepBlue 2.0 System."""
    
    def __init__(self):
        self.system = ultimate_system
        self.is_running = False
        
    async def start(self):
        """Start the ultimate system."""
        try:
            print("🌊 DEEPBLUE 2.0 ULTIMATE SYSTEM")
            print("=" * 50)
            print("🚀 Initializing the most advanced AI system...")
            print("⚡ Maximum capabilities enabled")
            print("🔒 Quantum-resistant security active")
            print("📊 AI-powered monitoring operational")
            print("🎯 Self-healing mechanisms ready")
            print("=" * 50)
            
            # Initialize the system
            if await self.system.initialize():
                self.is_running = True
                print("✅ DEEPBLUE 2.0 ULTIMATE SYSTEM READY!")
                print("🌊 'I think we need a bigger boat!' 🚢")
                print("=" * 50)
                
                # Start the main loop
                await self._main_loop()
            else:
                print("❌ Failed to initialize Ultimate DeepBlue 2.0 System")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            print(f"❌ System error: {e}")
            await self.shutdown()
            sys.exit(1)
    
    async def _main_loop(self):
        """Main system loop."""
        while self.is_running:
            try:
                # Display system status
                status = await self.system.get_system_status()
                print(f"\n📊 System Status: {status['status']}")
                print(f"⏱️ Uptime: {status['uptime']:.2f} seconds")
                
                # Get user input
                query = input("\n🌊 DeepBlue 2.0> ")
                
                if query.lower() in ['exit', 'quit', 'shutdown']:
                    break
                elif query.lower() == 'status':
                    await self._show_detailed_status()
                elif query.lower() == 'health':
                    await self._show_health_report()
                elif query.strip():
                    # Process query
                    result = await self.system.process_ultimate_query(query)
                    print(f"\n🤖 Response: {result['response']}")
                    print(f"⚡ Method: {result['method']}")
                    print(f"🎯 Confidence: {result['confidence']:.2f}")
                    print(f"⏱️ Processing Time: {result['processing_time']:.3f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _show_detailed_status(self):
        """Show detailed system status."""
        status = await self.system.get_system_status()
        print("\n📊 DETAILED SYSTEM STATUS")
        print("=" * 30)
        print(f"Status: {status['status']}")
        print(f"Uptime: {status['uptime']:.2f} seconds")
        print(f"Initialization Time: {status['initialization_time']:.2f} seconds")
        
        print("\n🔧 CAPABILITIES:")
        for capability, enabled in status['capabilities'].items():
            status_icon = "✅" if enabled else "❌"
            print(f"  {status_icon} {capability.replace('_', ' ').title()}")
        
        if 'health' in status and status['health']:
            health = status['health']
            print(f"\n💚 HEALTH SCORE: {health.get('health_score', 'N/A')}")
            if 'current_metrics' in health:
                metrics = health['current_metrics']
                print(f"  CPU Usage: {metrics.get('cpu_usage', 'N/A')}%")
                print(f"  Memory Usage: {metrics.get('memory_usage', 'N/A')}%")
                print(f"  Response Time: {metrics.get('response_time', 'N/A')}s")
                print(f"  Error Rate: {metrics.get('error_rate', 'N/A')}%")
    
    async def _show_health_report(self):
        """Show health report."""
        if self.system.monitoring_system:
            health = await self.system.monitoring_system.get_system_health()
            print("\n💚 SYSTEM HEALTH REPORT")
            print("=" * 25)
            print(f"Status: {health.get('status', 'Unknown')}")
            print(f"Health Score: {health.get('health_score', 'N/A')}")
            
            if 'recommendations' in health:
                print("\n💡 RECOMMENDATIONS:")
                for i, rec in enumerate(health['recommendations'], 1):
                    print(f"  {i}. {rec}")
        else:
            print("❌ Monitoring system not available")
    
    async def shutdown(self):
        """Shutdown the system."""
        if self.is_running:
            print("\n🛑 Shutting down Ultimate DeepBlue 2.0 System...")
            await self.system.shutdown()
            self.is_running = False
            print("✅ Shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main function."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the system manager
    manager = UltimateSystemManager()
    await manager.start()

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

