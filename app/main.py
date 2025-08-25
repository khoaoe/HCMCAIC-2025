import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.dependencies import get_app_settings
from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting, REPO_ROOT
from models.keyframe import Keyframe
from factory.factory import ServiceFactory
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from router import keyframe_api, agent_api, temporal_search_api
from factory.competition_factory import (
    CompetitionFactory,
    validate_competition_setup
)

# Global application state
app_state: Dict[str, Any] = {}



@asynccontextmanager
async def lifespan(app: FastAPI):
    """application lifespan with comprehensive initialization"""
    
    # Startup
    print("🚀 Starting HCMC AI Challenge 2025 System...")
    
    try:
        settings = get_app_settings()

        # Initialize MongoDB and Beanie (for repositories)
        mongo_settings = MongoDBSettings()
        milvus_settings = KeyFrameIndexMilvusSetting()
        if mongo_settings.MONGO_URI:
            mongo_connection_string = mongo_settings.MONGO_URI
        else:
            mongo_connection_string = (
                f"mongodb://{mongo_settings.MONGO_USER}:{mongo_settings.MONGO_PASSWORD}"
                f"@{mongo_settings.MONGO_HOST}:{mongo_settings.MONGO_PORT}/?authSource=admin"
            )
        mongo_client = AsyncIOMotorClient(mongo_connection_string)
        await mongo_client.admin.command('ping')
        database = mongo_client[mongo_settings.MONGO_DB]
        await init_beanie(
            database=database,
            document_models=[Keyframe]
        )
        
        # Validate competition setup
        validation_result = validate_competition_setup(
            data_folder=getattr(settings, 'DATA_FOLDER', '/data/keyframes'),
            objects_file=getattr(settings, 'OBJECTS_FILE', '/data/objects.json'),
            asr_file=getattr(settings, 'ASR_FILE', '/data/asr.json'),
            video_metadata_folder=getattr(settings, 'VIDEO_METADATA_FOLDER', None)
        )
        
        if not validation_result["valid"]:
            print("❌ Competition setup validation failed:")
            for error in validation_result["errors"]:
                print(f"   - {error}")
            # Downgrade to warning so the app can still start; routes may return warnings
            print("⚠️  Continuing startup with limited functionality. Please fix the setup above.")
        
        if validation_result["warnings"]:
            print("⚠️  Setup warnings:")
            for warning in validation_result["warnings"]:
                print(f"   - {warning}")
        
        print("✅ Setup validation complete:")
        stats = validation_result["data_stats"]
        print(f"   - Keyframes: {stats.get('keyframe_count', 0)}")
        print(f"   - Objects data: {'✓' if stats.get('objects_available') else '✗'}")
        print(f"   - ASR data: {'✓' if stats.get('asr_available') else '✗'}")
        print(f"   - Metadata: {'✓' if stats.get('metadata_available') else '✗'}")
        
        # Create competition system
        print("🔧 Initializing competition system...")
        
        factory = CompetitionFactory(settings)
        
        competition_system = factory.create_full_competition_system(
            data_folder=getattr(settings, 'DATA_FOLDER', str(REPO_ROOT / 'resources' / 'keyframes')),
            video_metadata_path=getattr(settings, 'VIDEO_METADATA_FOLDER', None),
            objects_file_path=getattr(settings, 'OBJECTS_FILE', '/data/objects.json'),
            asr_file_path=getattr(settings, 'ASR_FILE', '/data/asr.json'),
            optimization_profile=getattr(settings, 'OPTIMIZATION_PROFILE', 'balanced')
        )
        
        # Store in app state
        app_state.update({
            "competition_system": competition_system,
            "factory": factory,
            "settings": settings,
            "validation_result": validation_result
        })

        # Initialize shared ServiceFactory and attach to app state for legacy dependencies
        # Ensure reasonable default search params for Milvus
        milvus_search_params = {
            "metric_type": milvus_settings.METRIC_TYPE or "COSINE",
            "params": milvus_settings.SEARCH_PARAMS or {"nprobe": 16}
        }
        service_factory = ServiceFactory(
            milvus_collection_name=milvus_settings.COLLECTION_NAME,
            milvus_host=milvus_settings.HOST,
            milvus_port=milvus_settings.PORT,
            milvus_user="",
            milvus_password="",
            milvus_search_params=milvus_search_params,
            model_name=settings.MODEL_NAME,
            pretrained_weights=settings.PRETRAINED_WEIGHTS,
            use_pretrained=settings.USE_PRETRAINED,
            mongo_collection=Keyframe
        )
        app.state.service_factory = service_factory
        app.state.mongo_client = mongo_client
        
        # Include router
        app.include_router(competition_system["router"]) 
        # Backward-compatible routes for existing clients (e.g., /api/v1/keyframe/*)
        app.include_router(keyframe_api.router, prefix="/api/v1")
        app.include_router(agent_api.router, prefix='/api/v1')
        app.include_router(temporal_search_api.router, prefix="/api/v1")
        
        print("✅ Competition system initialized successfully!")
        print(f"   - Available tasks: {len(competition_system['config']['available_tasks'])}")
        print(f"   - Optimization profile: {competition_system['config']['optimization_profile']}")
        
        # Warm up system with test queries
        await _warm_up_system(competition_system)
        
        print("🎯 System ready for HCMC AI Challenge 2025!")
        
        yield
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        raise
    
    # Shutdown
    print("🛑 Shutting down competition system...")
    
    # Cleanup resources
    if "competition_system" in app_state:
        # Save performance metrics
        controller = app_state["competition_system"]["controller"]
        metrics = controller.get_performance_metrics()
        
        print("📊 Final performance metrics:")
        print(f"   - System status: {metrics.get('system_status', 'unknown')}")
        
        task_stats = metrics.get("controller_performance", {}).get("task_statistics", {})
        for task_type, stats in task_stats.items():
            if stats.get("count", 0) > 0:
                print(f"   - {task_type}: {stats['count']} requests, "
                      f"{stats['success_rate']:.2%} success rate, "
                      f"{stats['avg_time']:.2f}s avg time")
    
    print("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="HCMC AI Challenge 2025 - System",
        description="Advanced multimodal video retrieval and QA system with competition optimizations",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # root endpoint
    @app.get("/")
    async def root():
        """ root endpoint with system information"""
        
        if "competition_system" not in app_state:
            return {"status": "initializing", "message": "System is starting up..."}
        
        system = app_state["competition_system"]
        controller = system["controller"]
        
        # Get current metrics
        metrics = controller.get_performance_metrics()
        
        return {
            "status": "operational",
            "system": "HCMC AI Challenge 2025",
            "version": "2.0.0",
            "features": [
                "Advanced multimodal video retrieval",
                "Temporal localization",
                "Intelligent query processing",
                "Real-time interactive feedback",
                "Performance optimization",
                "Comprehensive evidence tracking"
            ],
            "available_tasks": system["config"]["available_tasks"],
            "optimization_profile": system["config"]["optimization_profile"],
            "performance": {
                "system_status": metrics.get("system_status", "unknown"),
                "active_sessions": metrics.get("controller_performance", {}).get("active_sessions", 0),
                "total_requests": sum(
                    stats.get("count", 0) 
                    for stats in metrics.get("controller_performance", {}).get("task_statistics", {}).values()
                )
            },
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/competition/v2/health",
                "performance_metrics": "/competition/v2/metrics",
                "vcmr_automatic": "/competition/v2/vcmr/automatic",
                "vcmr_interactive": "/competition/v2/vcmr/interactive",
                "video_qa": "/competition/v2/vqa",
                "kis_textual": "/competition/v2/kis/textual",
                "kis_visual": "/competition/v2/kis/visual",
                "kis_progressive": "/competition/v2/kis/progressive"
            }
        }
    
    # status endpoint
    @app.get("/status")
    async def status():
        """ status endpoint with detailed system information"""
        
        if "competition_system" not in app_state:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "message": "System not initialized"}
            )
        
        system = app_state["competition_system"]
        controller = system["controller"]
        validation_result = app_state.get("validation_result", {})
        
        # Get comprehensive status
        metrics = controller.get_performance_metrics()
        
        status_info = {
            "system_status": metrics.get("system_status", "unknown"),
            "initialization": {
                "completed": True,
                "validation_passed": validation_result.get("valid", False),
                "data_stats": validation_result.get("data_stats", {}),
                "warnings": validation_result.get("warnings", [])
            },
            "components": {
                "agent": "operational",
                "controller": "operational",
                "router": "operational",
                "llm": "operational",
                "keyframe_service": "operational",
                "model_service": "operational"
            },
            "performance": metrics,
            "configuration": system["config"]
        }
        
        # Determine HTTP status code based on system health
        status_code = 200 if metrics.get("system_status") == "operational" else 503
        
        return JSONResponse(
            status_code=status_code,
            content=status_info
        )
    
    # System control endpoints
    @app.post("/admin/reset-performance-stats")
    async def reset_performance_stats():
        """Reset performance statistics (admin endpoint)"""
        
        if "competition_system" not in app_state:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        controller = app_state["competition_system"]["controller"]
        
        # Reset task statistics
        for task_type in controller.task_stats:
            controller.task_stats[task_type] = {
                "count": 0, 
                "avg_time": 0.0, 
                "success_rate": 0.0
            }
        
        # Reset agent performance stats
        agent = app_state["competition_system"]["agent"]
        if hasattr(agent, 'performance_optimizer'):
            agent.performance_optimizer.performance_stats = {
                "total_queries": 0,
                "cache_hits": 0,
                "avg_response_time": 0.0,
                "response_times": []
            }
        
        return {"message": "Performance statistics reset successfully"}
    
    @app.post("/admin/clear-all-sessions")
    async def clear_all_sessions():
        """Clear all interactive sessions (admin endpoint)"""
        
        if "competition_system" not in app_state:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        controller = app_state["competition_system"]["controller"]
        agent = app_state["competition_system"]["agent"]
        
        # Clear controller sessions
        session_count = len(controller.interactive_sessions)
        controller.interactive_sessions.clear()
        
        # Clear agent session state
        agent_session_count = len(agent.session_state)
        agent.session_state.clear()
        
        return {
            "message": "All sessions cleared successfully",
            "controller_sessions_cleared": session_count,
            "agent_sessions_cleared": agent_session_count
        }
    
    return app


async def _warm_up_system(competition_system: Dict[str, Any]):
    """Warm up the system with test queries"""
    
    print("🔥 Warming up system...")
    
    try:
        controller = competition_system["controller"]
        
        # Test VCMR
        from schema.competition import VCMRAutomaticRequest
        test_vcmr = VCMRAutomaticRequest(
            query="person walking",
            corpus_index="test",
            top_k=5
        )
        
        await controller.process_vcmr_automatic(test_vcmr)
        print("   ✓ VCMR warmed up")
        
        # Small delay between warmup tasks
        await asyncio.sleep(0.1)
        
        print("✅ System warmup complete")
        
    except Exception as e:
        print(f"⚠️  Warmup failed (system will still work): {e}")


# Create the application
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
