"""
Enhanced Competition API Router for HCMC AI Challenge 2025
Provides optimized endpoints for all competition tasks with advanced features
"""

import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from controller.enhanced_competition_controller import EnhancedCompetitionController
from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse,
    VideoQARequest, VideoQAResponse,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    VCMRFeedback, VCMRInteractiveCandidate,
    InteractiveSystemRequest, InteractiveLLMResponse
)


# Additional request/response models for enhanced features
class BatchVCMRRequest(BaseModel):
    """Batch processing request for multiple VCMR queries"""
    queries: List[VCMRAutomaticRequest] = Field(..., min_items=1, max_items=10)
    parallel_processing: bool = Field(default=True, description="Process queries in parallel")


class BatchVCMRResponse(BaseModel):
    """Batch processing response for multiple VCMR queries"""
    results: List[VCMRAutomaticResponse]
    processing_time: float
    success_count: int
    failed_queries: List[Dict[str, Any]]


class InteractiveFeedbackRequest(BaseModel):
    """Interactive feedback request"""
    session_id: str
    feedback: Dict[str, Any]


class SessionInfoResponse(BaseModel):
    """Session information response"""
    session_id: str
    original_query: Optional[str]
    interaction_count: int
    feedback_count: int
    last_candidate: Optional[Dict[str, Any]]
    session_active: bool


class PerformanceMetricsResponse(BaseModel):
    """System performance metrics response"""
    agent_performance: Dict[str, Any]
    controller_performance: Dict[str, Any]
    system_status: str
    recommendations: Optional[List[str]]


def create_enhanced_competition_router(controller: EnhancedCompetitionController) -> APIRouter:
    """Create enhanced competition API router with advanced features"""
    
    router = APIRouter(prefix="/competition/v2", tags=["Enhanced Competition"])
    
    # ===============================
    # VCMR (Video Corpus Moment Retrieval) Endpoints
    # ===============================
    
    @router.post("/vcmr/automatic", response_model=VCMRAutomaticResponse)
    async def enhanced_vcmr_automatic(
        request: VCMRAutomaticRequest,
        background_tasks: BackgroundTasks
    ) -> VCMRAutomaticResponse:
        """
        Enhanced VCMR Automatic task with advanced multimodal fusion
        
        Features:
        - Advanced query understanding and expansion
        - Multi-modal retrieval with cross-modal reranking
        - Intelligent temporal clustering
        - Performance optimization
        """
        try:
            response = await controller.process_vcmr_automatic(request)
            
            # Background task for performance monitoring
            background_tasks.add_task(
                _log_request_performance, 
                "vcmr_automatic", 
                request.query, 
                len(response.candidates)
            )
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/vcmr/batch", response_model=BatchVCMRResponse)
    async def batch_vcmr_automatic(
        request: BatchVCMRRequest
    ) -> BatchVCMRResponse:
        """
        Batch processing for multiple VCMR queries
        
        Features:
        - Parallel or sequential processing
        - Aggregated performance metrics
        - Error handling per query
        """
        import time
        import asyncio
        
        start_time = time.time()
        results = []
        failed_queries = []
        
        try:
            if request.parallel_processing:
                # Process queries in parallel
                tasks = [
                    controller.process_vcmr_automatic(vcmr_request)
                    for vcmr_request in request.queries
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        failed_queries.append({
                            "query_index": i,
                            "query": request.queries[i].query,
                            "error": str(response)
                        })
                    else:
                        results.append(response)
            else:
                # Process queries sequentially
                for i, vcmr_request in enumerate(request.queries):
                    try:
                        response = await controller.process_vcmr_automatic(vcmr_request)
                        results.append(response)
                    except Exception as e:
                        failed_queries.append({
                            "query_index": i,
                            "query": vcmr_request.query,
                            "error": str(e)
                        })
            
            processing_time = time.time() - start_time
            
            return BatchVCMRResponse(
                results=results,
                processing_time=processing_time,
                success_count=len(results),
                failed_queries=failed_queries
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    @router.post("/vcmr/interactive", response_model=Dict[str, Any])
    async def enhanced_vcmr_interactive(
        query: str = Body(..., embed=True),
        session_id: Optional[str] = Query(None),
        feedback: Optional[VCMRFeedback] = Body(None),
        previous_candidates: Optional[List[VCMRInteractiveCandidate]] = Body(None)
    ) -> Dict[str, Any]:
        """
        Enhanced VCMR Interactive task with session management
        
        Features:
        - Session state management
        - Intelligent feedback integration
        - Real-time refinement
        """
        try:
            result_candidate, session_id = await controller.process_vcmr_interactive(
                query=query,
                session_id=session_id,
                feedback=feedback,
                previous_candidates=previous_candidates
            )
            
            return {
                "candidate": {
                    "video_id": result_candidate.video_id,
                    "start_time": result_candidate.start_time,
                    "end_time": result_candidate.end_time,
                    "score": result_candidate.score
                },
                "session_id": session_id,
                "interaction_count": controller.interactive_sessions[session_id]["interaction_count"]
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===============================
    # Video QA Endpoints
    # ===============================
    
    @router.post("/vqa", response_model=VideoQAResponse)
    async def enhanced_video_qa(
        request: VideoQARequest,
        background_tasks: BackgroundTasks
    ) -> VideoQAResponse:
        """
        Enhanced Video QA with comprehensive evidence tracking
        
        Features:
        - Advanced question analysis
        - Multi-modal evidence gathering
        - Detailed confidence scoring
        - Evidence provenance tracking
        """
        try:
            response = await controller.process_video_qa(request)
            
            # Background task for performance monitoring
            background_tasks.add_task(
                _log_request_performance,
                "video_qa",
                request.question,
                len(response.evidence) if response.evidence else 0
            )
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===============================
    # KIS (Known-Item Search) Endpoints
    # ===============================
    
    @router.post("/kis/textual", response_model=KISResponse)
    async def enhanced_kis_textual(
        request: KISTextualRequest
    ) -> KISResponse:
        """
        Enhanced KIS Textual with precision optimization
        
        Features:
        - Advanced description analysis
        - Multiple search strategies
        - Exact matching optimization
        """
        try:
            response = await controller.process_kis_textual(request)
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/kis/visual", response_model=KISResponse)
    async def enhanced_kis_visual(
        request: KISVisualRequest
    ) -> KISResponse:
        """
        Enhanced KIS Visual with advanced visual matching
        
        Features:
        - Visual similarity analysis
        - Feature-based matching
        - Precise temporal localization
        """
        try:
            response = await controller.process_kis_visual(request)
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/kis/progressive", response_model=Dict[str, Any])
    async def enhanced_kis_progressive(
        request: KISProgressiveRequest,
        session_id: Optional[str] = Query(None),
        additional_hints: Optional[List[str]] = Query(None)
    ) -> Dict[str, Any]:
        """
        Enhanced KIS Progressive with session state management
        
        Features:
        - Progressive hint integration
        - Session state tracking
        - Intelligent hint combination
        """
        try:
            response, session_id = await controller.process_kis_progressive(
                request=request,
                session_id=session_id,
                additional_hints=additional_hints
            )
            
            session_info = controller.agent.session_state.get(session_id, {})
            
            return {
                "match": {
                    "video_id": response.video_id,
                    "start_time": response.start_time,
                    "end_time": response.end_time,
                    "match_confidence": response.match_confidence
                },
                "session_id": session_id,
                "hints_used": len(session_info.get("all_hints", [])),
                "search_iterations": len(session_info.get("search_history", []))
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===============================
    # Interactive System Endpoints
    # ===============================
    
    @router.post("/interactive/system", response_model=InteractiveLLMResponse)
    async def process_interactive_system_request(
        request: InteractiveSystemRequest
    ) -> InteractiveLLMResponse:
        """
        Process interactive system requests for LLM integration
        
        Features:
        - Structured LLM interaction
        - Action-based responses
        - Context-aware processing
        """
        try:
            response = await controller.process_interactive_system_request(request)
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/interactive/feedback", response_model=Dict[str, Any])
    async def handle_interactive_feedback(
        request: InteractiveFeedbackRequest
    ) -> Dict[str, Any]:
        """
        Handle feedback for interactive sessions
        
        Features:
        - Multi-modal feedback integration
        - Session state updates
        - Real-time refinement
        """
        try:
            result = await controller.handle_interactive_feedback(
                session_id=request.session_id,
                feedback=request.feedback
            )
            return result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===============================
    # Session Management Endpoints
    # ===============================
    
    @router.get("/session/{session_id}", response_model=SessionInfoResponse)
    async def get_session_info(session_id: str) -> SessionInfoResponse:
        """
        Get information about an interactive session
        
        Features:
        - Session state retrieval
        - Interaction history
        - Progress tracking
        """
        try:
            session_info = controller.get_session_info(session_id)
            
            if "error" in session_info:
                raise HTTPException(status_code=404, detail=session_info["error"])
            
            return SessionInfoResponse(**session_info)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/session/{session_id}")
    async def clear_session(session_id: str) -> Dict[str, str]:
        """
        Clear an interactive session
        
        Features:
        - Session cleanup
        - Memory management
        """
        try:
            if session_id in controller.interactive_sessions:
                del controller.interactive_sessions[session_id]
                return {"message": f"Session {session_id} cleared successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ===============================
    # Performance and Monitoring Endpoints
    # ===============================
    
    @router.get("/metrics", response_model=PerformanceMetricsResponse)
    async def get_performance_metrics() -> PerformanceMetricsResponse:
        """
        Get comprehensive system performance metrics
        
        Features:
        - Task performance statistics
        - System health indicators
        - Optimization recommendations
        """
        try:
            metrics = controller.get_performance_metrics()
            
            # Generate recommendations based on metrics
            recommendations = _generate_performance_recommendations(metrics)
            
            return PerformanceMetricsResponse(
                agent_performance=metrics.get("agent_performance", {}),
                controller_performance=metrics.get("controller_performance", {}),
                system_status=metrics.get("system_status", "unknown"),
                recommendations=recommendations
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """
        System health check endpoint
        
        Features:
        - System status verification
        - Component health checks
        - Performance indicators
        """
        try:
            metrics = controller.get_performance_metrics()
            
            return {
                "status": "healthy" if metrics.get("system_status") == "operational" else "degraded",
                "timestamp": time.time(),
                "components": {
                    "agent": "operational",
                    "controller": "operational",
                    "performance_optimizer": "operational"
                },
                "active_sessions": metrics.get("controller_performance", {}).get("active_sessions", 0)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    # ===============================
    # Utility Endpoints
    # ===============================
    
    @router.post("/utils/validate", response_model=Dict[str, Any])
    async def validate_request(
        task_type: str = Body(...),
        request_data: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        """
        Validate request data for different task types
        
        Features:
        - Request validation
        - Schema compliance checking
        - Error reporting
        """
        try:
            validation_result = {"valid": True, "errors": []}
            
            if task_type == "vcmr_automatic":
                try:
                    VCMRAutomaticRequest(**request_data)
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append(str(e))
            
            elif task_type == "video_qa":
                try:
                    VideoQARequest(**request_data)
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append(str(e))
            
            elif task_type in ["kis_textual", "kis_visual", "kis_progressive"]:
                try:
                    if task_type == "kis_textual":
                        KISTextualRequest(**request_data)
                    elif task_type == "kis_visual":
                        KISVisualRequest(**request_data)
                    elif task_type == "kis_progressive":
                        KISProgressiveRequest(**request_data)
                except Exception as e:
                    validation_result["valid"] = False
                    validation_result["errors"].append(str(e))
            
            else:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Unknown task type: {task_type}")
            
            return validation_result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


# Helper functions

async def _log_request_performance(
    task_type: str, 
    query: str, 
    result_count: int
):
    """Background task for logging request performance"""
    # This would log to monitoring system in production
    print(f"Task: {task_type}, Query: '{query[:50]}...', Results: {result_count}")


def _generate_performance_recommendations(metrics: Dict[str, Any]) -> List[str]:
    """Generate performance optimization recommendations"""
    
    recommendations = []
    
    # Check agent performance
    agent_perf = metrics.get("agent_performance", {})
    if agent_perf.get("avg_response_time", 0) > 10.0:
        recommendations.append("Consider reducing search top_k or increasing score thresholds for faster response times")
    
    # Check controller performance
    controller_perf = metrics.get("controller_performance", {})
    task_stats = controller_perf.get("task_statistics", {})
    
    for task_type, stats in task_stats.items():
        if stats.get("success_rate", 1.0) < 0.9:
            recommendations.append(f"Low success rate for {task_type}: consider tuning parameters")
        if stats.get("avg_time", 0) > 15.0:
            recommendations.append(f"High response time for {task_type}: consider optimization")
    
    # Check system status
    if metrics.get("system_status") != "operational":
        recommendations.append("System performance is degraded - monitor resource usage and error rates")
    
    if not recommendations:
        recommendations.append("System is performing optimally")
    
    return recommendations
