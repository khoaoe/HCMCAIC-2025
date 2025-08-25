"""
 Competition Controller for HCMC AI Challenge 2025
Provides optimized task routing and session management for all competition tasks
"""

from typing import Dict, Any, Optional, List, Tuple
import asyncio
import time
import uuid
from pathlib import Path

from fastapi import HTTPException
from llama_index.core.llms import LLM

from agent.competition_tasks import VCMRTaskProcessor
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse,
    VideoQARequest, VideoQAResponse,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    VCMRFeedback, VCMRInteractiveCandidate,
    InteractiveSystemRequest, InteractiveLLMResponse
)
# from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting
# from factory.factory import ServiceFactory
from models.keyframe import Keyframe


class CompetitionController:
    """ controller with advanced task routing and optimization"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: Dict[str, List[str]],
        asr_data: Dict[str, Any],
        video_metadata_path: Optional[Path] = None
    ):
        # Initialize  agent
        self.agent = VCMRTaskProcessor(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            video_metadata_path=video_metadata_path
        )
        
        # Session management for interactive tasks
        self.interactive_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.task_stats = {
            "vcmr_automatic": {"count": 0, "avg_time": 0.0, "success_rate": 0.0},
            "video_qa": {"count": 0, "avg_time": 0.0, "success_rate": 0.0},
            "kis_textual": {"count": 0, "avg_time": 0.0, "success_rate": 0.0},
            "kis_visual": {"count": 0, "avg_time": 0.0, "success_rate": 0.0},
            "kis_progressive": {"count": 0, "avg_time": 0.0, "success_rate": 0.0},
            "vcmr_interactive": {"count": 0, "avg_time": 0.0, "success_rate": 0.0}
        }
    
    async def process_vcmr_automatic(
        self, 
        request: VCMRAutomaticRequest
    ) -> VCMRAutomaticResponse:
        """Process VCMR Automatic task with performance tracking"""
        
        start_time = time.time()
        task_type = "vcmr_automatic"
        
        try:
            # Validate request
            self._validate_vcmr_request(request)
            
            # Process with  agent
            response = await self.agent.vcmr_automatic(request)
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return response
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"VCMR Automatic processing failed: {str(e)}"
            )
    
    async def process_video_qa(
        self, 
        request: VideoQARequest
    ) -> VideoQAResponse:
        """Process Video QA task with  context handling"""
        
        start_time = time.time()
        task_type = "video_qa"
        
        try:
            # Validate request
            self._validate_video_qa_request(request)
            
            # Process with  agent
            response = await self.agent.video_qa(request)
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return response
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"Video QA processing failed: {str(e)}"
            )
    
    async def process_kis_textual(
        self, 
        request: KISTextualRequest
    ) -> KISResponse:
        """Process KIS Textual task with precision optimization"""
        
        start_time = time.time()
        task_type = "kis_textual"
        
        try:
            # Validate request
            self._validate_kis_textual_request(request)
            
            # Process with  agent
            response = await self.agent.kis_textual(request)
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return response
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"KIS Textual processing failed: {str(e)}"
            )
    
    async def process_kis_visual(
        self, 
        request: KISVisualRequest
    ) -> KISResponse:
        """Process KIS Visual task with advanced visual matching"""
        
        start_time = time.time()
        task_type = "kis_visual"
        
        try:
            # Validate request
            self._validate_kis_visual_request(request)
            
            # Process with  agent
            response = await self.agent.kis_visual(request)
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return response
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"KIS Visual processing failed: {str(e)}"
            )
    
    async def process_kis_progressive(
        self, 
        request: KISProgressiveRequest,
        session_id: Optional[str] = None,
        additional_hints: Optional[List[str]] = None
    ) -> Tuple[KISResponse, str]:
        """Process KIS Progressive task with session management"""
        
        start_time = time.time()
        task_type = "kis_progressive"
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Validate request
            self._validate_kis_progressive_request(request)
            
            # Process with  agent
            response = await self.agent.kis_progressive(
                request, session_id, additional_hints
            )
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return response, session_id
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"KIS Progressive processing failed: {str(e)}"
            )
    
    async def process_vcmr_interactive(
        self,
        query: str,
        session_id: Optional[str] = None,
        feedback: Optional[VCMRFeedback] = None,
        previous_candidates: Optional[List[VCMRInteractiveCandidate]] = None
    ) -> Tuple[VCMRInteractiveCandidate, str]:
        """Process VCMR Interactive task with session state management"""
        
        start_time = time.time()
        task_type = "vcmr_interactive"
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Initialize session if new
            if session_id not in self.interactive_sessions:
                self.interactive_sessions[session_id] = {
                    "original_query": query,
                    "interaction_count": 0,
                    "feedback_history": [],
                    "candidate_history": []
                }
            
            session = self.interactive_sessions[session_id]
            session["interaction_count"] += 1
            
            # Store feedback if provided
            if feedback:
                session["feedback_history"].append({
                    "feedback": feedback,
                    "previous_candidates": previous_candidates,
                    "timestamp": time.time()
                })
            
            # Process with  interactive logic
            result_candidate = await self._process_interactive_vcmr(
                query, session, feedback, previous_candidates
            )
            
            # Store result in session
            session["candidate_history"].append(result_candidate)
            
            # Track success
            self._update_task_stats(task_type, start_time, success=True)
            
            return result_candidate, session_id
            
        except Exception as e:
            self._update_task_stats(task_type, start_time, success=False)
            raise HTTPException(
                status_code=500,
                detail=f"VCMR Interactive processing failed: {str(e)}"
            )
    
    async def process_interactive_system_request(
        self,
        request: InteractiveSystemRequest
    ) -> InteractiveLLMResponse:
        """Process interactive system requests for LLM integration"""
        
        try:
            task_type = request.task.lower()
            
            if task_type == "vcmr_interactive":
                # Handle VCMR interactive request
                candidate, session_id = await self.process_vcmr_interactive(
                    query=request.query,
                    session_id=request.context.get("session_id")
                )
                
                return InteractiveLLMResponse(
                    action="predict_moment",
                    payload={
                        "video_id": candidate.video_id,
                        "start_time": candidate.start_time,
                        "end_time": candidate.end_time,
                        "score": candidate.score,
                        "session_id": session_id
                    }
                )
            
            elif "kis" in task_type:
                # Handle KIS requests based on specific type
                if "visual" in task_type:
                    query_clip_uri = request.context.get("query_clip_uri", "")
                    kis_request = KISVisualRequest(
                        query_clip_uri=query_clip_uri,
                        corpus_index=request.context.get("corpus_index", "default")
                    )
                    response = await self.process_kis_visual(kis_request)
                elif "textual" in task_type:
                    text_description = request.context.get("text_description", request.query)
                    kis_request = KISTextualRequest(
                        text_description=text_description,
                        corpus_index=request.context.get("corpus_index", "default")
                    )
                    response = await self.process_kis_textual(kis_request)
                elif "progressive" in task_type:
                    kis_request = KISProgressiveRequest(
                        initial_hint=request.query,
                        corpus_index=request.context.get("corpus_index", "default")
                    )
                    response, session_id = await self.process_kis_progressive(
                        kis_request,
                        session_id=request.context.get("session_id")
                    )
                else:
                    raise ValueError(f"Unknown KIS task type: {task_type}")
                
                return InteractiveLLMResponse(
                    action="return_match",
                    payload={
                        "video_id": response.video_id,
                        "start_time": response.start_time,
                        "end_time": response.end_time,
                        "match_confidence": response.match_confidence
                    }
                )
            
            else:
                raise ValueError(f"Unsupported interactive task: {task_type}")
                
        except Exception as e:
            return InteractiveLLMResponse(
                action="return_explanation",
                payload={"error": f"Processing failed: {str(e)}"}
            )
    
    async def handle_interactive_feedback(
        self,
        session_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle feedback for interactive sessions"""
        
        try:
            if session_id not in self.interactive_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.interactive_sessions[session_id]
            task_type = session.get("task_type", "vcmr_interactive")
            
            if task_type == "vcmr_interactive":
                # Parse VCMR feedback
                vcmr_feedback = VCMRFeedback(**feedback)
                previous_candidates = session.get("candidate_history", [])
                
                # Process refined search
                result_candidate, _ = await self.process_vcmr_interactive(
                    query=session["original_query"],
                    session_id=session_id,
                    feedback=vcmr_feedback,
                    previous_candidates=previous_candidates
                )
                
                return {
                    "success": True,
                    "result": {
                        "video_id": result_candidate.video_id,
                        "start_time": result_candidate.start_time,
                        "end_time": result_candidate.end_time,
                        "score": result_candidate.score
                    }
                }
            
            elif task_type.startswith("kis"):
                # Handle KIS feedback (additional hints)
                if "additional_hint" in feedback:
                    # For KIS Progressive
                    kis_request = KISProgressiveRequest(
                        initial_hint=session.get("original_query", ""),
                        corpus_index="default"
                    )
                    
                    response, _ = await self.process_kis_progressive(
                        kis_request,
                        session_id=session_id,
                        additional_hints=[feedback["additional_hint"]]
                    )
                    
                    return {
                        "success": True,
                        "result": {
                            "video_id": response.video_id,
                            "start_time": response.start_time,
                            "end_time": response.end_time,
                            "match_confidence": response.match_confidence
                        }
                    }
            
            return {"success": False, "error": "Unsupported feedback type"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about an interactive session"""
        
        if session_id not in self.interactive_sessions:
            return {"error": "Session not found"}
        
        session = self.interactive_sessions[session_id]
        
        return {
            "session_id": session_id,
            "original_query": session.get("original_query"),
            "interaction_count": session.get("interaction_count", 0),
            "feedback_count": len(session.get("feedback_history", [])),
            "last_candidate": session.get("candidate_history", [])[-1] if session.get("candidate_history") else None,
            "session_active": True
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        # Agent performance metrics
        agent_metrics = self.agent.get_performance_metrics()
        
        # Controller-level metrics
        controller_metrics = {
            "task_statistics": self.task_stats,
            "active_sessions": len(self.interactive_sessions),
            "total_sessions": len(self.interactive_sessions)
        }
        
        # Combine metrics
        combined_metrics = {
            "agent_performance": agent_metrics,
            "controller_performance": controller_metrics,
            "system_status": "operational" if self._check_system_health() else "degraded"
        }
        
        return combined_metrics
    
    # Private helper methods
    
    async def _process_interactive_vcmr(
        self,
        query: str,
        session: Dict[str, Any],
        feedback: Optional[VCMRFeedback],
        previous_candidates: Optional[List[VCMRInteractiveCandidate]]
    ) -> VCMRInteractiveCandidate:
        """Process interactive VCMR with session context"""
        
        # Create automatic request for base search
        auto_request = VCMRAutomaticRequest(
            query=query,
            corpus_index="default",
            top_k=20  # Smaller for interactive speed
        )
        
        # Get automatic results
        auto_response = await self.agent.vcmr_automatic(auto_request)
        
        if not auto_response.candidates:
            raise ValueError("No candidates found for interactive query")
        
        # Apply feedback if provided
        if feedback and previous_candidates:
            filtered_candidates = self._apply_interactive_feedback(
                auto_response.candidates, feedback, previous_candidates
            )
            if filtered_candidates:
                top_candidate = filtered_candidates[0]
            else:
                top_candidate = auto_response.candidates[0]
        else:
            top_candidate = auto_response.candidates[0]
        
        # Convert to interactive format
        return VCMRInteractiveCandidate(
            video_id=top_candidate.video_id,
            start_time=top_candidate.start_time,
            end_time=top_candidate.end_time,
            score=top_candidate.score
        )
    
    def _apply_interactive_feedback(
        self,
        candidates: List[Any],
        feedback: VCMRFeedback,
        previous_candidates: List[VCMRInteractiveCandidate]
    ) -> List[Any]:
        """Apply interactive feedback to filter candidates"""
        
        if feedback.relevance is False:
            # Remove similar candidates to previous ones
            filtered = []
            for candidate in candidates:
                is_similar = False
                for prev in previous_candidates:
                    if (candidate.video_id == prev.video_id and
                        abs(candidate.start_time - prev.start_time) < 5.0):
                        is_similar = True
                        break
                if not is_similar:
                    filtered.append(candidate)
            return filtered
        
        elif feedback.relevance_score is not None:
            # Filter by score threshold
            threshold = feedback.relevance_score
            return [c for c in candidates if c.score >= threshold]
        
        elif feedback.refine:
            # This would trigger a new search with refined query
            # For now, return original candidates
            return candidates
        
        return candidates
    
    def _validate_vcmr_request(self, request: VCMRAutomaticRequest):
        """Validate VCMR request parameters"""
        if not request.query or len(request.query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        if request.top_k <= 0 or request.top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
    
    def _validate_video_qa_request(self, request: VideoQARequest):
        """Validate Video QA request parameters"""
        if not request.question or len(request.question.strip()) < 3:
            raise ValueError("Question must be at least 3 characters long")
        if not request.video_id:
            raise ValueError("video_id is required")
    
    def _validate_kis_textual_request(self, request: KISTextualRequest):
        """Validate KIS Textual request parameters"""
        if not request.text_description or len(request.text_description.strip()) < 5:
            raise ValueError("Text description must be at least 5 characters long")
    
    def _validate_kis_visual_request(self, request: KISVisualRequest):
        """Validate KIS Visual request parameters"""
        if not request.query_clip_uri:
            raise ValueError("query_clip_uri is required")
    
    def _validate_kis_progressive_request(self, request: KISProgressiveRequest):
        """Validate KIS Progressive request parameters"""
        if not request.initial_hint or len(request.initial_hint.strip()) < 3:
            raise ValueError("Initial hint must be at least 3 characters long")
    
    def _update_task_stats(self, task_type: str, start_time: float, success: bool):
        """Update performance statistics for a task"""
        
        if task_type not in self.task_stats:
            return
        
        stats = self.task_stats[task_type]
        response_time = time.time() - start_time
        
        # Update count and average time
        count = stats["count"]
        avg_time = stats["avg_time"]
        
        stats["count"] = count + 1
        stats["avg_time"] = (avg_time * count + response_time) / (count + 1)
        
        # Update success rate
        if success:
            current_successes = stats["success_rate"] * count
            stats["success_rate"] = (current_successes + 1) / (count + 1)
        else:
            current_successes = stats["success_rate"] * count
            stats["success_rate"] = current_successes / (count + 1)
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        
        # Check if any task has very low success rate
        for task_type, stats in self.task_stats.items():
            if stats["count"] > 5 and stats["success_rate"] < 0.8:
                return False
        
        # Check if response times are too high
        for task_type, stats in self.task_stats.items():
            if stats["avg_time"] > 30.0:  # 30 second threshold
                return False
        
        return True
