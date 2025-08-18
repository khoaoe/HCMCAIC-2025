"""
Competition Controller for HCMC AI Challenge 2025
Coordinates competition task processing with proper resource management
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from llama_index.core.llms import LLM

from agent.competition_tasks import CompetitionTaskDispatcher
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse,
    VideoQARequest, VideoQAResponse,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    VCMRFeedback, VCMRInteractiveCandidate
)
import json


class CompetitionController:
    """
    Main controller for competition tasks
    Manages resources and coordinates between different task agents
    """
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data_path: Optional[Path] = None,
        asr_data_path: Optional[Path] = None,
        video_metadata_path: Optional[Path] = None
    ):
        # Load auxiliary data
        objects_data = self._load_json_data(objects_data_path) if objects_data_path else {}
        asr_data = self._load_json_data(asr_data_path) if asr_data_path else {}
        
        # Initialize task dispatcher
        self.task_dispatcher = CompetitionTaskDispatcher(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            video_metadata_path=video_metadata_path
        )
        
        # Store for interactive sessions
        self.interactive_sessions: Dict[str, Any] = {}
    
    def _load_json_data(self, path: Path) -> Dict[str, Any]:
        """Load JSON data with error handling"""
        try:
            if path and path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load data from {path}: {e}")
        return {}
    
    async def process_vcmr_automatic(
        self, 
        request: VCMRAutomaticRequest
    ) -> VCMRAutomaticResponse:
        """Process VCMR Automatic task"""
        return await self.task_dispatcher.vcmr_agent.process_automatic_vcmr(request)
    
    async def process_vcmr_interactive(
        self,
        query: str,
        feedback: Optional[VCMRFeedback] = None,
        session_id: str = "default"
    ) -> VCMRInteractiveCandidate:
        """Process VCMR Interactive task with feedback handling"""
        
        # Get previous candidates from session
        previous_candidates = self.interactive_sessions.get(session_id, {}).get("candidates")
        
        result = await self.task_dispatcher.vcmr_agent.process_interactive_vcmr(
            query=query,
            feedback=feedback,
            previous_candidates=previous_candidates
        )
        
        # Store session state
        self.interactive_sessions[session_id] = {
            "query": query,
            "candidates": [result],
            "feedback_history": self.interactive_sessions.get(session_id, {}).get("feedback_history", [])
        }
        
        if feedback:
            self.interactive_sessions[session_id]["feedback_history"].append(feedback)
        
        return result
    
    async def process_video_qa(self, request: VideoQARequest) -> VideoQAResponse:
        """Process Video QA task"""
        return await self.task_dispatcher.vqa_agent.process_video_qa(request)
    
    async def process_kis_textual(self, request: KISTextualRequest) -> KISResponse:
        """Process KIS Textual task"""
        return await self.task_dispatcher.kis_agent.process_kis_textual(request)
    
    async def process_kis_visual(self, request: KISVisualRequest) -> KISResponse:
        """Process KIS Visual task"""
        return await self.task_dispatcher.kis_agent.process_kis_visual(request)
    
    async def process_kis_progressive(
        self, 
        request: KISProgressiveRequest,
        additional_hints: Optional[List[str]] = None
    ) -> KISResponse:
        """Process KIS Progressive task"""
        return await self.task_dispatcher.kis_agent.process_kis_progressive(
            request, additional_hints
        )
    
    async def dispatch_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Universal task dispatcher"""
        return await self.task_dispatcher.dispatch_task(task_input)
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get interactive session state for debugging/monitoring"""
        return self.interactive_sessions.get(session_id, {})
    
    def clear_session(self, session_id: str):
        """Clear interactive session state"""
        if session_id in self.interactive_sessions:
            del self.interactive_sessions[session_id]
