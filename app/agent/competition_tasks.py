from typing import List, Dict, Any, Optional
from pathlib import Path
from llama_index.core.llms import LLM
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from agent.temporal_localization import TemporalLocalizer
from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse, VideoQARequest, VideoQAResponse,
    KISTextualRequest, KISVisualRequest, KISProgressiveRequest, KISResponse
)


class VCMRTaskProcessor:
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: Dict[str, List[str]],
        asr_data: Dict[str, Any],
        video_metadata_path: Optional[Path] = None,
    ):
        # Store dependencies for later logic migration
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.data_folder = data_folder
        self.objects_data = objects_data
        self.asr_data = asr_data
        self.temporal_localizer = TemporalLocalizer(video_metadata_path)

        # Basic performance stats placeholder
        self._performance_stats: Dict[str, Any] = {
            "total_queries": 0,
            "avg_latency_ms": 0.0,
        }

    # Methods expected by CompetitionController
    async def vcmr_automatic(self, request: VCMRAutomaticRequest) -> VCMRAutomaticResponse:
        raise NotImplementedError

    async def video_qa(self, request: VideoQARequest) -> VideoQAResponse:
        raise NotImplementedError

    async def kis_textual(self, request: KISTextualRequest) -> KISResponse:
        raise NotImplementedError

    async def kis_visual(self, request: KISVisualRequest) -> KISResponse:
        raise NotImplementedError

    async def kis_progressive(self, request: KISProgressiveRequest, session_id: str, additional_hints: Optional[List[str]] = None) -> KISResponse:
        raise NotImplementedError

    def get_performance_metrics(self) -> Dict[str, Any]:
        return self._performance_stats


class VQATaskProcessor:
    def __init__(self, llm: LLM, keyframe_service: KeyframeQueryService, model_service: ModelService, temporal_localizer: TemporalLocalizer):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer

    async def process(self, request: VideoQARequest) -> VideoQAResponse:
        raise NotImplementedError


class KISTaskProcessor:
    def __init__(self, llm: LLM, keyframe_service: KeyframeQueryService, model_service: ModelService, temporal_localizer: TemporalLocalizer):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer

    async def process_textual(self, request: KISTextualRequest) -> KISResponse:
        raise NotImplementedError

    async def process_visual(self, request: KISVisualRequest) -> KISResponse:
        raise NotImplementedError

    async def process_progressive(self, request: KISProgressiveRequest, session_id: str) -> KISResponse:
        raise NotImplementedError


