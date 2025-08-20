from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class BaseSearchRequest(BaseModel):
    """Base search request with common parameters"""
    query: str = Field(..., description="Search query text", min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500, description="Number of top results to return")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence score threshold")


class TextSearchRequest(BaseSearchRequest):
    """Simple text search request"""
    pass


class TextSearchWithExcludeGroupsRequest(BaseSearchRequest):
    """Text search request with group exclusion"""
    exclude_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to exclude from search results",
    )


class TextSearchWithSelectedGroupsAndVideosRequest(BaseSearchRequest):
    """Text search request with specific group and video selection"""
    include_groups: List[int] = Field(
        default_factory=list,
        description="List of group IDs to include in search results",
    )
    include_videos: List[int] = Field(
        default_factory=list,
        description="List of video IDs to include in search results",
    )


class TextSearchWithTimeRangeRequest(BaseSearchRequest):
    """Text search request with temporal filtering"""
    start_time: float = Field(..., ge=0.0, description="Start time in seconds")
    end_time: float = Field(..., ge=0.0, description="End time in seconds (must be > start_time)")
    video_id: Optional[str] = Field(None, description="Video ID in format 'Lxx/Vxxx' to restrict search to specific video")
    include_groups: Optional[List[int]] = Field(default=None, description="Optional list of group IDs to include")
    include_videos: Optional[List[int]] = Field(default=None, description="Optional list of video IDs to include")

    def model_validate(self, values):
        if isinstance(values, dict) and values.get("end_time", 0) <= values.get("start_time", 0):
            raise ValueError("end_time must be greater than start_time")
        return values


class AdvancedTemporalSearchRequest(BaseSearchRequest):
    """Advanced temporal search with multiple time windows and semantic expansion"""
    time_windows: List[Dict[str, float]] = Field(
        ..., 
        description="List of time windows, each with 'start_time' and 'end_time'",
        min_items=1
    )
    video_id: Optional[str] = Field(None, description="Video ID to restrict search")
    semantic_expansion: bool = Field(default=True, description="Enable semantic query expansion")
    temporal_clustering: bool = Field(default=True, description="Group results by temporal proximity")
    include_asr_context: bool = Field(default=True, description="Include ASR text in temporal windows")


