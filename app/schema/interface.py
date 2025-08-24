from pydantic import BaseModel, Field
from typing import List, Optional

class KeyframeInterface(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: int = Field(..., description="Video ID")
    group_num: int = Field(..., description="Group ID")
    keyframe_num: int = Field(..., description="Keyframe number")
    phash: Optional[str] = Field(None, description="Perceptual hash of the keyframe image")




class MilvusSearchRequest(BaseModel):
    embedding: List[float] = Field(..., description="Query embedding vector")
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of top results to return")
    exclude_ids: Optional[List[int]] = Field(default=None, description="IDs to exclude from search results")
    # Temporal search fields
    start_time: Optional[float] = Field(default=None, description="Start time in seconds for temporal filtering")
    end_time: Optional[float] = Field(default=None, description="End time in seconds for temporal filtering")
    video_nums: Optional[List[int]] = Field(default=None, description="Video numbers to filter by")
    group_nums: Optional[List[int]] = Field(default=None, description="Group numbers to filter by")


class MilvusSearchResult(BaseModel):
    """Individual search result"""
    id_: int = Field(..., description="Primary key of the result")
    distance: float = Field(..., description="Distance/similarity score")
    embedding: Optional[List[float]] = Field(default=None, description="Original embedding vector")
    # Temporal metadata fields
    timestamp: Optional[float] = Field(default=None, description="Timestamp in seconds")
    group_num: Optional[int] = Field(default=None, description="Group number")
    video_num: Optional[int] = Field(default=None, description="Video number")
    keyframe_num: Optional[int] = Field(default=None, description="Keyframe number")


class MilvusSearchResponse(BaseModel):
    """Response model for vector search"""
    results: List[MilvusSearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    search_time_ms: Optional[float] = Field(default=None, description="Search execution time in milliseconds")


