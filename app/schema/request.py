from pydantic import BaseModel, Field
from typing import List, Optional


class KeyframeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=500)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Optional Filters
    exclude_groups: Optional[List[int]] = None
    include_groups: Optional[List[int]] = None
    include_videos: Optional[List[int]] = None

    # Temporal Filters
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    video_id: Optional[str] = None

    # Hybrid Search Filters
    use_hybrid_search: bool = False
    filter_author: Optional[str] = None
    filter_keywords: Optional[List[str]] = None
    filter_publish_date: Optional[str] = None
    metadata_weight: float = Field(default=0.3, ge=0.0, le=1.0)


