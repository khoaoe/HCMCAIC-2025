from pydantic import BaseModel, Field
from typing import List, Optional


class KeyframeServiceReponse(BaseModel):
    key: int = Field(..., description="Keyframe key")
    video_num: int = Field(..., description="Video ID")
    group_num: int = Field(..., description="Group ID")
    keyframe_num: int = Field(..., description="Keyframe number")
    confidence_score: float = Field(..., description="Keyframe number")
    embedding: Optional[List[float]] = Field(None, description="The embedding vector for this keyframe")


class SingleKeyframeDisplay(BaseModel):
    path: str
    score: float

class KeyframeDisplay(BaseModel):
    results: list[SingleKeyframeDisplay]