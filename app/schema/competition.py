"""
Competition-specific schemas for HCMC AI Challenge 2025
Implements exact input/output schemas as specified in the competition description
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


# ===============================
# VCMR (Video Corpus Moment Retrieval) Schemas
# ===============================

class VideoMetadata(BaseModel):
    """Video metadata structure"""
    video_id: str
    duration: float
    metadata: Optional[Dict[str, Any]] = None


class VCMRAutomaticRequest(BaseModel):
    """Input schema for VCMR Automatic task"""
    task: str = Field(default="vcMr_automatic", description="Task identifier")
    query: str = Field(..., description="Free-form natural language query")
    corpus_index: str = Field(..., description="Identifier for corpus version")
    video_catalog: Optional[List[VideoMetadata]] = Field(None, description="Optional video catalog if system has local index")
    top_k: int = Field(default=100, le=100, description="Maximum number of candidates requested")


class VCMRCandidate(BaseModel):
    """Single moment candidate for VCMR"""
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds (must be > start_time)")
    score: float = Field(..., description="Model confidence/relevance score (higher = better)")

    def model_validate(self, values):
        if isinstance(values, dict) and values.get("end_time", 0) <= values.get("start_time", 0):
            raise ValueError("end_time must be greater than start_time")
        return values


class VCMRAutomaticResponse(BaseModel):
    """Output schema for VCMR Automatic task"""
    task: str = Field(default="vcMr_automatic", description="Task identifier")
    query: str = Field(..., description="Original query")
    candidates: List[VCMRCandidate] = Field(..., description="Ranked list of moment candidates")
    notes: Optional[str] = Field(None, description="Optional explanation for top candidate relevance")


# Interactive VCMR schemas
class VCMRInteractiveCandidate(BaseModel):
    """Single candidate for interactive VCMR"""
    video_id: str
    start_time: float
    end_time: float
    score: float


class VCMRFeedback(BaseModel):
    """User feedback for interactive VCMR - supports multiple feedback types"""
    relevance: Optional[bool] = None  # Binary relevance
    relevance_score: Optional[float] = Field(None, ge=0, le=1)  # Graded relevance
    refine: Optional[str] = None  # Free-text refinement


# ===============================
# Video QA Schemas
# ===============================

class VideoQAClip(BaseModel):
    """Optional clip specification for VQA"""
    start_time: float
    end_time: float


class VideoQAContext(BaseModel):
    """Optional context for VQA"""
    asr: Optional[str] = None
    ocr: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class VideoQARequest(BaseModel):
    """Input schema for Video QA task"""
    task: str = Field(default="video_qa", description="Task identifier")
    video_id: str = Field(..., description="Video identifier")
    video_uri: str = Field(..., description="Video URI or encoded frames reference")
    clip: Optional[VideoQAClip] = Field(None, description="Optional clip specification; full video if absent")
    question: str = Field(..., description="Natural language question")
    context: Optional[VideoQAContext] = Field(None, description="Optional context (ASR, OCR, metadata)")


class VideoQAEvidence(BaseModel):
    """Supporting evidence for VQA answer"""
    start_time: float
    end_time: float
    confidence: float


class VideoQAResponse(BaseModel):
    """Output schema for Video QA task"""
    task: str = Field(default="video_qa", description="Task identifier")
    video_id: str = Field(..., description="Video identifier")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Short text answer or closed-vocab label")
    evidence: Optional[List[VideoQAEvidence]] = Field(None, description="Supporting timestamps or frames")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")


# ===============================
# KIS (Known-Item Search) Schemas
# ===============================

class KISVisualRequest(BaseModel):
    """Input schema for KIS Visual task"""
    task: str = Field(default="kis_v", description="Task identifier")
    query_clip_uri: str = Field(..., description="Short clip showing the target")
    corpus_index: str = Field(..., description="Corpus identifier")


class KISTextualRequest(BaseModel):
    """Input schema for KIS Textual task"""
    task: str = Field(default="kis_t", description="Task identifier") 
    text_description: str = Field(..., description="Textual description of target")
    corpus_index: str = Field(..., description="Corpus identifier")


class KISProgressiveRequest(BaseModel):
    """Input schema for KIS Progressive task"""
    task: str = Field(default="kis_c", description="Task identifier")
    initial_hint: str = Field(..., description="Initial hint")
    corpus_index: str = Field(..., description="Corpus identifier")
    hint_time_step_sec: int = Field(default=60, description="Time between hints")


class KISResponse(BaseModel):
    """Output schema for all KIS tasks"""
    task: str = Field(default="kis", description="Task identifier")
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    match_confidence: float = Field(..., ge=0, le=1, description="Match confidence")


# ===============================
# Interactive System Schemas
# ===============================

class InteractiveSystemRequest(BaseModel):
    """System prompt/controller request to LLM"""
    role: str = Field(default="system", description="Message role")
    task: str = Field(..., description="Task type (e.g., vcMr_interactive)")
    query: str = Field(..., description="User query")
    context: Dict[str, Any] = Field(..., description="Context information")
    allowed_actions: List[str] = Field(..., description="Allowed LLM actions")


class InteractiveLLMResponse(BaseModel):
    """LLM response in interactive mode"""
    action: str = Field(..., description="Selected action")
    payload: Dict[str, Any] = Field(..., description="Action payload")


# ===============================
# Temporal Mapping Schema
# ===============================

class TemporalMapping(BaseModel):
    """Maps keyframe numbers to temporal information"""
    # Required fields for video identification
    group_num: int
    video_num: int
    
    # Optional fields that may be present in metadata
    video_id: Optional[str] = None
    fps: float = Field(default=25.0, description="Frames per second")
    total_frames: Optional[int] = None
    duration: Optional[float] = None
    
    # Additional metadata fields from JSON files
    author: Optional[str] = None
    channel_id: Optional[str] = None
    channel_url: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    length: Optional[int] = None  # Duration in seconds
    publish_date: Optional[str] = None
    thumbnail_url: Optional[str] = None
    title: Optional[str] = None
    watch_url: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields not defined in the schema


class MomentCandidate(BaseModel):
    """Enhanced moment candidate with temporal information"""
    video_id: str
    group_num: int
    video_num: int
    keyframe_start: int
    keyframe_end: int
    start_time: float
    end_time: float
    confidence_score: float
    evidence_keyframes: List[int]  # Supporting keyframe numbers
    asr_text: Optional[str] = None
    detected_objects: Optional[List[str]] = None
