"""
Competition API endpoints for HCMC AI Challenge 2025
Implements exact task specifications for VCMR, VQA, and KIS
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional

from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse,
    VideoQARequest, VideoQAResponse,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    VCMRFeedback, VCMRInteractiveCandidate
)
from controller.competition_controller import CompetitionController
from core.dependencies import get_competition_controller
from core.logger import SimpleLogger


router = APIRouter(
    prefix="/competition",
    tags=["competition"],
    responses={404: {"description": "Not found"}},
)
logger = SimpleLogger(__name__)


@router.post(
    "/vcmr/automatic",
    response_model=VCMRAutomaticResponse,
    summary="VCMR Automatic Task",
    description="""
    Video Corpus Moment Retrieval - Automatic Track
    
    Find relevant temporal segments (moments) across a large video corpus given a free-form text query.
    Returns a ranked top-K list of moment candidates with start/end times and confidence scores.
    
    **Input Requirements:**
    - query: Free-form natural language describing the desired moment
    - corpus_index: Identifier for the corpus version being searched
    - top_k: Maximum number of candidates to return (≤100)
    - video_catalog: Optional video metadata (uses local index if not provided)
    
    **Output Format:**
    - Ranked list of moment candidates with temporal boundaries
    - Each candidate includes video_id, start_time, end_time, and relevance score
    - Optional notes explaining top candidate relevance
    
    **Example:**
    ```json
    {
        "task": "vcMr_automatic",
        "query": "A woman places a framed picture on the wall",
        "corpus_index": "v1",
        "top_k": 10
    }
    ```
    """,
    response_description="Ranked list of temporal moment candidates"
)
async def vcmr_automatic(
    request: VCMRAutomaticRequest,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process VCMR Automatic task"""
    logger.info(f"VCMR Automatic request: query='{request.query}', top_k={request.top_k}")
    
    try:
        response = await controller.process_vcmr_automatic(request)
        logger.info(f"VCMR Automatic completed: {len(response.candidates)} candidates")
        return response
    except Exception as e:
        logger.error(f"VCMR Automatic error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"VCMR processing error: {str(e)}")


@router.post(
    "/vcmr/interactive",
    response_model=VCMRInteractiveCandidate,
    summary="VCMR Interactive Task", 
    description="""
    Video Corpus Moment Retrieval - Interactive Track
    
    Provides single moment candidate and accepts human feedback to refine results.
    Supports binary relevance, graded relevance, and free-text refinement feedback.
    
    **Feedback Types:**
    - Binary: {"relevance": true|false}
    - Graded: {"relevance_score": 0.8}
    - Refinement: {"refine": "focus on outdoor scenes"}
    """,
    response_description="Single moment candidate or refined result"
)
async def vcmr_interactive(
    query: str,
    feedback: Optional[VCMRFeedback] = None,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process VCMR Interactive task with feedback"""
    logger.info(f"VCMR Interactive request: query='{query}'")
    
    try:
        response = await controller.process_vcmr_interactive(query, feedback)
        logger.info(f"VCMR Interactive completed: {response.video_id} ({response.start_time}s-{response.end_time}s)")
        return response
    except Exception as e:
        logger.error(f"VCMR Interactive error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Interactive VCMR error: {str(e)}")


@router.post(
    "/vqa",
    response_model=VideoQAResponse,
    summary="Video Question Answering",
    description="""
    Answer natural language questions about video content.
    
    Supports both full video and clip-specific questions. Provides evidence timestamps
    and confidence scores for verification.
    
    **Input Requirements:**
    - question: Natural language question about the video
    - video_id: Target video identifier  
    - video_uri: Video location or encoded frames reference
    - clip: Optional temporal clip specification
    - context: Optional ASR, OCR, or metadata context
    
    **Output Format:**
    - Short factual answer or closed-vocabulary label
    - Supporting evidence with timestamps
    - Model confidence score
    
    **Example:**
    ```json
    {
        "task": "video_qa",
        "video_id": "L01/V001",
        "video_uri": "path/to/video.mp4",
        "question": "How many people are in the scene?",
        "clip": {"start_time": 10.0, "end_time": 20.0}
    }
    ```
    """,
    response_description="Answer with supporting evidence and confidence"
)
async def video_qa(
    request: VideoQARequest,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process Video QA task"""
    logger.info(f"Video QA request: video='{request.video_id}', question='{request.question}'")
    
    try:
        response = await controller.process_video_qa(request)
        logger.info(f"Video QA completed: answer='{response.answer}', confidence={response.confidence}")
        return response
    except Exception as e:
        logger.error(f"Video QA error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video QA error: {str(e)}")


@router.post(
    "/kis/textual",
    response_model=KISResponse,
    summary="Known-Item Search - Textual",
    description="""
    Locate exact target segment from textual description.
    
    Requires precise matching to find the specific segment described.
    Returns tight temporal boundaries around the exact match.
    """,
    response_description="Exact segment location with match confidence"
)
async def kis_textual(
    request: KISTextualRequest,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process KIS Textual task"""
    logger.info(f"KIS Textual request: description='{request.text_description}'")
    
    try:
        response = await controller.process_kis_textual(request)
        logger.info(f"KIS Textual completed: {response.video_id} ({response.start_time}s-{response.end_time}s)")
        return response
    except Exception as e:
        logger.error(f"KIS Textual error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"KIS Textual error: {str(e)}")


@router.post(
    "/kis/visual",
    response_model=KISResponse,
    summary="Known-Item Search - Visual",
    description="""
    Locate exact target segment from visual example clip.
    
    Uses visual similarity matching to find the segment that matches
    the provided query clip.
    """,
    response_description="Exact segment location with visual match confidence"
)
async def kis_visual(
    request: KISVisualRequest,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process KIS Visual task"""
    logger.info(f"KIS Visual request: query_clip='{request.query_clip_uri}'")
    
    try:
        response = await controller.process_kis_visual(request)
        logger.info(f"KIS Visual completed: {response.video_id} ({response.start_time}s-{response.end_time}s)")
        return response
    except Exception as e:
        logger.error(f"KIS Visual error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"KIS Visual error: {str(e)}")


@router.post(
    "/kis/progressive",
    response_model=KISResponse,
    summary="Known-Item Search - Progressive",
    description="""
    Locate exact target segment with progressive hints.
    
    Starts with minimal description and accepts additional hints over time
    to iteratively refine the search.
    """,
    response_description="Exact segment location with progressive match confidence"
)
async def kis_progressive(
    request: KISProgressiveRequest,
    additional_hints: Optional[List[str]] = None,
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Process KIS Progressive task"""
    logger.info(f"KIS Progressive request: initial='{request.initial_hint}', hints={additional_hints}")
    
    try:
        response = await controller.process_kis_progressive(request, additional_hints)
        logger.info(f"KIS Progressive completed: {response.video_id} ({response.start_time}s-{response.end_time}s)")
        return response
    except Exception as e:
        logger.error(f"KIS Progressive error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"KIS Progressive error: {str(e)}")


@router.post(
    "/dispatch",
    summary="Universal Task Dispatcher",
    description="""
    Universal endpoint that dispatches to appropriate task handler based on task type.
    
    Accepts any valid competition task JSON and routes to the correct processor.
    Task type is determined by the 'task' field in the input.
    
    **Supported Tasks:**
    - vcMr_automatic: VCMR Automatic track
    - video_qa: Video Question Answering
    - kis_t: Known-Item Search Textual
    - kis_v: Known-Item Search Visual  
    - kis_c: Known-Item Search Progressive
    """,
    response_description="Task-specific response based on input task type"
)
async def dispatch_task(
    task_input: Dict[str, Any],
    controller: CompetitionController = Depends(get_competition_controller)
):
    """Universal task dispatcher for all competition tasks"""
    task_type = task_input.get("task", "unknown")
    logger.info(f"Task dispatch request: task='{task_type}'")
    
    try:
        response = await controller.dispatch_task(task_input)
        logger.info(f"Task dispatch completed: task='{task_type}'")
        return response
    except Exception as e:
        logger.error(f"Task dispatch error: task='{task_type}', error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Task dispatch error: {str(e)}")
