"""
Temporal Search API Router
Provides advanced temporal search endpoints for keyframe retrieval
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

from schema.request import (
    TextSearchWithTimeRangeRequest,
    AdvancedTemporalSearchRequest,
)
from schema.competition import MomentCandidate
from schema.response import KeyframeServiceReponse, SingleKeyframeDisplay, KeyframeDisplay
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger

logger = SimpleLogger(__name__)

router = APIRouter(
    prefix="/temporal",
    tags=["temporal-search"],
    responses={404: {"description": "Not found"}},
)


def safe_convert_video_num(video_num) -> int:
    """Safely convert video_num to int, handling cases where it might be '26_V288' format"""
    if isinstance(video_num, str):
        # Handle cases where video_num might be '26_V288' format
        if '_V' in video_num:
            # Extract just the video number part
            video_part = video_num.split('_V')[-1]
            return int(video_part)
        else:
            return int(video_num)
    else:
        return int(video_num)


@router.post(
    "/search/time-range",
    response_model=KeyframeDisplay,
    summary="Text search with temporal filtering",
    description="""
    Perform text-based search for keyframes within a specific time range.
    
    This endpoint enables temporal search by filtering keyframes based on timestamps,
    allowing precise moment retrieval within videos.
    
    **Features:**
    - Native Milvus scalar field filtering for timestamps
    - Support for single video or cross-corpus temporal search
    - Efficient temporal indexing for fast queries
    - Combined semantic and temporal relevance scoring
    
    **Parameters:**
    - **query**: The search text
    - **start_time**: Start time in seconds
    - **end_time**: End time in seconds (must be > start_time)
    - **video_id**: Optional video ID in format 'Lxx/Vxxx' to restrict search
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    - **include_groups**: Optional list of group IDs to include
    - **include_videos**: Optional list of video IDs to include
    
    **Use Cases:**
    - Find specific moments within long videos
    - Search for events that occurred in a known time window
    - Temporal content discovery and analysis
    - Cross-video temporal correlation analysis
    
    **Example:**
    ```json
    {
        "query": "person walking in park",
        "start_time": 30.0,
        "end_time": 90.0,
        "video_id": "L01/V001",
        "top_k": 10,
        "score_threshold": 0.3
    }
    ```
    """,
    response_description="List of matching keyframes within the specified time range"
)
async def search_keyframes_time_range(
    request: TextSearchWithTimeRangeRequest,
    controller: QueryController = Depends(get_query_controller)
):
    """Search for keyframes within a specific time range using temporal filtering."""
    
    logger.info(f"Temporal search request: query='{request.query}', time={request.start_time}-{request.end_time}s, video={request.video_id}")
    
    try:
        results = await controller.search_text_temporal(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            start_time=request.start_time,
            end_time=request.end_time,
            video_id=request.video_id,
            include_groups=request.include_groups,
            include_videos=request.include_videos
        )
        
        logger.info(f"Found {len(results)} temporal search results for time range {request.start_time}-{request.end_time}s")
        
        display_results = list(
            map(
                lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
                map(controller.convert_model_to_path, results)
            )
        )
        return KeyframeDisplay(results=display_results)
        
    except Exception as e:
        logger.error(f"Temporal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Temporal search failed: {str(e)}")


@router.get(
    "/search/video-time-window",
    response_model=KeyframeDisplay,
    summary="Search within a specific video time window",
    description="""
    Search for keyframes within a specific time window of a single video.
    
    This endpoint is optimized for precise temporal search within individual videos,
    providing fine-grained moment retrieval capabilities.
    
    **Parameters:**
    - **query**: The search text
    - **video_id**: Video ID in format 'Lxx/Vxxx'
    - **start_time**: Start time in seconds
    - **end_time**: End time in seconds
    - **top_k**: Maximum number of results to return
    - **score_threshold**: Minimum confidence score
    
    **Example:**
    GET /api/v1/temporal/search/video-time-window?query=sunset&video_id=L02/V003&start_time=120.5&end_time=185.2
    """
)
async def search_video_time_window(
    query: str = Query(..., description="Search query"),
    video_id: str = Query(..., description="Video ID (Lxx/Vxxx format)"),
    start_time: float = Query(..., ge=0.0, description="Start time in seconds"),
    end_time: float = Query(..., ge=0.0, description="End time in seconds"),
    top_k: int = Query(default=25, ge=1, le=100, description="Max results"),
    score_threshold: float = Query(default=0.1, ge=0.0, le=1.0, description="Min score"),
    controller: QueryController = Depends(get_query_controller)
):
    """Search within a specific time window of a video using GET parameters."""
    
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    
    logger.info(f"Video time window search: query='{query}', video={video_id}, window={start_time}-{end_time}s")
    
    try:
        results = await controller.search_text_time_window(
            query=query,
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        logger.info(f"Found {len(results)} results in video {video_id} time window {start_time}-{end_time}s")
        
        display_results = list(
            map(
                lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
                map(controller.convert_model_to_path, results)
            )
        )
        return KeyframeDisplay(results=display_results)
        
    except Exception as e:
        logger.error(f"Video time window search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video time window search failed: {str(e)}")


@router.get("/search/temporal-stats")
async def get_temporal_search_stats(
    controller: QueryController = Depends(get_query_controller)
):
    """Get statistics about temporal search capability and data coverage."""
    
    try:
        # This would query the collection to get temporal coverage stats
        # For now, return basic info
        return {
            "status": "temporal_search_enabled",
            "features": [
                "Native timestamp filtering",
                "Video-specific time windows",
                "Cross-corpus temporal search",
                "Efficient scalar field indexing"
            ],
            "supported_formats": {
                "video_id": "Lxx/Vxxx (e.g., L01/V001)",
                "time_units": "seconds (float)",
                "temporal_resolution": "frame-level (typically 25 FPS)"
            },
            "endpoints": {
                "time_range_search": "/api/v1/temporal/search/time-range",
                "video_window_search": "/api/v1/temporal/search/video-time-window"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get temporal stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# GRAB Framework Endpoints
# ===============================

@router.post(
    "/search/enhanced-moments",
    summary="Temporal search with GRAB framework",
    description="""
    Advanced temporal search using GRAB (Global Re-ranking and Adaptive Bidirectional search) framework.
    
    **GRAB Framework Features:**
    - **Shot Detection**: Strategic keyframe sampling from video shots
    - **Perceptual Deduplication**: Remove near-duplicate frames using pHash
    - **SuperGlobal Reranking**: GeM pooling for feature refinement
    - **ABTS Algorithm**: Adaptive Bidirectional Temporal Search for precise boundaries
    - **Temporal Stability**: Balance semantic similarity with temporal consistency
    
    **Performance Modes:**
    - **fast**: Basic temporal search (λ_s=0.8, λ_t=0.2, 1s window)
    - **balanced**: Full GRAB pipeline (λ_s=0.7, λ_t=0.3, 2s window)  
    - **precision**: Maximum accuracy (λ_s=0.6, λ_t=0.4, 3s window)
    
    **Returns:**
    Precisely localized temporal moments with confidence scores and boundary analysis.
    """
)
async def temporal_search(
    query: str = Query(..., description="Search query"),
    start_time: Optional[float] = Query(None, ge=0.0, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, ge=0.0, description="End time in seconds"),
    video_id: Optional[str] = Query(None, description="Video ID (Lxx/Vxxx format)"),
    top_k: int = Query(default=20, ge=1, le=100, description="Max moments to return"),
    score_threshold: float = Query(default=0.1, ge=0.0, le=1.0, description="Min confidence"),
    optimization_level: str = Query(default="balanced", description="fast/balanced/precision"),
    controller: QueryController = Depends(get_query_controller)
):
    """Enhanced temporal search using GRAB framework optimizations."""
    
    if start_time is not None and end_time is not None and end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    
    logger.info(f"Temporal search (GRAB): query='{query}', mode={optimization_level}")
    
    try:
        moments = await controller.temporal_search(
            query=query,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            top_k=top_k,
            score_threshold=score_threshold,
            optimization_level=optimization_level
        )
        
        logger.info(f"GRAB framework found {len(moments)} enhanced moments")
        
        # Convert evidence keyframes to proper paths
        moments_with_paths = []
        for moment in moments:
            # Convert evidence keyframes to KeyframeServiceReponse objects for path conversion
            evidence_keyframe_objects = []
            for keyframe_num in moment.evidence_keyframes:
                # Create a KeyframeServiceReponse object for path conversion
                keyframe_obj = KeyframeServiceReponse(
                    key=int(keyframe_num),  
                    video_num=safe_convert_video_num(moment.video_num),
                    group_num=int(moment.group_num),
                    keyframe_num=int(keyframe_num),  
                    confidence_score=float(moment.confidence_score)
                )
                evidence_keyframe_objects.append(keyframe_obj)
            
            # Convert to paths using the controller's convert_model_to_path method
            evidence_paths = []
            for keyframe_obj in evidence_keyframe_objects:
                path, score = controller.convert_model_to_path(keyframe_obj)
                evidence_paths.append(path)
            
            moments_with_paths.append({
                "video_id": moment.video_id,
                "start_time": moment.start_time,
                "end_time": moment.end_time,
                "duration": round(moment.end_time - moment.start_time, 2),
                "confidence_score": round(moment.confidence_score, 4),
                "evidence_keyframes": evidence_paths,  # Now contains proper paths
                "keyframe_range": {
                    "start": moment.keyframe_start,
                    "end": moment.keyframe_end
                }
            })
        
        return {
            "framework": "GRAB",
            "query": query,
            "time_range": {"start": start_time, "end": end_time} if start_time and end_time else None,
            "video_id": video_id,
            "optimization_level": optimization_level,
            "moments": moments_with_paths,
            "total_moments": len(moments)
        }
        
    except Exception as e:
        logger.error(f"Temporal search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Temporal search failed: {str(e)}")


@router.post(
    "/search/temporal-analysis",
    summary="Comprehensive temporal moment analysis",
    description="""
    Comprehensive temporal analysis comparing traditional and GRAB-enhanced search.
    
    Provides detailed comparison between:
    - Traditional keyframe search
    - GRAB-enhanced temporal moments
    - Performance metrics and optimization statistics
    """
)
async def temporal_analysis(
    query: str = Query(..., description="Search query"),
    start_time: float = Query(..., ge=0.0, description="Start time in seconds"),
    end_time: float = Query(..., ge=0.0, description="End time in seconds"),
    video_id: Optional[str] = Query(None, description="Video ID (optional)"),
    precision_mode: bool = Query(default=False, description="Enable precision mode"),
    controller: QueryController = Depends(get_query_controller)
):
    """Comprehensive temporal analysis with GRAB framework comparison."""
    
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    
    logger.info(f"Temporal analysis: query='{query}', range=[{start_time}, {end_time}], precision={precision_mode}")
    
    try:
        analysis_result = await controller.search_temporal_moments(
            query=query,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            precision_mode=precision_mode
        )
        
        logger.info(f"Temporal analysis complete: {analysis_result['enhanced_moments_count']} enhanced moments vs {analysis_result['traditional_keyframes_count']} traditional keyframes")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Temporal analysis failed: {str(e)}")
