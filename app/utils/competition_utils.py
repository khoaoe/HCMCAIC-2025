"""
Utility functions for competition task processing
Helper functions for data conversion, validation, and optimization
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from schema.response import KeyframeServiceReponse
from schema.competition import VCMRCandidate, MomentCandidate


from utils.common_utils import safe_convert_video_num


def validate_competition_output(task_type: str, output: Dict[str, Any]) -> bool:
    """Validate competition output format compliance"""
    
    required_fields = {
        "vcMr_automatic": ["task", "query", "candidates"],
        "video_qa": ["task", "video_id", "question", "answer", "confidence"],
        "kis": ["task", "video_id", "start_time", "end_time", "match_confidence"]
    }
    
    if task_type not in required_fields:
        return False
    
    for field in required_fields[task_type]:
        if field not in output:
            return False
    
    # Additional validations
    if task_type == "vcMr_automatic":
        candidates = output.get("candidates", [])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                return False
            if candidate.get("end_time", 0) <= candidate.get("start_time", 0):
                return False
    
    return True


def convert_keyframes_to_vcmr_format(
    moments: List[MomentCandidate],
    max_candidates: int = 100
) -> List[Dict[str, Any]]:
    """Convert moment candidates to VCMR competition format"""
    
    candidates = []
    for moment in moments[:max_candidates]:
        candidate = {
            "video_id": moment.video_id,
            "start_time": round(moment.start_time, 2),
            "end_time": round(moment.end_time, 2),
            "score": round(moment.confidence_score, 4)
        }
        candidates.append(candidate)
    
    return candidates


def parse_video_id(video_id: str) -> Tuple[int, int]:
    """
    Parse video ID into group_num and video_num
    Supports formats: L01/V001, 1/1, L1/V1
    """
    
    # Remove path separators and normalize
    video_id = video_id.replace('\\', '/').strip('/')
    
    patterns = [
        r'L(\d+)/V(\d+)',  # L01/V001 format
        r'(\d+)/(\d+)',    # 1/1 format
        r'L(\d+)V(\d+)',   # L1V1 format (no slash)
    ]
    
    for pattern in patterns:
        match = re.match(pattern, video_id)
        if match:
            group_num = int(match.group(1))
            video_num = int(match.group(2))
            return group_num, video_num
    
    raise ValueError(f"Invalid video_id format: {video_id}")


def create_video_metadata_index(
    keyframes: List[KeyframeServiceReponse],
    fps: float = 25.0
) -> Dict[str, Dict[str, Any]]:
    """Create video metadata index from keyframes for temporal mapping"""
    
    video_metadata = {}
    
    # Group keyframes by video
    video_keyframes = {}
    for kf in keyframes:
        video_key = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}"
        if video_key not in video_keyframes:
            video_keyframes[video_key] = []
        video_keyframes[video_key].append(kf)
    
    # Calculate metadata for each video
    for video_key, kf_list in video_keyframes.items():
        max_frame = max(kf.keyframe_num for kf in kf_list)
        min_frame = min(kf.keyframe_num for kf in kf_list)
        
        video_metadata[video_key] = {
            "video_id": video_key,
            "group_num": kf_list[0].group_num,
            "video_num": kf_list[0].video_num,
            "fps": fps,
            "total_frames": max_frame,
            "duration": max_frame / fps,
            "keyframe_count": len(kf_list),
            "frame_range": (min_frame, max_frame)
        }
    
    return video_metadata


def optimize_search_parameters(
    query: str,
    task_type: str,
    interactive_mode: bool = False
) -> Dict[str, Any]:
    """
    Dynamically optimize search parameters based on query characteristics
    """
    
    # Analyze query complexity
    query_length = len(query.split())
    has_temporal_words = any(word in query.lower() for word in 
                           ['before', 'after', 'during', 'while', 'then', 'first', 'last', 'start', 'end'])
    has_specific_objects = any(word in query.lower() for word in
                             ['person', 'car', 'house', 'dog', 'cat', 'chair', 'table'])
    
    # Determine complexity
    complexity_score = 0
    complexity_score += min(query_length / 10, 1.0)  # Length factor
    complexity_score += 0.3 if has_temporal_words else 0
    complexity_score += 0.2 if has_specific_objects else 0
    
    if complexity_score < 0.3:
        complexity = "simple"
    elif complexity_score < 0.7:
        complexity = "moderate"  
    else:
        complexity = "complex"
    
    # Base parameters by task type
    task_params = {
        "vcMr_automatic": {
            "base_top_k": 200,
            "score_threshold": 0.05,
            "enable_reranking": True
        },
        "video_qa": {
            "base_top_k": 50,
            "score_threshold": 0.1,
            "enable_reranking": True
        },
        "kis": {
            "base_top_k": 100,
            "score_threshold": 0.2,
            "enable_reranking": True
        }
    }
    
    params = task_params.get(task_type, task_params["vcMr_automatic"])
    
    # Adjust for complexity
    complexity_adjustments = {
        "simple": {"top_k_mult": 0.7, "threshold_add": 0.05},
        "moderate": {"top_k_mult": 1.0, "threshold_add": 0.0},
        "complex": {"top_k_mult": 1.3, "threshold_add": -0.02}
    }
    
    adj = complexity_adjustments[complexity]
    params["top_k"] = int(params["base_top_k"] * adj["top_k_mult"])
    params["score_threshold"] = max(0.01, params["score_threshold"] + adj["threshold_add"])
    
    # Interactive mode adjustments
    if interactive_mode:
        params["top_k"] = min(params["top_k"], 100)  # Speed limit
        params["enable_reranking"] = False  # Skip reranking for speed
        params["score_threshold"] = max(params["score_threshold"], 0.1)
    
    return params


def create_competition_summary_report(
    system_performance: Dict[str, Any],
    task_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Create comprehensive summary report for competition evaluation"""
    
    total_tasks = sum(len(results) for results in task_results.values())
    
    # Calculate task-specific metrics
    task_metrics = {}
    for task_type, results in task_results.items():
        if results:
            avg_confidence = np.mean([r.get("confidence", 0) for r in results])
            avg_response_time = np.mean([r.get("response_time", 0) for r in results])
            
            task_metrics[task_type] = {
                "total_requests": len(results),
                "avg_confidence": round(avg_confidence, 3),
                "avg_response_time": round(avg_response_time, 3),
                "success_rate": len([r for r in results if r.get("success", False)]) / len(results)
            }
    
    return {
        "competition_summary": {
            "total_tasks_processed": total_tasks,
            "system_uptime": system_performance.get("uptime", 0),
            "avg_response_time": system_performance.get("avg_response_time", 0),
            "memory_usage": system_performance.get("memory_usage", "unknown"),
            "cache_efficiency": system_performance.get("cache_hits", 0) / max(system_performance.get("total_queries", 1), 1)
        },
        "task_performance": task_metrics,
        "optimization_recommendations": {
            "performance_bottlenecks": _identify_bottlenecks(task_metrics),
            "suggested_improvements": _get_improvement_suggestions(task_metrics)
        }
    }


def _identify_bottlenecks(task_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Identify performance bottlenecks from metrics"""
    
    bottlenecks = []
    
    for task_type, metrics in task_metrics.items():
        if metrics.get("avg_response_time", 0) > 8.0:
            bottlenecks.append(f"{task_type}: Response time too high")
        
        if metrics.get("success_rate", 1.0) < 0.8:
            bottlenecks.append(f"{task_type}: Low success rate")
        
        if metrics.get("avg_confidence", 1.0) < 0.5:
            bottlenecks.append(f"{task_type}: Low confidence scores")
    
    return bottlenecks


def _get_improvement_suggestions(task_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    """Generate improvement suggestions based on metrics"""
    
    suggestions = []
    
    for task_type, metrics in task_metrics.items():
        if metrics.get("avg_response_time", 0) > 5.0:
            suggestions.append(f"Optimize {task_type} retrieval pipeline for speed")
        
        if metrics.get("avg_confidence", 1.0) < 0.6:
            suggestions.append(f"Improve {task_type} relevance scoring and filtering")
    
    # Global suggestions
    if any(m.get("avg_response_time", 0) > 6.0 for m in task_metrics.values()):
        suggestions.append("Consider implementing result caching")
        suggestions.append("Evaluate hardware scaling options")
    
    return suggestions
