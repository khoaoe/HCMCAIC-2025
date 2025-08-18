"""
Temporal Localization Module
Converts keyframe-based results to temporal moments with start_time/end_time
"""

from typing import List, Dict, Tuple, Optional, Any
import json
from pathlib import Path
from schema.response import KeyframeServiceReponse
from schema.competition import TemporalMapping, MomentCandidate


class TemporalLocalizer:
    """Handles conversion from keyframes to temporal moments"""
    
    def __init__(
        self, 
        video_metadata_path: Optional[Path] = None,
        default_fps: float = 25.0
    ):
        self.default_fps = default_fps
        self.video_metadata: Dict[str, TemporalMapping] = {}
        
        if video_metadata_path and video_metadata_path.exists():
            self._load_video_metadata(video_metadata_path)
    
    def _load_video_metadata(self, metadata_path: Path):
        """Load video metadata including duration, fps, etc."""
        try:
            with open(metadata_path) as f:
                data = json.load(f)
                for video_info in data:
                    video_key = f"L{video_info['group_num']:02d}/V{video_info['video_num']:03d}"
                    self.video_metadata[video_key] = TemporalMapping(**video_info)
        except Exception as e:
            print(f"Warning: Could not load video metadata: {e}")
    
    def keyframe_to_timestamp(
        self, 
        group_num: int, 
        video_num: int, 
        keyframe_num: int,
        fps: Optional[float] = None
    ) -> float:
        """Convert keyframe number to timestamp in seconds"""
        video_key = f"L{group_num:02d}/V{video_num:03d}"
        
        if video_key in self.video_metadata:
            fps = fps or self.video_metadata[video_key].fps
        else:
            fps = fps or self.default_fps
            
        return keyframe_num / fps
    
    def create_moment_from_keyframes(
        self,
        keyframes: List[KeyframeServiceReponse],
        expand_window: float = 2.0,  # seconds to expand around keyframes
        min_duration: float = 1.0,   # minimum moment duration
        max_duration: float = 30.0   # maximum moment duration
    ) -> MomentCandidate:
        """
        Create a temporal moment from a cluster of keyframes
        Uses intelligent windowing to create meaningful temporal segments
        """
        if not keyframes:
            raise ValueError("Cannot create moment from empty keyframes list")
        
        # Sort keyframes by frame number
        sorted_keyframes = sorted(keyframes, key=lambda x: x.keyframe_num)
        first_kf = sorted_keyframes[0]
        last_kf = sorted_keyframes[-1]
        
        # Get video metadata
        video_key = f"L{first_kf.group_num:02d}/V{first_kf.video_num:03d}"
        fps = self.default_fps
        if video_key in self.video_metadata:
            fps = self.video_metadata[video_key].fps
        
        # Convert keyframes to timestamps
        start_timestamp = self.keyframe_to_timestamp(
            first_kf.group_num, first_kf.video_num, first_kf.keyframe_num, fps
        )
        end_timestamp = self.keyframe_to_timestamp(
            last_kf.group_num, last_kf.video_num, last_kf.keyframe_num, fps
        )
        
        # Expand window around keyframes
        start_time = max(0, start_timestamp - expand_window)
        end_time = end_timestamp + expand_window
        
        # Ensure minimum duration
        if end_time - start_time < min_duration:
            center = (start_time + end_time) / 2
            start_time = max(0, center - min_duration / 2)
            end_time = center + min_duration / 2
        
        # Enforce maximum duration
        if end_time - start_time > max_duration:
            start_time = end_timestamp - max_duration / 2
            end_time = end_timestamp + max_duration / 2
            start_time = max(0, start_time)
        
        # Calculate average confidence
        avg_confidence = sum(kf.confidence_score for kf in keyframes) / len(keyframes)
        
        return MomentCandidate(
            video_id=f"L{first_kf.group_num:02d}/V{first_kf.video_num:03d}",
            group_num=first_kf.group_num,
            video_num=first_kf.video_num,
            keyframe_start=first_kf.keyframe_num,
            keyframe_end=last_kf.keyframe_num,
            start_time=start_time,
            end_time=end_time,
            confidence_score=avg_confidence,
            evidence_keyframes=[kf.keyframe_num for kf in sorted_keyframes]
        )
    
    def cluster_keyframes_by_temporal_proximity(
        self,
        keyframes: List[KeyframeServiceReponse],
        max_gap_seconds: float = 5.0
    ) -> List[List[KeyframeServiceReponse]]:
        """
        Cluster keyframes into temporal groups for moment creation
        """
        if not keyframes:
            return []
        
        # Sort by video first, then by keyframe number
        sorted_keyframes = sorted(
            keyframes, 
            key=lambda x: (x.group_num, x.video_num, x.keyframe_num)
        )
        
        clusters = []
        current_cluster = [sorted_keyframes[0]]
        
        for i in range(1, len(sorted_keyframes)):
            current_kf = sorted_keyframes[i]
            prev_kf = sorted_keyframes[i-1]
            
            # If different video, start new cluster
            if (current_kf.group_num != prev_kf.group_num or 
                current_kf.video_num != prev_kf.video_num):
                clusters.append(current_cluster)
                current_cluster = [current_kf]
                continue
            
            # Check temporal gap within same video
            current_time = self.keyframe_to_timestamp(
                current_kf.group_num, current_kf.video_num, current_kf.keyframe_num
            )
            prev_time = self.keyframe_to_timestamp(
                prev_kf.group_num, prev_kf.video_num, prev_kf.keyframe_num
            )
            
            if current_time - prev_time <= max_gap_seconds:
                current_cluster.append(current_kf)
            else:
                clusters.append(current_cluster)
                current_cluster = [current_kf]
        
        clusters.append(current_cluster)
        return clusters
    
    def create_moments_from_keyframes(
        self,
        keyframes: List[KeyframeServiceReponse],
        max_moments: int = 100
    ) -> List[MomentCandidate]:
        """
        Convert keyframes to temporal moments for VCMR output
        """
        clusters = self.cluster_keyframes_by_temporal_proximity(keyframes)
        moments = []
        
        for cluster in clusters:
            if cluster:  # Skip empty clusters
                moment = self.create_moment_from_keyframes(cluster)
                moments.append(moment)
        
        # Sort by confidence and return top results
        moments.sort(key=lambda x: x.confidence_score, reverse=True)
        return moments[:max_moments]


class ASRTemporalAligner:
    """Aligns ASR data with temporal moments for enhanced context"""
    
    def __init__(self, asr_data: Dict[str, Any]):
        self.asr_data = asr_data
    
    def get_asr_for_moment(
        self, 
        video_id: str, 
        start_time: float, 
        end_time: float
    ) -> Optional[str]:
        """Extract ASR text for a specific temporal moment"""
        if not self.asr_data:
            return None
        
        try:
            # Find matching video in ASR data
            video_asr = None
            for entry in self.asr_data.values():
                if isinstance(entry, dict) and entry.get("file_path") == video_id:
                    video_asr = entry
                    break
                elif isinstance(entry, str) and entry == video_id:
                    continue  # Simple string mapping, no temporal data
            
            if not video_asr or "result" not in video_asr:
                return None
            
            # Extract text segments that overlap with the moment
            asr_segments = []
            for segment in video_asr["result"]:
                if not isinstance(segment, dict):
                    continue
                
                seg_start = float(segment.get("start_time", 0))
                seg_end = float(segment.get("end_time", 0))
                
                # Check for overlap with moment
                if (seg_start <= end_time and seg_end >= start_time):
                    text = segment.get("text", "").strip()
                    if text:
                        asr_segments.append(text)
            
            return " ".join(asr_segments) if asr_segments else None
            
        except Exception as e:
            print(f"Warning: Error extracting ASR for moment: {e}")
            return None
