"""
Shot Detection and Strategic Keyframe Extraction Module
Implements GRAB framework's preprocessing stage for efficient temporal search
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
import imagehash
from PIL import Image
import os
from pathlib import Path

from schema.response import KeyframeServiceReponse


from utils.common_utils import safe_convert_video_num


@dataclass
class Shot:
    """Represents a video shot with start/end frames"""
    start_frame: int
    end_frame: int
    duration: int
    representative_keyframes: List[int]
    shot_id: str


@dataclass
class KeyframeMeta:
    """Enhanced keyframe with temporal and visual metadata"""
    keyframe_num: int
    timestamp: float
    shot_id: str
    perceptual_hash: str
    visual_stability: float
    is_representative: bool


class ShotDetector:
    """
    Shot detection using simplified approach (can be extended with TransNetV2)
    Extracts strategic keyframes following GRAB methodology
    """
    
    def __init__(self, 
                 keyframe_extraction_count: int = 4,
                 similarity_threshold: float = 0.8,
                 fps: float = 25.0):
        self.keyframe_extraction_count = keyframe_extraction_count
        self.similarity_threshold = similarity_threshold
        self.fps = fps
    
    def detect_shots_from_keyframes(
        self,
        keyframes: List[KeyframeServiceReponse],
        temporal_gap_threshold: float = 5.0
    ) -> List[Shot]:
        """
        Detect shots from existing keyframes using temporal gaps
        (Simplified approach - in production would use TransNetV2)
        """
        
        if not keyframes:
            return []
        
        # Sort keyframes by temporal order
        sorted_keyframes = sorted(keyframes, key=lambda x: (x.group_num, x.video_num, x.keyframe_num))
        
        shots = []
        current_shot_start = sorted_keyframes[0].keyframe_num
        current_shot_frames = [sorted_keyframes[0].keyframe_num]
        
        for i in range(1, len(sorted_keyframes)):
            curr_kf = sorted_keyframes[i]
            prev_kf = sorted_keyframes[i-1]
            
            # Calculate temporal gap
            curr_time = curr_kf.keyframe_num / self.fps
            prev_time = prev_kf.keyframe_num / self.fps
            gap = curr_time - prev_time
            
            # Check if same video and within temporal threshold
            if (curr_kf.group_num == prev_kf.group_num and 
                safe_convert_video_num(curr_kf.video_num) == safe_convert_video_num(prev_kf.video_num) and 
                gap <= temporal_gap_threshold):
                current_shot_frames.append(curr_kf.keyframe_num)
            else:
                # End current shot and start new one
                if len(current_shot_frames) > 1:
                    shot = Shot(
                        start_frame=current_shot_start,
                        end_frame=current_shot_frames[-1],
                        duration=len(current_shot_frames),
                        representative_keyframes=self._extract_representative_keyframes(current_shot_frames),
                        shot_id=f"shot_{prev_kf.group_num}_{prev_kf.video_num}_{len(shots)}"
                    )
                    shots.append(shot)
                
                current_shot_start = curr_kf.keyframe_num
                current_shot_frames = [curr_kf.keyframe_num]
        
        # Add final shot
        if len(current_shot_frames) > 1:
            last_kf = sorted_keyframes[-1]
            shot = Shot(
                start_frame=current_shot_start,
                end_frame=current_shot_frames[-1],
                duration=len(current_shot_frames),
                representative_keyframes=self._extract_representative_keyframes(current_shot_frames),
                shot_id=f"shot_{last_kf.group_num}_{last_kf.video_num}_{len(shots)}"
            )
            shots.append(shot)
        
        return shots
    
    def _extract_representative_keyframes(self, shot_frames: List[int]) -> List[int]:
        """
        Extract representative keyframes from shot using GRAB formula
        K_extract = {K_{a+floor(i*(b-a)/3)} | for i in {0,1,2,3}}
        """
        
        if len(shot_frames) <= self.keyframe_extraction_count:
            return shot_frames
        
        a = 0  # Start index
        b = len(shot_frames) - 1  # End index
        
        representative_indices = []
        for i in range(self.keyframe_extraction_count):
            idx = a + int(i * (b - a) / (self.keyframe_extraction_count - 1))
            idx = min(idx, b)  # Ensure within bounds
            representative_indices.append(shot_frames[idx])
        
        return list(set(representative_indices))  # Remove duplicates


class PerceptualHashDeduplicator:
    """
    Implements perceptual hashing deduplication from GRAB framework
    """
    
    def __init__(self, similarity_threshold: float = 0.8, hash_size: int = 8):
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
    
    def compute_perceptual_hash(self, image_path: str) -> str:
        """Compute perceptual hash for image"""
        try:
            image = Image.open(image_path)
            # Convert to grayscale and compute difference hash
            phash = imagehash.dhash(image, hash_size=self.hash_size)
            return str(phash)
        except Exception as e:
            print(f"Warning: Could not compute hash for {image_path}: {e}")
            return ""
    
    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two perceptual hashes
        Returns value between 0 (different) and 1 (identical)
        """
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        
        # Hamming distance
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        max_distance = len(hash1) * 4  # Each hex character represents 4 bits
        
        similarity = 1.0 - (hamming_distance / max_distance)
        return similarity
    
    def deduplicate_keyframes(
        self,
        keyframes: List[KeyframeServiceReponse]
    ) -> List[KeyframeServiceReponse]:
        """
        Remove near-duplicate keyframes using pre-loaded perceptual hashes
        Following GRAB's deduplication approach
        """
        
        if not keyframes:
            return []
        
        # Use pre-loaded hashes from KeyframeServiceReponse objects
        keyframe_hashes = {}
        for kf in keyframes:
            if kf.phash:
                keyframe_hashes[kf.key] = kf.phash
        
        # Deduplicate based on hash similarity
        unique_keyframes = []
        processed_hashes = set()
        
        for kf in keyframes:
            kf_hash = keyframe_hashes.get(kf.key, "")
            
            if not kf_hash:
                unique_keyframes.append(kf)  # Keep if no hash available
                continue
            
            # Check similarity with already processed hashes
            is_duplicate = False
            for processed_hash in processed_hashes:
                similarity = self.calculate_hash_similarity(kf_hash, processed_hash)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_keyframes.append(kf)
                processed_hashes.add(kf_hash)
        
        dedup_ratio = len(unique_keyframes) / len(keyframes) if keyframes else 0
        print(f"Deduplication: {len(keyframes)} → {len(unique_keyframes)} keyframes ({dedup_ratio:.2%} retained)")
        
        return unique_keyframes


class TemporalStabilityAnalyzer:
    """
    Analyzes temporal stability of keyframes for ABTS algorithm
    """
    
    def __init__(self, neighborhood_size: int = 5):
        self.neighborhood_size = neighborhood_size
    
    def calculate_temporal_stability(
        self,
        keyframes: List[KeyframeServiceReponse],
        target_keyframe: KeyframeServiceReponse,
        embeddings_cache: Dict[int, np.ndarray]
    ) -> float:
        """
        Calculate temporal stability score following GRAB methodology
        t_i = 1 - min(1, 2 * σ({e_j · e_i | j ∈ N_i}))
        """
        
        target_embedding = embeddings_cache.get(target_keyframe.key)
        if target_embedding is None:
            return 0.5  # Default stability
        
        # Find temporal neighbors
        neighbors = []
        target_time = target_keyframe.keyframe_num / 25.0  # Assuming 25 FPS
        
        for kf in keyframes:
            if (kf.group_num == target_keyframe.group_num and 
                kf.video_num == target_keyframe.video_num and
                kf.key != target_keyframe.key):
                
                kf_time = kf.keyframe_num / 25.0
                time_diff = abs(kf_time - target_time)
                
                if time_diff <= self.neighborhood_size:  # Within neighborhood
                    neighbors.append(kf)
        
        if len(neighbors) < 2:
            return 0.5  # Not enough neighbors for stability calculation
        
        # Calculate cosine similarities with neighbors
        similarities = []
        for neighbor in neighbors:
            neighbor_embedding = embeddings_cache.get(neighbor.key)
            if neighbor_embedding is not None:
                # Cosine similarity
                similarity = np.dot(target_embedding, neighbor_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(neighbor_embedding)
                )
                similarities.append(similarity)
        
        if not similarities:
            return 0.5
        
        # Calculate temporal stability
        std_dev = np.std(similarities)
        stability = 1.0 - min(1.0, 2.0 * std_dev)
        
        return max(0.0, min(1.0, stability))  # Clamp to [0, 1]
