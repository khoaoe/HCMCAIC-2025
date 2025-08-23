"""
Adaptive Bidirectional Temporal Search (ABTS) Algorithm
Implements GRAB framework's precise temporal boundary detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import asyncio
from dataclasses import dataclass
import os

from schema.response import KeyframeServiceReponse
from schema.competition import MomentCandidate
from service.model_service import ModelService
from .shot_detection import TemporalStabilityAnalyzer


@dataclass
class ABTSResult:
    """Result from ABTS boundary detection"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence_score: float
    pivot_frame: int
    boundary_scores: Dict[str, float]


class AdaptiveBidirectionalTemporalSearch:
    """
    Implements ABTS algorithm for precise temporal boundary detection
    Following GRAB paper's methodology
    """
    
    def __init__(
        self,
        model_service: ModelService,
        lambda_s: float = 0.7,  # Semantic similarity weight
        lambda_t: float = 0.3,  # Temporal stability weight
        search_window: int = 75,  # Frames to search in each direction (3 seconds at 25fps)
        confidence_threshold: float = 0.3,
        fps: float = 25.0
    ):
        self.model_service = model_service
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.search_window = search_window
        self.confidence_threshold = confidence_threshold
        self.fps = fps
        
        self.stability_analyzer = TemporalStabilityAnalyzer()
    
    async def find_optimal_boundaries(
        self,
        query: str,
        pivot_keyframe: KeyframeServiceReponse,
        context_keyframes: List[KeyframeServiceReponse],
        data_folder: str
    ) -> ABTSResult:
        """
        Find optimal temporal boundaries using ABTS algorithm
        
        Args:
            query: Search query text
            pivot_keyframe: Initial keyframe to search from
            context_keyframes: All available keyframes for the video
            data_folder: Path to keyframe images
        
        Returns:
            ABTSResult with optimal boundaries and confidence scores
        """
        
        print(f"ABTS: Finding boundaries for pivot frame {pivot_keyframe.keyframe_num}")
        
        # Get query embedding
        query_embedding = self.model_service.embedding(query)[0]
        
        # Filter context keyframes to same video
        video_keyframes = [
            kf for kf in context_keyframes
            if kf.group_num == pivot_keyframe.group_num and kf.video_num == pivot_keyframe.video_num
        ]
        
        # Sort by temporal order
        video_keyframes.sort(key=lambda x: x.keyframe_num)
        
        # Find pivot index
        pivot_idx = None
        for i, kf in enumerate(video_keyframes):
            if kf.keyframe_num == pivot_keyframe.keyframe_num:
                pivot_idx = i
                break
        
        if pivot_idx is None:
            # Fallback to single frame
            return self._create_single_frame_result(pivot_keyframe)
        
        # Prepare embeddings cache for stability calculation
        embeddings_cache = await self._build_embeddings_cache(video_keyframes, data_folder)
        
        # Bidirectional search
        start_idx = await self._search_backward(
            query_embedding, video_keyframes, pivot_idx, embeddings_cache
        )
        
        end_idx = await self._search_forward(
            query_embedding, video_keyframes, pivot_idx, embeddings_cache
        )
        
        # Convert to temporal boundaries
        start_frame = video_keyframes[start_idx].keyframe_num
        end_frame = video_keyframes[end_idx].keyframe_num
        start_time = start_frame / self.fps
        end_time = end_frame / self.fps
        
        # Calculate overall confidence
        pivot_confidence = await self._calculate_confidence_score(
            query_embedding, video_keyframes[pivot_idx], video_keyframes, embeddings_cache
        )
        
        boundary_scores = {
            "start_confidence": await self._calculate_confidence_score(
                query_embedding, video_keyframes[start_idx], video_keyframes, embeddings_cache
            ),
            "end_confidence": await self._calculate_confidence_score(
                query_embedding, video_keyframes[end_idx], video_keyframes, embeddings_cache
            ),
            "pivot_confidence": pivot_confidence
        }
        
        result = ABTSResult(
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            confidence_score=pivot_confidence,
            pivot_frame=pivot_keyframe.keyframe_num,
            boundary_scores=boundary_scores
        )
        
        print(f"ABTS: Found boundaries [{start_time:.1f}s - {end_time:.1f}s] with confidence {pivot_confidence:.3f}")
        
        return result
    
    async def _search_backward(
        self,
        query_embedding: np.ndarray,
        video_keyframes: List[KeyframeServiceReponse],
        pivot_idx: int,
        embeddings_cache: Dict[int, np.ndarray]
    ) -> int:
        """Search backward from pivot to find optimal start boundary"""
        
        best_idx = pivot_idx
        best_confidence = await self._calculate_confidence_score(
            query_embedding, video_keyframes[pivot_idx], video_keyframes, embeddings_cache
        )
        
        # Search backward within window
        start_search = max(0, pivot_idx - self.search_window)
        
        for i in range(pivot_idx - 1, start_search - 1, -1):
            confidence = await self._calculate_confidence_score(
                query_embedding, video_keyframes[i], video_keyframes, embeddings_cache
            )
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_idx = i
            elif confidence < self.confidence_threshold:
                break  # Stop search if confidence drops too low
        
        return best_idx
    
    async def _search_forward(
        self,
        query_embedding: np.ndarray,
        video_keyframes: List[KeyframeServiceReponse],
        pivot_idx: int,
        embeddings_cache: Dict[int, np.ndarray]
    ) -> int:
        """Search forward from pivot to find optimal end boundary"""
        
        best_idx = pivot_idx
        best_confidence = await self._calculate_confidence_score(
            query_embedding, video_keyframes[pivot_idx], video_keyframes, embeddings_cache
        )
        
        # Search forward within window
        end_search = min(len(video_keyframes), pivot_idx + self.search_window)
        
        for i in range(pivot_idx + 1, end_search):
            confidence = await self._calculate_confidence_score(
                query_embedding, video_keyframes[i], video_keyframes, embeddings_cache
            )
            
            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_idx = i
            elif confidence < self.confidence_threshold:
                break  # Stop search if confidence drops too low
        
        return best_idx
    
    async def _calculate_confidence_score(
        self,
        query_embedding: np.ndarray,
        keyframe: KeyframeServiceReponse,
        all_keyframes: List[KeyframeServiceReponse],
        embeddings_cache: Dict[int, np.ndarray]
    ) -> float:
        """
        Calculate GRAB confidence score: c_i = λ_s * s_i + λ_t * t_i
        """
        
        keyframe_embedding = embeddings_cache.get(keyframe.key)
        if keyframe_embedding is None:
            return 0.0
        
        # Semantic similarity (s_i)
        semantic_sim = self._cosine_similarity(query_embedding, keyframe_embedding)
        
        # Temporal stability (t_i)
        temporal_stability = self.stability_analyzer.calculate_temporal_stability(
            all_keyframes, keyframe, embeddings_cache
        )
        
        # Combined confidence score
        confidence = self.lambda_s * semantic_sim + self.lambda_t * temporal_stability
        
        return confidence
    
    async def _build_embeddings_cache(
        self,
        keyframes: List[KeyframeServiceReponse],
        data_folder: str
    ) -> Dict[int, np.ndarray]:
        """Build cache of embeddings for stability calculation"""
        
        embeddings_cache = {}
        
        for kf in keyframes:
            try:
                # Build image path
                image_path = f"{data_folder}/L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(kf.video_num):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
                
                # Extract real embedding from image
                if os.path.exists(image_path):
                    embedding = self.model_service.embed_image(image_path)
                    embeddings_cache[kf.key] = embedding
                else:
                    print(f"Warning: Image not found for keyframe {kf.key}: {image_path}")
                    # Fallback to zero embedding if image not found
                    embedding = np.zeros(1024, dtype=np.float32)
                    embeddings_cache[kf.key] = embedding
                    
            except Exception as e:
                print(f"Error extracting embedding for keyframe {kf.key}: {e}")
                # Fallback to zero embedding on error
                embedding = np.zeros(1024, dtype=np.float32)
                embeddings_cache[kf.key] = embedding
        
        print(f"ABTS: Built embeddings cache with {len(embeddings_cache)} keyframes")
        return embeddings_cache
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _create_single_frame_result(self, keyframe: KeyframeServiceReponse) -> ABTSResult:
        """Create result for single frame when ABTS cannot be applied"""
        
        timestamp = keyframe.keyframe_num / self.fps
        
        return ABTSResult(
            start_frame=keyframe.keyframe_num,
            end_frame=keyframe.keyframe_num,
            start_time=timestamp,
            end_time=timestamp + 1.0,  # 1 second default duration
            confidence_score=keyframe.confidence_score,
            pivot_frame=keyframe.keyframe_num,
            boundary_scores={"fallback": keyframe.confidence_score}
        )
