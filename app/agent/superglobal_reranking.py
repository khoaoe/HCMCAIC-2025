"""
SuperGlobal Reranking Module
Implements GRAB framework's advanced reranking using Generalized Mean (GeM) pooling
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from schema.response import KeyframeServiceReponse
from service.model_service import ModelService


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


class GeMPooling:
    """
    Generalized Mean Pooling for feature refinement
    Following GRAB paper's SuperGlobal reranking approach
    """
    
    def __init__(self, p_database: float = 1.0, p_query: float = float('inf')):
        """
        Initialize GeM pooling parameters
        p_database = 1.0 (average pooling for database image refinement)
        p_query = inf (max pooling for query expansion)
        """
        self.p_database = p_database
        self.p_query = p_query
    
    def gem_pool(self, features: np.ndarray, p: float) -> np.ndarray:
        """
        Apply Generalized Mean pooling
        f_k = (1/|X_k| * Σ_{x∈X_k} x^{p_k})^{1/p_k}
        """
        
        if p == float('inf'):
            # Max pooling
            return np.max(features, axis=0)
        elif p == 1.0:
            # Average pooling
            return np.mean(features, axis=0)
        else:
            # General GeM pooling
            powered = np.power(np.abs(features), p)
            pooled = np.power(np.mean(powered, axis=0), 1.0 / p)
            return pooled * np.sign(np.mean(features, axis=0))
    
    def refine_database_features(
        self, 
        features: List[np.ndarray],
        neighbor_indices: Dict[int, List[int]]
    ) -> List[np.ndarray]:
        """
        Refine database features using average pooling with neighbors
        """
        
        refined_features = []
        
        for i, feature in enumerate(features):
            neighbors = neighbor_indices.get(i, [i])  # Include self if no neighbors
            neighbor_features = [features[j] for j in neighbors if j < len(features)]
            
            if neighbor_features:
                neighbor_array = np.stack(neighbor_features)
                refined_feature = self.gem_pool(neighbor_array, self.p_database)
                refined_features.append(refined_feature)
            else:
                refined_features.append(feature)
        
        return refined_features
    
    def expand_query_features(
        self, 
        query_embedding: np.ndarray,
        top_k_features: List[np.ndarray],
        k_expansion: int = 5
    ) -> np.ndarray:
        """
        Expand query using max pooling with top-K similar features
        """
        
        if not top_k_features:
            return query_embedding
        
        # Take top-k most similar features for expansion
        expansion_features = top_k_features[:k_expansion]
        expansion_features.append(query_embedding)  # Include original query
        
        expansion_array = np.stack(expansion_features)
        expanded_query = self.gem_pool(expansion_array, self.p_query)
        
        return expanded_query


class SuperGlobalReranker:
    """
    Implements SuperGlobal Reranking from GRAB framework
    Uses GeM pooling for feature refinement and two-stage similarity calculation
    """
    
    def __init__(
        self,
        model_service: ModelService,
        neighbor_k: int = 10,
        expansion_k: int = 5,
        lambda_s1: float = 0.5,
        lambda_s2: float = 0.5
    ):
        self.model_service = model_service
        self.neighbor_k = neighbor_k
        self.expansion_k = expansion_k
        self.lambda_s1 = lambda_s1  # Weight for S1 (original query vs refined DB)
        self.lambda_s2 = lambda_s2  # Weight for S2 (expanded query vs original DB)
        self.gem_pooling = GeMPooling()
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    async def rerank_keyframes(
        self,
        query: str,
        keyframes: List[KeyframeServiceReponse],
        data_folder: str,
        initial_top_k: int = 100
    ) -> List[KeyframeServiceReponse]:
        """
        Rerank keyframes using SuperGlobal approach
        Returns reranked list with improved relevance scores
        """
        
        if len(keyframes) <= 1:
            return keyframes
        
        print(f"SuperGlobal reranking {len(keyframes)} keyframes...")
        
        # Step 1: Get embeddings for all keyframes
        keyframe_embeddings = await self._get_keyframe_embeddings(keyframes, data_folder)
        
        # Step 2: Get query embedding
        query_embedding = self.model_service.embedding(query)[0]  # Get first row
        
        # Step 3: Find neighbors for each keyframe
        neighbor_indices = self._find_neighbors(keyframe_embeddings)
        
        # Step 4: Refine database features using GeM pooling
        refined_features = self.gem_pooling.refine_database_features(
            keyframe_embeddings, neighbor_indices
        )
        
        # Step 5: Calculate S1 (original query vs refined database)
        s1_scores = []
        for refined_feature in refined_features:
            similarity = self._cosine_similarity(query_embedding, refined_feature)
            s1_scores.append(similarity)
        
        # Step 6: Get top-K features for query expansion
        initial_similarities = [
            self._cosine_similarity(query_embedding, feat) 
            for feat in keyframe_embeddings
        ]
        top_k_indices = np.argsort(initial_similarities)[-self.expansion_k:][::-1]
        top_k_features = [keyframe_embeddings[i] for i in top_k_indices]
        
        # Step 7: Expand query using GeM pooling
        expanded_query = self.gem_pooling.expand_query_features(
            query_embedding, top_k_features, self.expansion_k
        )
        
        # Step 8: Calculate S2 (expanded query vs original database)
        s2_scores = []
        for original_feature in keyframe_embeddings:
            similarity = self._cosine_similarity(expanded_query, original_feature)
            s2_scores.append(similarity)
        
        # Step 9: Combine scores and rerank
        final_scores = []
        for i in range(len(keyframes)):
            s1 = s1_scores[i]
            s2 = s2_scores[i]
            final_score = (self.lambda_s1 * s1 + self.lambda_s2 * s2) / (self.lambda_s1 + self.lambda_s2)
            final_scores.append(final_score)
        
        # Update keyframe confidence scores and sort
        reranked_keyframes = []
        for i, kf in enumerate(keyframes):
            kf_copy = KeyframeServiceReponse(
                key=kf.key,
                video_num=safe_convert_video_num(kf.video_num),
                group_num=kf.group_num,
                keyframe_num=kf.keyframe_num,
                confidence_score=final_scores[i]
            )
            reranked_keyframes.append(kf_copy)
        
        # Sort by final score
        reranked_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        print(f"SuperGlobal reranking complete. Score improvement: {final_scores[0]:.3f} -> {reranked_keyframes[0].confidence_score:.3f}")
        
        return reranked_keyframes
    
    async def _get_keyframe_embeddings(
        self,
        keyframes: List[KeyframeServiceReponse],
        data_folder: str
    ) -> List[np.ndarray]:
        """
        Get or compute embeddings for keyframes
        Uses image-based embedding extraction
        """
        
        embeddings = []
        
        for kf in keyframes:
            cache_key = f"{kf.group_num}_{kf.video_num}_{kf.keyframe_num}"
            
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                # Compute embedding from image
                image_path = os.path.join(
                    data_folder,
                    f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
                )
                
                if os.path.exists(image_path):
                    # Extract real embedding from image using model service
                    try:
                        embedding = self.model_service.embed_image(image_path)
                        embeddings.append(embedding)
                        self.embedding_cache[cache_key] = embedding
                    except Exception as e:
                        print(f"Error extracting embedding for {image_path}: {e}")
                        # Fallback to zero embedding on error
                        zero_embedding = np.zeros(512, dtype=np.float32)
                        embeddings.append(zero_embedding)
                        self.embedding_cache[cache_key] = zero_embedding
                else:
                    # Fallback to zero embedding if image not found
                    zero_embedding = np.zeros(512, dtype=np.float32)
                    embeddings.append(zero_embedding)
        
        return embeddings
    
    def _find_neighbors(
        self,
        embeddings: List[np.ndarray]
    ) -> Dict[int, List[int]]:
        """
        Find k-nearest neighbors for each embedding
        """
        
        if len(embeddings) <= 1:
            return {0: [0]}
        
        neighbor_indices = {}
        
        for i, embedding in enumerate(embeddings):
            similarities = []
            for j, other_embedding in enumerate(embeddings):
                if i != j:
                    sim = self._cosine_similarity(embedding, other_embedding)
                    similarities.append((sim, j))
            
            # Sort by similarity and take top-k
            similarities.sort(reverse=True)
            k = min(self.neighbor_k, len(similarities))
            neighbors = [j for _, j in similarities[:k]]
            neighbors.append(i)  # Include self
            
            neighbor_indices[i] = neighbors
        
        return neighbor_indices
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)


class GRABTemporalSearchOptimizer:
    """
    Integrates GRAB framework optimizations into temporal search
    """
    
    def __init__(
        self,
        model_service: ModelService,
        data_folder: str,
        enable_shot_detection: bool = True,
        enable_deduplication: bool = True,
        enable_superglobal_reranking: bool = True
    ):
        self.model_service = model_service
        self.data_folder = data_folder
        self.enable_shot_detection = enable_shot_detection
        self.enable_deduplication = enable_deduplication
        self.enable_superglobal_reranking = enable_superglobal_reranking
        
        # Initialize components
        self.shot_detector = ShotDetector() if enable_shot_detection else None
        self.deduplicator = PerceptualHashDeduplicator() if enable_deduplication else None
        self.reranker = SuperGlobalReranker(model_service) if enable_superglobal_reranking else None
        self.stability_analyzer = TemporalStabilityAnalyzer()
    
    async def optimize_temporal_search_results(
        self,
        query: str,
        keyframes: List[KeyframeServiceReponse]
    ) -> List[KeyframeServiceReponse]:
        """
        Apply GRAB framework optimizations to temporal search results
        """
        
        if not keyframes:
            return keyframes
        
        optimized_keyframes = keyframes.copy()
        
        # Stage 1: Shot-based keyframe selection
        if self.enable_shot_detection and self.shot_detector:
            shots = self.shot_detector.detect_shots_from_keyframes(optimized_keyframes)
            representative_keyframes = []
            
            for shot in shots:
                shot_keyframes = [
                    kf for kf in optimized_keyframes 
                    if kf.keyframe_num in shot.representative_keyframes
                ]
                representative_keyframes.extend(shot_keyframes)
            
            if representative_keyframes:
                optimized_keyframes = representative_keyframes
                print(f"Shot detection: {len(keyframes)} → {len(optimized_keyframes)} representative keyframes")
        
        # Stage 2: Perceptual deduplication
        if self.enable_deduplication and self.deduplicator:
            optimized_keyframes = self.deduplicator.deduplicate_keyframes(
                optimized_keyframes, self.data_folder
            )
        
        # Stage 3: SuperGlobal reranking
        if self.enable_superglobal_reranking and self.reranker and len(optimized_keyframes) > 1:
            optimized_keyframes = await self.reranker.rerank_keyframes(
                query, optimized_keyframes, self.data_folder
            )
        
        return optimized_keyframes
    
    def calculate_grab_confidence_score(
        self,
        semantic_similarity: float,
        temporal_stability: float,
        lambda_s: float = 0.7,
        lambda_t: float = 0.3
    ) -> float:
        """
        Calculate GRAB confidence score combining semantic and temporal factors
        c_i = λ_s * s_i + λ_t * t_i
        """
        
        return lambda_s * semantic_similarity + lambda_t * temporal_stability
