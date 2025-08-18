"""
Performance Optimization Module for Competition
Implements speed and efficiency optimizations for real-time competition scenarios
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from schema.response import KeyframeServiceReponse
from schema.competition import MomentCandidate


class PerformanceOptimizer:
    """Optimizes system performance for competition time constraints"""
    
    def __init__(self, max_workers: int = 4, cache_size: int = 1000):
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.query_cache: Dict[str, Any] = {}
        self.embedding_cache: Dict[str, List[float]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0,
            "response_times": []
        }
    
    def cache_embedding(self, query: str, embedding: List[float]):
        """Cache embeddings to avoid recomputation"""
        if len(self.embedding_cache) < self.cache_size:
            self.embedding_cache[query] = embedding
    
    def get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Retrieve cached embedding"""
        return self.embedding_cache.get(query)
    
    async def parallel_keyframe_processing(
        self,
        keyframes: List[KeyframeServiceReponse],
        processing_func,
        batch_size: int = 10
    ) -> List[Any]:
        """Process keyframes in parallel batches for speed"""
        
        results = []
        for i in range(0, len(keyframes), batch_size):
            batch = keyframes[i:i + batch_size]
            
            # Process batch in parallel
            tasks = [processing_func(kf) for kf in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and add valid results
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
        
        return results
    
    def optimize_top_k_strategy(
        self,
        query_complexity: str,
        interactive_mode: bool = False
    ) -> Dict[str, int]:
        """
        Dynamic top_k optimization based on query complexity and mode
        
        Args:
            query_complexity: "simple", "moderate", "complex"
            interactive_mode: Whether in interactive/timed mode
        
        Returns:
            Optimized search parameters
        """
        
        if interactive_mode:
            # Optimize for speed in interactive scenarios
            base_top_k = 50
            rerank_limit = 10
        else:
            # Optimize for precision in automatic mode
            base_top_k = 200
            rerank_limit = 30
        
        complexity_multipliers = {
            "simple": 0.5,
            "moderate": 1.0, 
            "complex": 1.5
        }
        
        multiplier = complexity_multipliers.get(query_complexity, 1.0)
        
        return {
            "initial_top_k": int(base_top_k * multiplier),
            "rerank_top_k": int(rerank_limit * multiplier),
            "score_threshold": 0.1 if interactive_mode else 0.05
        }
    
    def smart_temporal_clustering(
        self,
        keyframes: List[KeyframeServiceReponse],
        target_moments: int = 10
    ) -> List[List[KeyframeServiceReponse]]:
        """
        Intelligent temporal clustering optimized for competition requirements
        """
        
        if not keyframes:
            return []
        
        # Sort keyframes by video and temporal order
        sorted_keyframes = sorted(
            keyframes,
            key=lambda x: (x.group_num, x.video_num, x.keyframe_num)
        )
        
        # Dynamic gap calculation based on keyframe density
        gaps = []
        for i in range(1, len(sorted_keyframes)):
            curr = sorted_keyframes[i]
            prev = sorted_keyframes[i-1]
            
            if (curr.group_num == prev.group_num and 
                curr.video_num == prev.video_num):
                gap = curr.keyframe_num - prev.keyframe_num
                gaps.append(gap)
        
        # Calculate adaptive threshold
        if gaps:
            median_gap = np.median(gaps)
            # Use larger threshold for sparser keyframes
            adaptive_threshold = max(median_gap * 2, 50)  # At least 2 seconds at 25fps
        else:
            adaptive_threshold = 125  # 5 seconds at 25fps
        
        # Cluster with adaptive threshold
        clusters = []
        current_cluster = [sorted_keyframes[0]]
        
        for i in range(1, len(sorted_keyframes)):
            curr = sorted_keyframes[i]
            prev = sorted_keyframes[i-1]
            
            # Different video = new cluster
            if (curr.group_num != prev.group_num or 
                curr.video_num != prev.video_num):
                clusters.append(current_cluster)
                current_cluster = [curr]
                continue
            
            # Check temporal gap
            gap = curr.keyframe_num - prev.keyframe_num
            if gap <= adaptive_threshold:
                current_cluster.append(curr)
            else:
                clusters.append(current_cluster)
                current_cluster = [curr]
        
        clusters.append(current_cluster)
        
        # Sort clusters by average confidence and limit
        cluster_scores = [
            (sum(kf.confidence_score for kf in cluster) / len(cluster), cluster)
            for cluster in clusters
        ]
        cluster_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [cluster for _, cluster in cluster_scores[:target_moments]]
    
    def track_performance(self, start_time: float, query: str, result_count: int):
        """Track performance metrics for optimization"""
        
        response_time = time.time() - start_time
        
        self.performance_stats["total_queries"] += 1
        self.performance_stats["response_times"].append(response_time)
        
        # Update rolling average (last 100 queries)
        recent_times = self.performance_stats["response_times"][-100:]
        self.performance_stats["avg_response_time"] = np.mean(recent_times)
        
        # Log performance warnings
        if response_time > 10.0:  # Competition time limits
            print(f"WARNING: Slow query ({response_time:.2f}s): '{query[:50]}...'")
        
        if result_count == 0:
            print(f"WARNING: No results for query: '{query}'")
    
    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get suggestions for system optimization"""
        
        stats = self.performance_stats
        suggestions = []
        
        if stats["avg_response_time"] > 5.0:
            suggestions.append("Consider reducing top_k or increasing score_threshold")
        
        if stats["cache_hits"] / max(stats["total_queries"], 1) < 0.3:
            suggestions.append("Query patterns have low cache utilization")
        
        if len(stats["response_times"]) > 0:
            percentile_95 = np.percentile(stats["response_times"], 95)
            if percentile_95 > 15.0:
                suggestions.append("95th percentile response time exceeds competition limits")
        
        return {
            "current_performance": stats,
            "optimization_suggestions": suggestions,
            "recommended_settings": self._get_recommended_settings()
        }
    
    def _get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings based on performance history"""
        
        avg_time = self.performance_stats["avg_response_time"]
        
        if avg_time > 8.0:
            # Optimize for speed
            return {
                "top_k": 100,
                "score_threshold": 0.2,
                "enable_reranking": False,
                "max_moments": 50
            }
        elif avg_time < 2.0:
            # Can afford more precision
            return {
                "top_k": 300,
                "score_threshold": 0.05,
                "enable_reranking": True,
                "max_moments": 100
            }
        else:
            # Balanced settings
            return {
                "top_k": 200,
                "score_threshold": 0.1,
                "enable_reranking": True,
                "max_moments": 75
            }


class CompetitionModeOptimizer:
    """Specialized optimizer for different competition modes"""
    
    @staticmethod
    def get_automatic_mode_settings() -> Dict[str, Any]:
        """Optimized settings for automatic track (precision focus)"""
        return {
            "top_k": 300,
            "score_threshold": 0.05,
            "enable_reranking": True,
            "enable_query_expansion": True,
            "temporal_clustering_gap": 5.0,
            "max_moments": 100,
            "asr_weight": 0.3,
            "visual_weight": 0.7
        }
    
    @staticmethod
    def get_interactive_mode_settings() -> Dict[str, Any]:
        """Optimized settings for interactive track (speed focus)"""
        return {
            "top_k": 100,
            "score_threshold": 0.1,
            "enable_reranking": False,  # Skip for speed
            "enable_query_expansion": False,  # Skip for speed
            "temporal_clustering_gap": 3.0,
            "max_moments": 20,
            "asr_weight": 0.4,
            "visual_weight": 0.6
        }
    
    @staticmethod
    def get_kis_mode_settings() -> Dict[str, Any]:
        """Optimized settings for KIS tasks (precision focus)"""
        return {
            "top_k": 150,
            "score_threshold": 0.3,  # Higher threshold for exact matching
            "enable_reranking": True,
            "enable_query_expansion": False,  # Don't expand for exact matching
            "temporal_clustering_gap": 1.0,  # Tight clustering
            "max_moments": 10,
            "asr_weight": 0.5,
            "visual_weight": 0.5
        }
