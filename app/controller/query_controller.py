from pathlib import Path
import json
from typing import List, Dict, Optional, Any

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from service import ModelService, KeyframeQueryService
from service.temporal_search import TemporalSearchService
from schema.response import KeyframeServiceReponse
from schema.competition import MomentCandidate
from agent.temporal_localization import TemporalLocalizer


class QueryController:
    
    def __init__(
        self,
        data_folder: Path,
        id2index_path: Path,
        model_service: ModelService,
        keyframe_service: KeyframeQueryService
    ):
        self.data_folder = data_folder
        self.id2index = json.load(open(id2index_path, 'r'))
        self.model_service = model_service
        self.keyframe_service = keyframe_service
        
        # Initialize temporal search with GRAB framework
        self.temporal_search_service = TemporalSearchService(
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=str(data_folder),
            optimization_level="balanced"
        )

    
    def convert_model_to_path(
        self,
        model: KeyframeServiceReponse
    ) -> tuple[str, float]:
        # Dataset structure: <DATA_FOLDER>/Lxx/kaggle/working/Lxx/Vxxx/<frame>.webp
        return os.path.join(
            self.data_folder,
            f"L{model.group_num:02d}",
            f"V{model.video_num:03d}",
            f"{model.keyframe_num:08d}.webp",
        ), model.confidence_score
    
        
    async def search_text(
        self, 
        query: str,
        top_k: int,
        score_threshold: float
    ):
        embedding = self.model_service.embedding(query).tolist()[0]

        result = await self.keyframe_service.search_by_text(embedding, top_k, score_threshold)
        return result


    async def search_text_with_exlude_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_group_exlude: list[int]
    ):
        exclude_ids = [
            int(k) for k, v in self.id2index.items()
            if int(v.split('/')[0]) in list_group_exlude
        ]

        
        
        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result


    async def search_with_selected_video_group(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        list_of_include_groups: list[int]  ,
        list_of_include_videos: list[int]  
    ):     
        

        exclude_ids = None
        if len(list_of_include_groups) > 0   and len(list_of_include_videos) == 0:
            print("hi")
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[0]) not in list_of_include_groups
            ]
        
        elif len(list_of_include_groups) == 0   and len(list_of_include_videos) >0 :
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if int(v.split('/')[1]) not in list_of_include_videos
            ]

        elif len(list_of_include_groups) == 0  and len(list_of_include_videos) == 0 :
            exclude_ids = []
        else:
            exclude_ids = [
                int(k) for k, v in self.id2index.items()
                if (
                    int(v.split('/')[0]) not in list_of_include_groups or
                    int(v.split('/')[1]) not in list_of_include_videos
                )
            ]



        embedding = self.model_service.embedding(query).tolist()[0]
        result = await self.keyframe_service.search_by_text_exclude_ids(embedding, top_k, score_threshold, exclude_ids)
        return result
    

    async def search_text_temporal(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        start_time: float | None = None,
        end_time: float | None = None,
        video_id: str | None = None,
        include_groups: list[int] | None = None,
        include_videos: list[int] | None = None
    ):
        """
        Search keyframes with temporal filtering using native Milvus scalar fields
        """
        
        # Parse video_id if provided
        group_nums = include_groups
        video_nums = include_videos
        
        if video_id:
            parts = video_id.replace('L', '').replace('V', '').split('/')
            try:
                group_num = int(parts[0])
                video_num = int(parts[1])
                # Override with specific video if provided
                group_nums = [group_num]
                video_nums = [video_num]
            except (ValueError, IndexError):
                raise ValueError(f"Invalid video_id format: {video_id}")

        embedding = self.model_service.embedding(query).tolist()[0]
        
        result = await self.keyframe_service.search_by_text_temporal(
            text_embedding=embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            start_time=start_time,
            end_time=end_time,
            video_nums=video_nums,
            group_nums=group_nums
        )
        return result

    async def search_text_time_window(
        self,
        query: str,
        video_id: str,
        start_time: float,
        end_time: float,
        top_k: int = 50,
        score_threshold: float = 0.1
    ):
        """
        Search within a specific time window of a video
        """
        embedding = self.model_service.embedding(query).tolist()[0]
        
        result = await self.keyframe_service.search_by_text_time_window(
            text_embedding=embedding,
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            top_k=top_k,
            score_threshold=score_threshold
        )
        return result
        

    async def temporal_search(
        self,
        query: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        video_id: Optional[str] = None,
        top_k: int = 50,
        score_threshold: float = 0.1,
        optimization_level: str = "balanced"
    ) -> List[MomentCandidate]:
        """
        Temporal search using GRAB framework optimizations
        
        Returns precisely localized temporal moments with:
        - Shot detection and strategic keyframe sampling
        - Perceptual hash deduplication
        - SuperGlobal reranking with GeM pooling
        - Adaptive Bidirectional Temporal Search (ABTS) for boundary detection
        """
        
        # Update optimization level if different
        if optimization_level != "balanced":
            self.temporal_search_service.optimization_config = self.temporal_search_service._get_optimization_config(optimization_level)
        
        moments = await self.temporal_search_service.temporal_search(
            query=query,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            top_k=top_k,
            score_threshold=score_threshold,
            enable_boundary_refinement=True
        )
        
        return moments

    async def search_temporal_moments(
        self,
        query: str,
        start_time: float,
        end_time: float,
        video_id: Optional[str] = None,
        precision_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Search for temporal moments with detailed analysis
        Returns both keyframes and precisely localized moments
        """
        
        optimization_level = "precision" if precision_mode else "balanced"
        
        # Get enhanced moments using GRAB framework
        moments = await self.temporal_search(
            query=query,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            top_k=20,
            score_threshold=0.1,
            optimization_level=optimization_level
        )
        
        # Also get traditional keyframe results for comparison
        traditional_keyframes = await self.search_text_temporal(
            query=query,
            top_k=50,
            score_threshold=0.1,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id
        )
        
        # Get temporal search statistics
        stats = await self.temporal_search_service.get_temporal_search_stats()
        
        return {
            "query": query,
            "time_range": {"start": start_time, "end": end_time},
            "video_id": video_id,
            "enhanced_moments": [
                {
                    "video_id": moment.video_id,
                    "start_time": moment.start_time,
                    "end_time": moment.end_time,
                    "duration": moment.end_time - moment.start_time,
                    "confidence": moment.confidence_score,
                    "evidence_keyframes": moment.evidence_keyframes
                }
                for moment in moments
            ],
            "traditional_keyframes_count": len(traditional_keyframes),
            "enhanced_moments_count": len(moments),
            "performance_stats": stats,
            "optimization_applied": optimization_level
        }
        

