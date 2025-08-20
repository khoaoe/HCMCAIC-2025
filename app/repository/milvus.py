"""
The implementation of Vector Repository. The following class is responsible for getting the vector by many ways
Including Faiss and Usearch
"""


import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from typing import cast
from common.repository import MilvusBaseRepository
from pymilvus import Collection as MilvusCollection
from pymilvus.client.search_result import SearchResult
from schema.interface import  MilvusSearchRequest, MilvusSearchResult, MilvusSearchResponse


class KeyframeVectorRepository(MilvusBaseRepository):
    def __init__(
        self, 
        collection: MilvusCollection,
        search_params: dict
    ):
        
        super().__init__(collection)
        self.search_params = search_params
    
    async def search_by_embedding(
        self,
        request: MilvusSearchRequest
    ):
        # Build expression filter for temporal search
        expr_clauses = []
        
        # Check if temporal fields exist in the collection schema
        collection_fields = [field.name for field in self.collection.schema.fields]
        has_temporal_fields = "timestamp" in collection_fields
        
        if request.exclude_ids:
            expr_clauses.append(f"id not in {request.exclude_ids}")
        
        # Only add temporal filters if temporal fields exist
        if has_temporal_fields:
            if request.start_time is not None and request.end_time is not None:
                expr_clauses.append(f"timestamp >= {request.start_time} and timestamp <= {request.end_time}")
            
            if request.video_nums:
                video_nums_str = str(request.video_nums).replace('[', '').replace(']', '')
                expr_clauses.append(f"video_num in [{video_nums_str}]")
            
            if request.group_nums:
                group_nums_str = str(request.group_nums).replace('[', '').replace(']', '')
                expr_clauses.append(f"group_num in [{group_nums_str}]")
        
        expr = " and ".join(expr_clauses) if expr_clauses else None
        
        # Determine output fields based on collection schema
        base_output_fields = ["id", "embedding"]
        temporal_fields = ["timestamp", "group_num", "video_num", "keyframe_num"]
        
        # Reuse the collection_fields from above
        available_temporal_fields = [field for field in temporal_fields if field in collection_fields]
        
        output_fields = base_output_fields + available_temporal_fields
        
        search_results= cast(SearchResult, self.collection.search(
            data=[request.embedding],
            anns_field="embedding",
            param=self.search_params,
            limit=request.top_k,
            expr=expr,
            output_fields=output_fields,
            _async=False
        ))


        results = []
        for hits in search_results:
            for hit in hits:
                # Extract metadata from search results (temporal fields optional)
                entity = getattr(hit, 'entity', {})
                result = MilvusSearchResult(
                    id_=hit.id,
                    distance=hit.distance,
                    embedding=entity.get("embedding") if entity else None,
                    timestamp=entity.get("timestamp") if entity else None,
                    group_num=entity.get("group_num") if entity else None,
                    video_num=entity.get("video_num") if entity else None,
                    keyframe_num=entity.get("keyframe_num") if entity else None
                )
                results.append(result)
        
        return MilvusSearchResponse(
            results=results,
            total_found=len(results),
        )
    
    def get_all_id(self) -> list[int]:
        return list(range(self.collection.num_entities))



    
    

