import os
import sys
import numpy as np
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)


from repository.milvus import KeyframeVectorRepository
from repository.milvus import MilvusSearchRequest
from repository.mongo import KeyframeRepository
from repository.video_metadata import VideoMetadataRepository

from schema.response import KeyframeServiceReponse


from utils.video_utils import safe_convert_video_num


class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            video_metadata_repo: VideoMetadataRepository = None,
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo
        self.video_metadata_repo = video_metadata_repo


    async def _retrieve_keyframes(self, ids: list[int]):
        keyframes = await self.keyframe_mongo_repo.get_keyframe_by_list_of_keys(ids)
        print(keyframes[:5])
  
        keyframe_map = {k.key: k for k in keyframes}
        return_keyframe = [
            keyframe_map[k] for k in ids
        ]   
        return return_keyframe

    async def _search_keyframes(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        exclude_indices: list[int] | None = None
    ) -> list[KeyframeServiceReponse]:
        
        # Ensure embedding data type consistency
        if isinstance(text_embedding, np.ndarray):
            # Convert to float32 for consistency
            text_embedding = text_embedding.astype(np.float32).tolist()
        elif isinstance(text_embedding, list):
            # Convert list to float32
            text_embedding = [float(x) for x in text_embedding]
        
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            exclude_ids=exclude_indices
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        sorted_ids = [result.id_ for result in sorted_results]

        keyframes = await self._retrieve_keyframes(sorted_ids)



        keyframe_map = {k.key: k for k in keyframes}
        response = []

        for result in sorted_results:
            keyframe = keyframe_map.get(result.id_) 
            if keyframe is not None:
                # Debug: Check if video_num is corrupted
                if isinstance(keyframe.video_num, str) and '_V' in keyframe.video_num:
                    print(f"WARNING: Corrupted video_num detected: {keyframe.video_num} for keyframe {keyframe.key}")
                
                response.append(
                    KeyframeServiceReponse(
                        key=int(keyframe.key),
                        video_num=safe_convert_video_num(keyframe.video_num),
                        group_num=int(keyframe.group_num),
                        keyframe_num=int(keyframe.keyframe_num),
                        confidence_score=float(result.distance),
                        embedding=result.embedding,
                        phash=keyframe.phash
                    )
                )
        return response
    

    async def search_by_text(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = 0.5,
    ):
        return await self._search_keyframes(text_embedding, top_k, score_threshold, None)
    
    async def search_hybrid(
        self,
        text_embedding: list[float],
        query: str,
        top_k: int = 100,
        score_threshold: float | None = 0.5,
        filter_author: str | None = None,
        filter_keywords: list[str] | None = None,
        filter_publish_date: str | None = None,
        metadata_weight: float = 0.3
    ) -> list[KeyframeServiceReponse]:
        """
        Hybrid search: Filter-then-Rank.
        1) Use metadata filters (and optional text) to get candidate videos.
        2) Restrict Milvus visual search to those videos via scalar filters.
        3) Boost scores for results whose (group_num, video_num) matched metadata.
        """
        # Normalize embedding
        if isinstance(text_embedding, np.ndarray):
            text_embedding = text_embedding.astype(np.float32).tolist()
        else:
            text_embedding = [float(x) for x in text_embedding]

        # Step 1: metadata filter set
        matched_pairs: set[tuple[int, int]] = set()
        group_nums: list[int] | None = None
        video_nums: list[int] | None = None

        if self.video_metadata_repo and (filter_author or filter_keywords or filter_publish_date or (query and query.strip())):
            try:
                # If only filters present: use filters-only. If query present, include text query.
                if filter_author or filter_keywords or filter_publish_date:
                    pairs = await self.video_metadata_repo.search_by_filters(
                        filter_author=filter_author,
                        filter_keywords=filter_keywords,
                        filter_publish_date=filter_publish_date,
                        limit=500,
                    )
                else:
                    # No explicit filters, but hybrid requested: use text across metadata
                    docs = await self.video_metadata_repo.search_by_text(
                        query=query,
                        limit=500,
                    )
                    pairs = [(int(d.group_num), int(d.video_num)) for d in docs]
                matched_pairs = set(pairs)
                if matched_pairs:
                    group_nums = list({g for g, _ in matched_pairs})
                    video_nums = list({v for _, v in matched_pairs})
            except Exception as e:
                print(f"Metadata filtering failed: {e}")
                matched_pairs = set()
                group_nums = None
                video_nums = None

        # Step 2: visual search restricted by scalar filters (if any)
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 2,
            score_threshold=None,
            start_time=None,
            end_time=None,
            video_nums=video_nums,
            group_nums=group_nums,
        )
        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        # Step 3: backfill from Mongo and compute boosted scores
        filtered = [r for r in search_response.results if score_threshold is None or r.distance > score_threshold]
        sorted_results = sorted(filtered, key=lambda r: r.distance, reverse=True)

        ids = [r.id_ for r in sorted_results]
        keyframes = await self._retrieve_keyframes(ids)
        kf_map = {k.key: k for k in keyframes}

        responses: list[KeyframeServiceReponse] = []
        for r in sorted_results:
            kf = kf_map.get(r.id_)
            if not kf:
                continue
            g = int(kf.group_num)
            v = safe_convert_video_num(kf.video_num)
            score = float(r.distance)
            # Boost if this video's metadata matched
            if matched_pairs and (g, v) in matched_pairs:
                score = score + float(metadata_weight) * (1.0 - min(score, 1.0))
            responses.append(
                KeyframeServiceReponse(
                    key=int(kf.key),
                    video_num=v,
                    group_num=g,
                    keyframe_num=int(kf.keyframe_num),
                    confidence_score=score,
                    embedding=r.embedding,
                    phash=kf.phash,
                )
            )

        # Truncate to top_k
        responses.sort(key=lambda x: x.confidence_score, reverse=True)
        return responses[:top_k]
    
    async def _search_metadata(
        self,
        query: str,
        top_k: int,
        filter_author: str | None = None,
        filter_keywords: list[str] | None = None,
        filter_publish_date: str | None = None
    ) -> list[KeyframeServiceReponse]:
        """
        Deprecated: Kept for compatibility. Previously returned synthetic keyframes.
        Now unused by search_hybrid.
        """
        return []
    
    def _fuse_results_rrf(
        self,
        visual_results: list[KeyframeServiceReponse],
        metadata_results: list[KeyframeServiceReponse],
        metadata_weight: float,
        top_k: int
    ) -> list[KeyframeServiceReponse]:
        """
        Legacy RRF fusion (kept for compatibility with callers that still use it).
        """
        # Create score maps for both result sets
        visual_scores = {}
        metadata_scores = {}
        
        # Process visual results
        for i, result in enumerate(visual_results):
            video_key = f"{result.group_num}_{result.video_num}_{result.keyframe_num}"
            visual_scores[video_key] = 1.0 / (60 + i)  # RRF formula with k=60
        
        # Process metadata results
        for i, result in enumerate(metadata_results):
            video_key = f"{result.group_num}_{result.video_num}_{result.keyframe_num}"
            metadata_scores[video_key] = metadata_weight * (1.0 / (60 + i))
        
        # Combine scores
        combined_scores = {}
        all_keys = set(visual_scores.keys()) | set(metadata_scores.keys())
        
        for key in all_keys:
            visual_score = visual_scores.get(key, 0.0)
            metadata_score = metadata_scores.get(key, 0.0)
            combined_scores[key] = visual_score + metadata_score
        
        # Sort by combined score
        sorted_keys = sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)
        
        # Reconstruct results
        result_map = {f"{r.group_num}_{r.video_num}_{r.keyframe_num}": r for r in visual_results}
        result_map.update({f"{r.group_num}_{r.video_num}_{r.keyframe_num}": r for r in metadata_results})
        
        fused_results = []
        for key in sorted_keys[:top_k]:
            if key in result_map:
                fused_results.append(result_map[key])
        
        return fused_results   
    

    async def search_by_text_range(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        range_queries: list[tuple[int,int]]
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """

        all_ids = self.keyframe_vector_repo.get_all_id()
        allowed_ids = set()
        for start, end in range_queries:
            allowed_ids.update(range(start, end + 1))
        
        
        exclude_ids = [id_ for id_ in all_ids if id_ not in allowed_ids]

        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    

    async def search_by_text_exclude_ids(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None,
        exclude_ids: list[int] | None
    ):
        """
        range_queries: a bunch of start end indices, and we just search inside these, ignore everything
        """
        return await self._search_keyframes(text_embedding, top_k, score_threshold, exclude_ids)   
    

    async def search_by_text_temporal(
        self,
        text_embedding: list[float],
        top_k: int,
        score_threshold: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        video_nums: list[int] | None = None,
        group_nums: list[int] | None = None
    ) -> list[KeyframeServiceReponse]:
        """
        Temporal search using native Milvus scalar field filtering
        Filters by timestamp range and/or specific videos/groups
        """
        
        search_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k,
            start_time=start_time,
            end_time=end_time,
            video_nums=video_nums,
            group_nums=group_nums
        )

        search_response = await self.keyframe_vector_repo.search_by_embedding(search_request)

        # Apply score threshold filtering
        filtered_results = [
            result for result in search_response.results
            if score_threshold is None or result.distance > score_threshold
        ]

        sorted_results = sorted(
            filtered_results, key=lambda r: r.distance, reverse=True
        )

        # Convert to KeyframeServiceReponse with temporal metadata
        # Always attempt to backfill from MongoDB to avoid synthetic defaults
        response = []

        # Always build a backfill map using Mongo so we can resolve metadata reliably
        keyframe_map = {}
        try:
            ids = [r.id_ for r in sorted_results]
            keyframes = await self._retrieve_keyframes(ids)
            keyframe_map = {k.key: k for k in keyframes}
        except Exception:
            keyframe_map = {}

        for result in sorted_results:
            # Prefer authoritative metadata from Mongo (backfill)
            kf = keyframe_map.get(result.id_)
            if kf is not None:
                # Debug: Check if video_num is corrupted
                if isinstance(kf.video_num, str) and '_V' in kf.video_num:
                    print(f"WARNING: Corrupted video_num detected in temporal search: {kf.video_num} for keyframe {kf.key}")
                
                response.append(
                    KeyframeServiceReponse(
                        key=kf.key,
                        video_num=safe_convert_video_num(kf.video_num),
                        group_num=kf.group_num,
                        keyframe_num=kf.keyframe_num,
                        confidence_score=result.distance,
                        embedding=result.embedding,
                        phash=kf.phash
                    )
                )
                continue

            # If Milvus provided complete metadata, use it
            if (
                result.video_num is not None
                and result.group_num is not None
                and result.keyframe_num is not None
            ):
                response.append(
                    KeyframeServiceReponse(
                        key=result.id_,
                        video_num=result.video_num,
                        group_num=result.group_num,
                        keyframe_num=result.keyframe_num,
                        confidence_score=result.distance,
                        embedding=result.embedding,
                        phash=None  # No phash available from Milvus metadata
                    )
                )
                continue

            # Otherwise, skip ambiguous entries instead of defaulting to L01/V001
            # to avoid misleading paths in results
            try:
                print(
                    f"Skipping result {result.id_}: missing metadata and no Mongo backfill available"
                )
            except Exception:
                pass

        # If no results returned but filters were applied, fallback to non-temporal search
        filters_applied = any([
            start_time is not None and end_time is not None,
            video_nums is not None and len(video_nums) > 0,
            group_nums is not None and len(group_nums) > 0,
        ])

        # If we got enough results or no filters applied, return
        if response or not filters_applied:
            # If we have some results but fewer than requested and filters were applied,
            # augment using a non-temporal fallback and post-filtering by metadata.
            if filters_applied and 0 < len(response) < top_k:
                pass
            return response

        # Fallback path (should rarely execute due to above return)
        return response


    




    
        



        

        

        
        
        


        

        








