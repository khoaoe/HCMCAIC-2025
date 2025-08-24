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

from schema.response import KeyframeServiceReponse


from utils.video_utils import safe_convert_video_num


class KeyframeQueryService:
    def __init__(
            self, 
            keyframe_vector_repo: KeyframeVectorRepository,
            keyframe_mongo_repo: KeyframeRepository,
            
        ):

        self.keyframe_vector_repo = keyframe_vector_repo
        self.keyframe_mongo_repo= keyframe_mongo_repo


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
                        embedding=result.embedding
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
                        embedding=result.embedding
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
                        embedding=result.embedding
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
                try:
                    fallback_request = MilvusSearchRequest(
                        embedding=text_embedding,
                        top_k=top_k * 10
                    )
                    fallback_results = await self.keyframe_vector_repo.search_by_embedding(fallback_request)

                    sorted_results_fb = sorted(
                        [r for r in fallback_results.results if score_threshold is None or r.distance > score_threshold],
                        key=lambda r: r.distance,
                        reverse=True,
                    )

                    ids_fb = [r.id_ for r in sorted_results_fb]
                    keyframe_map_fb = {k.key: k for k in (await self._retrieve_keyframes(ids_fb))}

                    fps = 25.0
                    augmented: list[KeyframeServiceReponse] = response[:]
                    for r in sorted_results_fb:
                        if any(k.key == r.id_ for k in augmented):
                            continue
                        kf = keyframe_map_fb.get(r.id_)
                        if not kf:
                            continue
                        if group_nums and kf.group_num not in group_nums:
                            continue
                        if video_nums and kf.video_num not in video_nums:
                            continue
                        if start_time is not None and end_time is not None:
                            ts = kf.keyframe_num / fps
                            if not (start_time <= ts <= end_time):
                                continue
                        augmented.append(
                            KeyframeServiceReponse(
                                key=kf.key,
                                video_num=safe_convert_video_num(kf.video_num),
                                group_num=kf.group_num,
                                keyframe_num=kf.keyframe_num,
                                confidence_score=r.distance,
                            )
                        )

                    augmented.sort(key=lambda x: x.confidence_score, reverse=True)
                    return augmented[:top_k]
                except Exception:
                    return response
            return response

        # Fallback: retrieve without temporal expression, then post-filter using Mongo metadata
        fallback_request = MilvusSearchRequest(
            embedding=text_embedding,
            top_k=top_k * 10  # fetch more to allow filtering
        )
        fallback_results = await self.keyframe_vector_repo.search_by_embedding(fallback_request)

        # Backfill metadata if needed using existing utility above
        sorted_results_fb = sorted(
            [r for r in fallback_results.results if score_threshold is None or r.distance > score_threshold],
            key=lambda r: r.distance,
            reverse=True,
        )

        # Build keyframe map for backfill
        ids_fb = [r.id_ for r in sorted_results_fb]
        keyframe_map_fb = {k.key: k for k in (await self._retrieve_keyframes(ids_fb))}

        fps = 25.0
        filtered_response: list[KeyframeServiceReponse] = []
        for r in sorted_results_fb:
            kf = keyframe_map_fb.get(r.id_)
            if not kf:
                continue

            # Apply video/group filters if provided
            if group_nums and kf.group_num not in group_nums:
                continue
            if video_nums and kf.video_num not in video_nums:
                continue

            # Apply time window if provided
            if start_time is not None and end_time is not None:
                ts = kf.keyframe_num / fps
                if not (start_time <= ts <= end_time):
                    continue

            filtered_response.append(
                KeyframeServiceReponse(
                    key=kf.key,
                    video_num=safe_convert_video_num(kf.video_num),
                    group_num=kf.group_num,
                    keyframe_num=kf.keyframe_num,
                    confidence_score=r.distance,
                )
            )

        filtered_response.sort(key=lambda x: x.confidence_score, reverse=True)
        return filtered_response[:top_k]

    async def search_by_text_time_window(
        self,
        text_embedding: list[float],
        video_id: str,
        start_time: float,
        end_time: float,
        top_k: int = 50,
        score_threshold: float | None = None
    ) -> list[KeyframeServiceReponse]:
        """
        Search within a specific time window of a video
        video_id format: "Lxx/Lxx_Vxxx"
        """
        
        # Parse video_id to extract group_num and video_num
        parts = video_id.split('/')
        if len(parts) >= 2:
            group_part = parts[0].replace('L', '')
            video_part = parts[1]
            
            # Handle different video part formats
            if video_part.startswith('V'):
                # Format: "V001" -> extract "001"
                video_num = int(video_part[1:])
            elif '_V' in video_part:
                # Format: "L20_V001" -> extract "001"
                video_num = int(video_part.split('_V')[-1])
            else:
                # Assume it's already a number
                video_num = int(video_part)
            
            group_num = int(group_part)
        else:
            raise ValueError(f"Invalid video_id format: {video_id}")

        return await self.search_by_text_temporal(
            text_embedding=text_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            start_time=start_time,
            end_time=end_time,
            video_nums=[video_num],
            group_nums=[group_num]
        )


    




    
        



        

        

        
        
        


        

        








