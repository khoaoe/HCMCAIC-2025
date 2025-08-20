import os
import sys
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
                response.append(
                    KeyframeServiceReponse(
                        key=keyframe.key,
                        video_num=keyframe.video_num,
                        group_num=keyframe.group_num,
                        keyframe_num=keyframe.keyframe_num,
                        confidence_score=result.distance
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
                response.append(
                    KeyframeServiceReponse(
                        key=kf.key,
                        video_num=kf.video_num,
                        group_num=kf.group_num,
                        keyframe_num=kf.keyframe_num,
                        confidence_score=result.distance,
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

        if response or not filters_applied:
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
                    video_num=kf.video_num,
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
        video_id format: "Lxx/Vxxx" or "group/video"
        """
        
        # Parse video_id to extract group_num and video_num
        if '/' in video_id:
            parts = video_id.replace('L', '').replace('V', '').split('/')
            try:
                group_num = int(parts[0])
                video_num = int(parts[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid video_id format: {video_id}")
        else:
            raise ValueError(f"video_id must be in format 'Lxx/Vxxx' or 'group/video'")

        return await self.search_by_text_temporal(
            text_embedding=text_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            start_time=start_time,
            end_time=end_time,
            video_nums=[video_num],
            group_nums=[group_num]
        )


    




    
        



        

        

        
        
        


        

        








