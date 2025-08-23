import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from typing import List, cast
from llama_index.core.llms import LLM

from .agent import VisualEventExtractor, AnswerGenerator

from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse


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


def apply_object_filter(
        keyframes: List[KeyframeServiceReponse], 
        objects_data: dict[str, list[str]], 
        target_objects: List[str]
    ) -> List[KeyframeServiceReponse]:
        
        if not target_objects:
            return keyframes
        
        target_objects_set = {obj.lower() for obj in target_objects}
        filtered_keyframes = []

        for kf in keyframes:
            keyy = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
            keyframe_objects = objects_data.get(keyy, [])
            print(f"{keyy=}")
            print(f"{keyframe_objects=}")
            keyframe_objects_set = {obj.lower() for obj in keyframe_objects}
            
            if target_objects_set.intersection(keyframe_objects_set):
                filtered_keyframes.append(kf)

        print(f"{filtered_keyframes=}")
        return filtered_keyframes


def deduplicate_keyframes(keyframes: List[KeyframeServiceReponse]) -> List[KeyframeServiceReponse]:
    """Remove duplicate keyframes and fuse scores with multi-query fusion"""
    
    keyframe_map = {}
    keyframe_sources = {}  # Track which query variations found each keyframe
    
    for kf in keyframes:
        key = f"{kf.group_num}_{kf.video_num}_{kf.keyframe_num}"
        
        if key in keyframe_map:
            # Fuse scores using weighted combination
            existing_kf = keyframe_map[key]
            
            # Count how many different query variations found this keyframe
            source = getattr(kf, 'search_source', 'unknown')
            if key not in keyframe_sources:
                keyframe_sources[key] = set()
            keyframe_sources[key].add(source)
            
            # Boost score if found by multiple variations (indicates high relevance)
            variation_bonus = 0.1 * (len(keyframe_sources[key]) - 1)
            
            # Take the maximum score plus variation bonus
            if kf.confidence_score > existing_kf.confidence_score:
                kf.confidence_score = min(kf.confidence_score + variation_bonus, 1.0)
                keyframe_map[key] = kf
            else:
                existing_kf.confidence_score = min(existing_kf.confidence_score + variation_bonus, 1.0)
                keyframe_map[key] = existing_kf
        else:
            keyframe_map[key] = kf
            source = getattr(kf, 'search_source', 'unknown')
            keyframe_sources[key] = {source}
    
    # Sort by fused confidence score
    unique_keyframes = list(keyframe_map.values())
    unique_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
    
    print(f"Deduplication with fusion: {len(keyframes)} → {len(unique_keyframes)} keyframes")
    
    return unique_keyframes


class KeyframeSearchAgent:
    def __init__(
        self, 
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: dict[str, list[str]],
        asr_data: dict[str, str | list[dict[str,str]]],
        top_k: int = 10
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.data_folder = data_folder
        self.top_k = top_k

        self.objects_data = objects_data or {}
        self.asr_data = asr_data or {}

        self.query_extractor = VisualEventExtractor(llm)
        self.answer_generator = AnswerGenerator(llm, data_folder)

    
    async def process_query(self, user_query: str) -> str:
        """
        Main agent flow:
        1. Extract visual/event elements and rephrase query
        2. Search for top-K keyframes using rephrased query
        3. Process ALL keyframes from ALL videos (CORPUS-LEVEL SEARCH)
        4. Apply deduplication and temporal clustering
        5. Optionally apply COCO object filtering
        6. Generate final answer with visual context
        """

        agent_response = await self.query_extractor.extract_visual_events(user_query)
        search_query = agent_response.refined_query
        suggested_objects = agent_response.list_of_objects
        query_variations = agent_response.query_variations or [search_query]

        print(f"{search_query=}")
        print(f"{suggested_objects=}")
        print(f"Query variations: {query_variations}")

        # Stage 1: Multi-query retrieval with semantic variations for robust search
        all_keyframes = []
        
        for i, variation in enumerate(query_variations):
            print(f"Searching with variation {i+1}: {variation}")
            
            embedding = self.model_service.embedding(variation).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=self.top_k * 2,  # Get more per variation for better coverage
                score_threshold=0.1
            )
            
            # Add source information to track which variation found each keyframe
            for kf in keyframes:
                if not hasattr(kf, 'search_source'):
                    kf.search_source = f"variation_{i+1}"
            
            all_keyframes.extend(keyframes)
            print(f"Variation {i+1} found {len(keyframes)} keyframes")
        
        print(f"Total keyframes from all variations: {len(all_keyframes)}")
        top_k_keyframes = all_keyframes

        # Stage 2: FIXED - Process ALL keyframes from ALL videos (CORPUS-LEVEL SEARCH)
        # Remove the flawed "best video only" logic
        print(f"Retrieved {len(top_k_keyframes)} keyframes from all videos")
        
        # Stage 3: Deduplication and sorting
        unique_keyframes = deduplicate_keyframes(top_k_keyframes)
        print(f"After deduplication: {len(unique_keyframes)} unique keyframes")
        
        # Stage 4: Apply object filtering if suggested
        final_keyframes = unique_keyframes
        print(f"Length of keyframes before objects {len(final_keyframes)}")
        if suggested_objects:
            filtered_keyframes = apply_object_filter(
                keyframes=unique_keyframes,
                objects_data=self.objects_data,
                target_objects=suggested_objects
            )
            if filtered_keyframes:  
                final_keyframes = filtered_keyframes
        print(f"Length of keyframes after objects {len(final_keyframes)}")
        
        # Stage 5: Select top results for answer generation
        # Take top keyframes across all videos, not just one video
        final_keyframes = final_keyframes[:self.top_k]
        
        if not final_keyframes:
            return "Không tìm thấy kết quả phù hợp với truy vấn của bạn."
        
        # Stage 6: Extract temporal information from the best keyframes
        # Handle multiple videos properly
        video_groups = {}
        for kf in final_keyframes:
            video_key = f"{kf.group_num}_{kf.video_num}"
            if video_key not in video_groups:
                video_groups[video_key] = []
            video_groups[video_key].append(kf)
        
        # Find the video with the highest average confidence score
        best_video_key = max(video_groups.keys(), 
                           key=lambda k: sum(kf.confidence_score for kf in video_groups[k]) / len(video_groups[k]))
        best_video_keyframes = video_groups[best_video_key]
        
        smallest_kf = min(best_video_keyframes, key=lambda x: int(x.keyframe_num))
        max_kf = max(best_video_keyframes, key=lambda x: int(x.keyframe_num))

        print(f"Best video keyframes: {len(best_video_keyframes)} from video {best_video_key}")
        print(f"{smallest_kf=}")
        print(f"{max_kf=}")

        group_num = smallest_kf.group_num
        video_num = smallest_kf.video_num

        print(f"{group_num}")
        print(f"{video_num}")
        print(f"L{str(group_num):0>2s}/L{str(group_num):0>2s}_V{str(video_num):0>3s}")
        
        # Extract ASR text for the temporal segment
        matching_asr = None
        for entry in self.asr_data.values():
            if isinstance(entry, dict) and entry.get("file_path") == f"L{str(group_num):0>2s}/L{str(group_num):0>2s}_V{str(video_num):0>3s}":
                matching_asr = entry
                break
        
        asr_text = ""
        if matching_asr and "result" in matching_asr:
            asr_entries = matching_asr["result"]
            asr_text_segments = []
            for seg in asr_entries:
                if isinstance(seg, dict):
                    start_frame = int(seg.get("start_frame", 0))
                    end_frame = int(seg.get("end_frame", 0))
                    if (int(smallest_kf.keyframe_num) <= start_frame <= int(max_kf.keyframe_num) or
                        int(smallest_kf.keyframe_num) <= end_frame <= int(max_kf.keyframe_num)):
                        text = seg.get("text", "").strip()
                        if text:
                            asr_text_segments.append(text)
            asr_text = " ".join(asr_text_segments)
        print(f"ASR text for segment: {asr_text[:200]}...")

        # Stage 7: Generate answer with all relevant keyframes from all videos
        # Include information about which videos were found
        video_info = f"Find results from {len(video_groups)} videos. "
        if len(video_groups) > 1:
            video_info += f"Main video: {best_video_key}, other videos: {', '.join([k for k in video_groups.keys() if k != best_video_key])}"
        
        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes,
            objects_data=self.objects_data,
            asr_data=asr_text
        )

        return cast(str, answer)

        