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

from .agent import VisualEventExtractor, AnswerGenerator, COCO_CLASS
from schema.competition import MomentCandidate
from schema.agent import QueryAnalysisResult

from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse


from utils.video_utils import safe_convert_video_num


def apply_diversity_enhancement(
    keyframes: List[KeyframeServiceReponse], 
    max_results: int = 10
) -> List[KeyframeServiceReponse]:
    """
    Apply diversity enhancement to ensure results come from different videos/temporal segments.
    Simple implementation of Maximum Marginal Relevance (MMR) concept.
    """
    if len(keyframes) <= max_results:
        return keyframes
    
    selected = []
    remaining = keyframes.copy()
    
    # Select the highest scoring keyframe first
    if remaining:
        selected.append(remaining.pop(0))
    
    # Iteratively select diverse keyframes
    while len(selected) < max_results and remaining:
        best_candidate = None
        best_score = -1
        
        for candidate in remaining:
            # Calculate diversity penalty based on similarity to already selected
            diversity_penalty = 0.0
            
            for selected_kf in selected:
                # Penalize if from same video and similar temporal position
                if (candidate.video_num == selected_kf.video_num and 
                    candidate.group_num == selected_kf.group_num):
                    
                    # Calculate temporal distance penalty
                    temporal_diff = abs(int(candidate.keyframe_num) - int(selected_kf.keyframe_num))
                    if temporal_diff < 50:  # Within 50 frames
                        diversity_penalty += 0.3
                    elif temporal_diff < 100:  # Within 100 frames
                        diversity_penalty += 0.1
            
            # Final score = original confidence - diversity penalty
            final_score = candidate.confidence_score - diversity_penalty
            
            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        else:
            break
    
    return selected


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
        self.llm = llm  # Store LLM for reranking
        
        # Import prompts for query analysis
        from .prompts import CompetitionPrompts
        self.query_analyzer_prompt = CompetitionPrompts.QUERY_ANALYSIS_PROMPT

    
    async def process_query(self, user_query: str) -> str:
        """
        Quy trình xử lý truy vấn thích ứng theo triết lý "Từ 'Hoặc' đến 'Và' và 'Tùy chỉnh'":
        1. Phân tích truy vấn để xác định chiến lược (object-centric, action-centric, etc.)
        2. Thực hiện tìm kiếm kết hợp (hybrid search) để có tập ứng viên rộng
        3. Ưu tiên điểm số (soft filtering) nếu có đối tượng quan trọng
        4. Luôn thực hiện LLM Re-ranking trên top N ứng viên để có độ chính xác cao nhất
        5. Tạo câu trả lời tổng hợp
        """

        # Bước 1: Phân tích truy vấn để có chiến lược
        query_analysis = await self._analyze_query_intent(user_query)
        print(f"[Strategy] Query type: {query_analysis.query_type}, Key objects: {query_analysis.key_objects}")

        # Bước 2: Tìm kiếm kết hợp (Hybrid Search) - tương tự đề xuất trước
        agent_response = await self.query_extractor.extract_visual_events(user_query)
        query_variations = agent_response.query_variations or [agent_response.refined_query]
        
        # Thêm cả truy vấn gốc để đảm bảo không mất ý định người dùng
        if user_query not in query_variations:
            query_variations.insert(0, user_query)

        print(f"Original query: {user_query}")
        print(f"Query variations: {query_variations}")

        # Thực hiện tìm kiếm với tất cả các biến thể
        all_keyframes = []
        for i, variation in enumerate(query_variations):
            print(f"Searching with variation {i+1}: {variation}")
            
            embedding = self.model_service.embedding(variation).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=self.top_k * 2, # Lấy nhiều hơn để có không gian cho việc xếp hạng lại
                score_threshold=0.1
            )
            
            # Add source information to track which variation found each keyframe
            for kf in keyframes:
                kf.search_source = f"variation_{i+1}"
            
            all_keyframes.extend(keyframes)
            print(f"Variation {i+1} found {len(keyframes)} keyframes")
        
        print(f"Total keyframes from hybrid search: {len(all_keyframes)}")

        # Stage 3: Deduplication and sorting
        unique_keyframes = deduplicate_keyframes(all_keyframes)
        print(f"After deduplication: {len(unique_keyframes)} unique keyframes")
        
        # Bước 3: Ưu tiên điểm số (Soft Filtering) dựa trên kết quả phân tích
        final_candidates = unique_keyframes
        if query_analysis.query_type == 'object-centric' and query_analysis.key_objects:
            print(f"Applying score boost for objects: {query_analysis.key_objects}")
            target_objects_set = {obj.lower() for obj in query_analysis.key_objects}
            boosted_keyframes = []
            for kf in unique_keyframes:
                keyy = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
                keyframe_objects_set = {obj.lower() for obj in self.objects_data.get(keyy, [])}
                
                # Nếu có đối tượng khớp, tăng điểm (boost) thay vì lọc bỏ
                if target_objects_set.intersection(keyframe_objects_set):
                    kf.confidence_score = min(1.0, kf.confidence_score * 1.15) # Tăng 15%
                boosted_keyframes.append(kf)
            
            # Sắp xếp lại danh sách sau khi đã tăng điểm
            boosted_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
            final_candidates = boosted_keyframes
        
        # Bước 4: Luôn thực hiện LLM Re-ranking trên Top N ứng viên
        # Take top candidates for reranking (50-100 for good coverage)
        rerank_candidates = final_candidates[:min(50, len(final_candidates))]
        
        if len(rerank_candidates) > 1:
            print(f"Applying LLM reranking to {len(rerank_candidates)} candidates")
            
            # Convert to MomentCandidate format for reranker
            
            candidates = []
            for kf in rerank_candidates:
                # Get objects for this keyframe
                keyy = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
                objects = self.objects_data.get(keyy, [])
                
                # Get ASR text for this video
                asr_text = ""
                for entry in self.asr_data.values():
                    if isinstance(entry, dict) and entry.get("file_path") == f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}":
                        if "result" in entry:
                            asr_entries = entry["result"]
                            asr_segments = []
                            for seg in asr_entries:
                                if isinstance(seg, dict):
                                    start_frame = int(seg.get("start_frame", 0))
                                    end_frame = int(seg.get("end_frame", 0))
                                    kf_frame = int(kf.keyframe_num)
                                    if start_frame <= kf_frame <= end_frame:
                                        text = seg.get("text", "").strip()
                                        if text:
                                            asr_segments.append(text)
                            asr_text = " ".join(asr_segments)
                        break
                
                candidate = MomentCandidate(
                    video_id=f"{kf.group_num}_{kf.video_num}",
                    start_time=float(kf.keyframe_num) / 30.0,  # Assuming 30 fps
                    end_time=float(kf.keyframe_num) / 30.0 + 1.0,
                    confidence_score=kf.confidence_score,
                    detected_objects=objects,
                    asr_text=asr_text
                )
                candidates.append(candidate)
            
            # Apply LLM reranking
            try:
                from .competition_agent import LLMReranker
                llm_reranker = LLMReranker(self.llm)
                reranked_candidates = await llm_reranker.rerank_candidates(
                    query=user_query,
                    candidates=candidates
                )
                
                # Update confidence scores in original keyframes
                for i, candidate in enumerate(reranked_candidates):
                    if i < len(rerank_candidates):
                        rerank_candidates[i].confidence_score = candidate.confidence_score
                
                print(f"LLM reranking completed successfully")
                
            except Exception as e:
                print(f"Warning: LLM reranking failed: {e}")
                # Continue with original scores if reranking fails
        
        # Stage 5: Apply diversity enhancement
        final_keyframes_for_answer = rerank_candidates[:self.top_k]
        print(f"Final keyframes for answer: {len(final_keyframes_for_answer)} keyframes")
        
        if not final_keyframes_for_answer:
            return "Không tìm thấy kết quả phù hợp với truy vấn của bạn."
        
        # Stage 6: Group keyframes by video for temporal analysis
        video_groups = {}
        for kf in final_keyframes_for_answer:
            video_key = f"{kf.group_num}_{kf.video_num}"
            if video_key not in video_groups:
                video_groups[video_key] = []
            video_groups[video_key].append(kf)
        
        # Stage 7: Extract temporal information from ALL keyframes across ALL videos
        # Process each video group separately to maintain temporal context
        all_video_segments = []
        
        for video_key, video_keyframes in video_groups.items():
            if not video_keyframes:
                continue
                
            # Sort keyframes by frame number for this video
            sorted_keyframes = sorted(video_keyframes, key=lambda x: int(x.keyframe_num))
            smallest_kf = sorted_keyframes[0]
            max_kf = sorted_keyframes[-1]
            
            group_num = smallest_kf.group_num
            video_num = smallest_kf.video_num
            
            print(f"Processing video {video_key}: {len(video_keyframes)} keyframes")
            print(f"Temporal range: {smallest_kf.keyframe_num} to {max_kf.keyframe_num}")
            
            # Extract ASR text for this video's temporal segment
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
            
            # Store video segment information
            video_segment = {
                'video_key': video_key,
                'group_num': group_num,
                'video_num': video_num,
                'keyframes': video_keyframes,
                'temporal_range': (int(smallest_kf.keyframe_num), int(max_kf.keyframe_num)),
                'asr_text': asr_text,
                'avg_confidence': sum(kf.confidence_score for kf in video_keyframes) / len(video_keyframes)
            }
            all_video_segments.append(video_segment)
        
        # Sort video segments by average confidence score
        all_video_segments.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        print(f"Processed {len(all_video_segments)} video segments")
        for segment in all_video_segments:
            print(f"Video {segment['video_key']}: {len(segment['keyframes'])} keyframes, avg confidence: {segment['avg_confidence']:.3f}")

        # Stage 8: Generate comprehensive answer with information from all videos
        # Combine ASR text from all relevant videos
        combined_asr_text = ""
        if all_video_segments:
            # Use ASR from the highest confidence video as primary, others as supplementary
            primary_asr = all_video_segments[0]['asr_text']
            supplementary_asr = []
            
            for segment in all_video_segments[1:]:
                if segment['asr_text'] and segment['asr_text'] != primary_asr:
                    supplementary_asr.append(f"[Video {segment['video_key']}]: {segment['asr_text']}")
            
            combined_asr_text = primary_asr
            if supplementary_asr:
                combined_asr_text += " " + " ".join(supplementary_asr)
        
        # Create video summary information
        video_summary = f"Found relevant content in {len(all_video_segments)} videos: "
        video_details = []
        for i, segment in enumerate(all_video_segments):
            if i == 0:
                video_details.append(f"Primary: Video {segment['video_key']} ({len(segment['keyframes'])} keyframes)")
            else:
                video_details.append(f"Video {segment['video_key']} ({len(segment['keyframes'])} keyframes)")
        
        video_summary += "; ".join(video_details)
        print(f"Video summary: {video_summary}")
        
        answer = await self.answer_generator.generate_answer(
            original_query=user_query,
            final_keyframes=final_keyframes_for_answer,
            objects_data=self.objects_data,
            asr_data=combined_asr_text
        )

        return cast(str, answer)

    async def _analyze_query_intent(self, user_query: str) -> QueryAnalysisResult:
        """
        Sử dụng LLM để phân tích và phân loại truy vấn của người dùng.
        """
        if self.llm is None:
            # Fallback cho trường hợp không có LLM
            return QueryAnalysisResult(
                query_type='action-centric',
                key_objects=[],
                requires_contextual_understanding=True
            )

        prompt = self.query_analyzer_prompt.format(query=user_query, coco=COCO_CLASS)
        try:
            response = await self.llm.as_structured_llm(QueryAnalysisResult).acomplete(prompt)
            analysis = cast(QueryAnalysisResult, response.raw)
            print(f"[Query Analysis] Type: {analysis.query_type}, Objects: {analysis.key_objects}, Context Needed: {analysis.requires_contextual_understanding}")
            return analysis
        except Exception as e:
            print(f"Warning: Query analysis failed: {e}. Defaulting to robust strategy.")
            # Fallback an toàn khi LLM lỗi
            return QueryAnalysisResult(
                query_type='action-centric',
                key_objects=[],
                requires_contextual_understanding=True
            )

        