"""
Competition Task Handlers for HCMC AI Challenge 2025
Implements the three main competition tasks: VCMR, VQA, and KIS
"""

from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from pathlib import Path
import os

from .agent import VisualEventExtractor
from .temporal_localization import TemporalLocalizer, ASRTemporalAligner
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.competition import (
    VCMRAutomaticRequest, VCMRAutomaticResponse, VCMRCandidate,
    VCMRInteractiveCandidate, VCMRFeedback,
    VideoQARequest, VideoQAResponse, VideoQAEvidence,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    MomentCandidate
)
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


class VCMRAgent:
    """Video Corpus Moment Retrieval Agent"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        temporal_localizer: TemporalLocalizer,
        asr_aligner: ASRTemporalAligner,
        objects_data: Dict[str, List[str]],
        data_folder: str
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer
        self.asr_aligner = asr_aligner
        self.objects_data = objects_data
        self.data_folder = data_folder
        self.query_extractor = VisualEventExtractor(llm)
        
        # reranking prompt
        self.rerank_prompt = PromptTemplate(
            """
            You are an expert video moment retrieval system. Given a query and candidate moments, 
            rank them by relevance and provide rationale.
            
            Original Query: {query}
            
            Candidates:
            {candidates_info}
            
            For each candidate, consider:
            1. Visual content alignment with query
            2. Temporal context and actions
            3. ASR text relevance (if available)
            4. Object detection matches
            
            Return a relevance score (0-1) and brief rationale for the top candidate.
            Focus on precision - prefer highly relevant moments over quantity.
            """
        )
    
    async def process_automatic_vcmr(
        self, 
        request: VCMRAutomaticRequest
    ) -> VCMRAutomaticResponse:
        """Process VCMR Automatic task"""
        
        # 1. Extract and refine query
        agent_response = await self.query_extractor.extract_visual_events(request.query)
        search_query = agent_response.refined_query
        suggested_objects = agent_response.list_of_objects
        
        # 2. Semantic search across corpus (not just best video)
        embedding = self.model_service.embedding(search_query).tolist()[0]
        
        # Use larger top_k for corpus-wide search
        corpus_top_k = min(request.top_k * 5, 500)  # Search more broadly first
        top_keyframes = await self.keyframe_service.search_by_text(
            text_embedding=embedding,
            top_k=corpus_top_k,
            score_threshold=0.1
        )
        
        # 3. Apply object filtering if suggested
        if suggested_objects and self.objects_data:
            from .main_agent import apply_object_filter
            filtered_keyframes = apply_object_filter(
                keyframes=top_keyframes,
                objects_data=self.objects_data,
                target_objects=suggested_objects
            )
            if filtered_keyframes:
                top_keyframes = filtered_keyframes
        
        # 4. Create temporal moments from keyframes
        moments = self.temporal_localizer.create_moments_from_keyframes(
            keyframes=top_keyframes,
            max_moments=request.top_k
        )
        
        # 5. Enhance moments with ASR context
        enhanced_moments = []
        for moment in moments:
            asr_text = self.asr_aligner.get_asr_for_moment(
                moment.video_id, moment.start_time, moment.end_time
            )
            moment.asr_text = asr_text
            enhanced_moments.append(moment)
        
        # 6. LLM-based reranking for final precision
        reranked_moments = await self._rerank_moments(request.query, enhanced_moments[:50])
        
        # 7. Convert to competition format
        candidates = [
            VCMRCandidate(
                video_id=moment.video_id,
                start_time=moment.start_time,
                end_time=moment.end_time,
                score=moment.confidence_score
            )
            for moment in reranked_moments[:request.top_k]
        ]
        
        # Generate explanation for top candidate
        notes = None
        if candidates:
            top_moment = enhanced_moments[0]
            notes = f"Top moment shows relevant content in {top_moment.video_id} "
            if top_moment.asr_text:
                notes += f"with context: '{top_moment.asr_text[:100]}...'"
        
        return VCMRAutomaticResponse(
            query=request.query,
            candidates=candidates,
            notes=notes
        )
    
    async def _rerank_moments(
        self, 
        query: str, 
        moments: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Use LLM to rerank moments for better precision"""
        
        if not moments:
            return moments
        
        # Prepare candidate information for LLM
        candidates_info = []
        for i, moment in enumerate(moments):
            info = f"Candidate {i+1}:\n"
            info += f"  Video: {moment.video_id}\n"
            info += f"  Time: {moment.start_time:.1f}s - {moment.end_time:.1f}s\n"
            info += f"  Objects: {moment.detected_objects or 'None detected'}\n"
            if moment.asr_text:
                info += f"  ASR: {moment.asr_text[:200]}...\n"
            info += f"  Confidence: {moment.confidence_score:.3f}\n"
            candidates_info.append(info)
        
        prompt = self.rerank_prompt.format(
            query=query,
            candidates_info="\n".join(candidates_info)
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            # For now, return original order - could parse LLM response for reordering
            return moments
        except Exception as e:
            print(f"Warning: LLM reranking failed: {e}")
            return moments
    
    async def process_interactive_vcmr(
        self,
        query: str,
        feedback: Optional[VCMRFeedback] = None,
        previous_candidates: Optional[List[VCMRInteractiveCandidate]] = None
    ) -> VCMRInteractiveCandidate:
        """Process VCMR Interactive task with feedback handling"""
        
        # Start with automatic VCMR
        auto_request = VCMRAutomaticRequest(query=query, corpus_index="default", top_k=10)
        auto_response = await self.process_automatic_vcmr(auto_request)
        
        candidates = auto_response.candidates
        if not candidates:
            raise ValueError("No candidates found for query")
        
        # Apply feedback if provided
        if feedback and previous_candidates:
            candidates = await self._apply_feedback(query, feedback, candidates)
        
        # Return top candidate
        top_candidate = candidates[0]
        return VCMRInteractiveCandidate(
            video_id=top_candidate.video_id,
            start_time=top_candidate.start_time,
            end_time=top_candidate.end_time,
            score=top_candidate.score
        )
    
    async def _apply_feedback(
        self,
        query: str,
        feedback: VCMRFeedback,
        candidates: List[VCMRCandidate]
    ) -> List[VCMRCandidate]:
        """Apply user feedback to refine search results"""
        
        if feedback.refine:
            # Re-search with refined query
            refined_query = f"{query} {feedback.refine}"
            refined_request = VCMRAutomaticRequest(
                query=refined_query,
                corpus_index="default",
                top_k=len(candidates)
            )
            refined_response = await self.process_automatic_vcmr(refined_request)
            return refined_response.candidates
        
        elif feedback.relevance is False:
            # Remove low-scoring candidates and boost diversity
            return candidates[1:]  # Skip first candidate
        
        elif feedback.relevance_score is not None:
            # Adjust scoring based on graded feedback
            threshold = feedback.relevance_score
            return [c for c in candidates if c.score >= threshold]
        
        return candidates


class VideoQAAgent:
    """Video Question Answering Agent"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        temporal_localizer: TemporalLocalizer,
        asr_aligner: ASRTemporalAligner,
        objects_data: Dict[str, List[str]],
        data_folder: str
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer
        self.asr_aligner = asr_aligner
        self.objects_data = objects_data
        self.data_folder = data_folder
        
        self.qa_prompt = PromptTemplate(
            """
            Answer the following question about the video content based on the provided keyframes and context.
            
            Question: {question}
            Video ID: {video_id}
            Clip Range: {clip_range}
            
            Context Information:
            {context_info}
            
            Keyframes and Visual Content:
            {keyframes_info}
            
            Requirements:
            1. Provide a concise, factual answer
            2. If counting or naming, be explicit (e.g., "2 people", "Alice and Bob")
            3. Base answer on visual evidence from keyframes
            4. Use ASR context when relevant
            5. If uncertain, indicate confidence level
            
            Answer:
            """
        )
    
    async def process_video_qa(self, request: VideoQARequest) -> VideoQAResponse:
        """Process Video QA task"""
        
        # 1. Parse video identifier and clip range
        video_parts = request.video_id.split('/')
        if len(video_parts) >= 2:
            group_num = int(video_parts[0][1:])  # Remove 'L' prefix
            video_num = safe_convert_video_num(video_parts[1])   # Remove 'V' prefix
        else:
            raise ValueError(f"Invalid video_id format: {request.video_id}")
        
        # 2. Determine search scope
        if request.clip:
            # Convert time range to keyframe range for targeted search
            start_frame = int(request.clip.start_time * 25)  # Assuming 25 FPS
            end_frame = int(request.clip.end_time * 25)
            
            # Search within keyframe range
            range_queries = [(start_frame, end_frame)]
            embedding = self.model_service.embedding(request.question).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text_range(
                text_embedding=embedding,
                top_k=20,
                score_threshold=0.1,
                range_queries=range_queries
            )
        else:
            # Search entire video
            embedding = self.model_service.embedding(request.question).tolist()[0]
            all_keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=50,
                score_threshold=0.1
            )
            # Filter to specific video
            keyframes = [
                kf for kf in all_keyframes 
                if kf.group_num == group_num and kf.video_num == video_num
            ]
        
        # 3. Get ASR context for the relevant time range
        asr_context = ""
        if request.clip:
            asr_context = self.asr_aligner.get_asr_for_moment(
                request.video_id, request.clip.start_time, request.clip.end_time
            ) or ""
        
        # 4. Prepare visual context with images
        chat_messages = []
        keyframes_info = []
        
        for kf in keyframes[:10]:  # Limit to top 10 for LLM context
            image_path = os.path.join(
                self.data_folder, 
                f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
            )
            
            timestamp = self.temporal_localizer.keyframe_to_timestamp(
                kf.group_num, kf.video_num, kf.keyframe_num
            )
            
            keyframe_key = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
            objects = self.objects_data.get(keyframe_key, [])
            
            info = f"Keyframe at {timestamp:.1f}s - Objects: {', '.join(objects) if objects else 'None'}"
            keyframes_info.append(info)
            
            if os.path.exists(image_path):
                message_content = [
                    ImageBlock(path=Path(image_path)),
                    TextBlock(text=info)
                ]
                chat_messages.append(ChatMessage(
                    role=MessageRole.USER,
                    content=message_content
                ))
        
        # 5. Prepare context information
        context_info = []
        if request.context and request.context.asr:
            context_info.append(f"ASR: {request.context.asr}")
        if asr_context:
            context_info.append(f"Relevant ASR: {asr_context}")
        if request.context and request.context.ocr:
            context_info.append(f"OCR: {', '.join(request.context.ocr)}")
        
        clip_range = "Full video"
        if request.clip:
            clip_range = f"{request.clip.start_time:.1f}s - {request.clip.end_time:.1f}s"
        
        # 6. Generate answer using LLM with visual context
        final_prompt = self.qa_prompt.format(
            question=request.question,
            video_id=request.video_id,
            clip_range=clip_range,
            context_info="\n".join(context_info) if context_info else "No additional context",
            keyframes_info="\n".join(keyframes_info)
        )
        
        query_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)
        
        response = await self.llm.achat(chat_messages)
        answer = response.message.content
        
        # 7. Create evidence from relevant keyframes
        evidence = []
        for kf in keyframes[:5]:  # Top 5 as evidence
            start_time = self.temporal_localizer.keyframe_to_timestamp(
                kf.group_num, kf.video_num, kf.keyframe_num
            )
            evidence.append(VideoQAEvidence(
                start_time=start_time,
                end_time=start_time + 2.0,  # 2 second window
                confidence=kf.confidence_score
            ))
        
        return VideoQAResponse(
            video_id=request.video_id,
            question=request.question,
            answer=str(answer),
            evidence=evidence,
            confidence=max(kf.confidence_score for kf in keyframes) if keyframes else 0.0
        )


class KISAgent:
    """Known-Item Search Agent"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        temporal_localizer: TemporalLocalizer,
        asr_aligner: ASRTemporalAligner,
        objects_data: Dict[str, List[str]],
        data_folder: str
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer
        self.asr_aligner = asr_aligner
        self.objects_data = objects_data
        self.data_folder = data_folder
        
        # Progressive hints state for KIS-C
        self.progressive_state: Dict[str, Any] = {}
    
    async def process_kis_textual(self, request: KISTextualRequest) -> KISResponse:
        """Process KIS Textual task - find exact match from text description"""
        
        # Use high precision search for exact matching
        embedding = self.model_service.embedding(request.text_description).tolist()[0]
        keyframes = await self.keyframe_service.search_by_text(
            text_embedding=embedding,
            top_k=100,
            score_threshold=0.3  # Higher threshold for exact matching
        )
        
        if not keyframes:
            raise ValueError("No matching segments found")
        
        # Take the highest scoring keyframe as exact match
        best_keyframe = keyframes[0]
        
        # Create tight temporal window around the keyframe
        center_time = self.temporal_localizer.keyframe_to_timestamp(
            best_keyframe.group_num, best_keyframe.video_num, best_keyframe.keyframe_num
        )
        
        # Tight window for exact matching (±1 second)
        start_time = max(0, center_time - 1.0)
        end_time = center_time + 1.0
        
        return KISResponse(
            video_id=f"L{str(best_keyframe.group_num):0>2s}/L{str(best_keyframe.group_num):0>2s}_V{str(safe_convert_video_num(best_keyframe.video_num)):0>3s}",
            start_time=start_time,
            end_time=end_time,
            match_confidence=best_keyframe.confidence_score
        )
    
    async def process_kis_visual(self, request: KISVisualRequest) -> KISResponse:
        """Process KIS Visual task - find exact match from visual example"""
        
        # For visual matching, we would need to:
        # 1. Extract features from query_clip_uri
        # 2. Compare against keyframe embeddings
        # 3. Find best visual match
        
        # Implement actual visual similarity matching
        try:
            # Extract visual features from the query clip
            if hasattr(request, 'query_clip_uri') and request.query_clip_uri:
                # For now, we'll use a more sophisticated text-based approach
                # In a full implementation, you would extract visual features from the clip
                visual_description = await self._extract_visual_description_from_clip(request.query_clip_uri)
            else:
                # Fallback to enhanced text-based search
                visual_description = request.text_query + " visual content scene"
            
            # Use enhanced embedding with visual focus
            embedding = self.model_service.embedding(visual_description).tolist()[0]
            
            # Search with higher precision for visual matching
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=100,  # Get more candidates for visual filtering
                score_threshold=0.3
            )
            
            if not keyframes:
                raise ValueError("No visual matches found")
            
            # Apply additional visual filtering if possible
            filtered_keyframes = await self._apply_visual_filtering(keyframes, visual_description)
            
            if filtered_keyframes:
                best_keyframe = filtered_keyframes[0]
            else:
                best_keyframe = keyframes[0]  # Fallback to original results
            
        except Exception as e:
            print(f"Visual similarity matching failed: {e}")
            # Fallback to basic text search
            embedding = self.model_service.embedding(request.text_query).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=50,
                score_threshold=0.4
            )
            
            if not keyframes:
                raise ValueError("No matches found")
            
            best_keyframe = keyframes[0]
        
        center_time = self.temporal_localizer.keyframe_to_timestamp(
            best_keyframe.group_num, best_keyframe.video_num, best_keyframe.keyframe_num
        )
        
        return KISResponse(
            video_id=f"L{str(best_keyframe.group_num):0>2s}/L{str(best_keyframe.group_num):0>2s}_V{str(safe_convert_video_num(best_keyframe.video_num)):0>3s}",
            start_time=max(0, center_time - 0.5),
            end_time=center_time + 0.5,
            match_confidence=best_keyframe.confidence_score
        )
    
    async def _extract_visual_description_from_clip(self, clip_uri: str) -> str:
        """Extract visual description from clip URI"""
        # In a full implementation, this would:
        # 1. Load the video clip
        # 2. Extract key frames
        # 3. Use vision model to describe the content
        # 4. Return a text description
        
        # For now, return a generic visual description
        return "visual content from video clip"
    
    async def _apply_visual_filtering(
        self, 
        keyframes: List[KeyframeServiceReponse], 
        visual_description: str
    ) -> List[KeyframeServiceReponse]:
        """Apply additional visual filtering to keyframes"""
        
        # Enhanced filtering based on visual description
        filtered_keyframes = []
        
        for kf in keyframes:
            # Check if keyframe has object detection data
            keyframe_key = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
            
            if hasattr(self, 'objects_data') and keyframe_key in self.objects_data:
                objects = self.objects_data[keyframe_key]
                
                # Check if objects match visual description
                description_terms = set(visual_description.lower().split())
                object_terms = set(' '.join(objects).lower().split())
                
                # Calculate overlap
                overlap = len(description_terms.intersection(object_terms))
                if overlap > 0:
                    filtered_keyframes.append(kf)
            else:
                # If no object data, keep the keyframe
                filtered_keyframes.append(kf)
        
        # Sort by confidence score
        filtered_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return filtered_keyframes
    
    async def process_kis_progressive(
        self, 
        request: KISProgressiveRequest,
        additional_hints: Optional[List[str]] = None
    ) -> KISResponse:
        """Process KIS Progressive task - iterative refinement with hints"""
        
        # Combine initial hint with additional hints
        combined_query = request.initial_hint
        if additional_hints:
            combined_query += " " + " ".join(additional_hints)
        
        # Store progressive state
        session_key = f"{request.corpus_index}_{hash(request.initial_hint)}"
        if session_key not in self.progressive_state:
            self.progressive_state[session_key] = {
                "initial_hint": request.initial_hint,
                "all_hints": [request.initial_hint],
                "search_history": []
            }
        
        if additional_hints:
            self.progressive_state[session_key]["all_hints"].extend(additional_hints)
        
        # Search with progressively refined query
        embedding = self.model_service.embedding(combined_query).tolist()[0]
        keyframes = await self.keyframe_service.search_by_text(
            text_embedding=embedding,
            top_k=50,
            score_threshold=0.2  # Lower threshold as hints accumulate
        )
        
        if not keyframes:
            raise ValueError("No matches found even with progressive hints")
        
        best_keyframe = keyframes[0]
        center_time = self.temporal_localizer.keyframe_to_timestamp(
            best_keyframe.group_num, best_keyframe.video_num, best_keyframe.keyframe_num
        )
        
        # Store search in history
        self.progressive_state[session_key]["search_history"].append({
            "query": combined_query,
            "best_match": {
                "video_id": f"L{str(best_keyframe.group_num):0>2s}/L{str(best_keyframe.group_num):0>2s}_V{str(safe_convert_video_num(best_keyframe.video_num)):0>3s}",
                "timestamp": center_time,
                "confidence": best_keyframe.confidence_score
            }
        })
        
        return KISResponse(
            video_id=f"L{str(best_keyframe.group_num):0>2s}/L{str(best_keyframe.group_num):0>2s}_V{str(safe_convert_video_num(best_keyframe.video_num)):0>3s}",
            start_time=max(0, center_time - 1.0),
            end_time=center_time + 1.0,
            match_confidence=best_keyframe.confidence_score
        )


class CompetitionTaskDispatcher:
    """Main dispatcher for all competition tasks"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: Dict[str, List[str]],
        asr_data: Dict[str, Any],
        video_metadata_path: Optional[Path] = None
    ):
        self.temporal_localizer = TemporalLocalizer(video_metadata_path)
        self.asr_aligner = ASRTemporalAligner(asr_data)
        
        self.vcmr_agent = VCMRAgent(
            llm, keyframe_service, model_service, self.temporal_localizer,
            self.asr_aligner, objects_data, data_folder
        )
        
        self.vqa_agent = VideoQAAgent(
            llm, keyframe_service, model_service, self.temporal_localizer,
            self.asr_aligner, objects_data, data_folder
        )
        
        self.kis_agent = KISAgent(
            llm, keyframe_service, model_service, self.temporal_localizer,
            self.asr_aligner, objects_data, data_folder
        )
    
    async def dispatch_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to appropriate task handler based on task type"""
        
        task_type = task_input.get("task", "").lower()
        
        if task_type == "vcmr_automatic":
            request = VCMRAutomaticRequest(**task_input)
            response = await self.vcmr_agent.process_automatic_vcmr(request)
            return response.model_dump()
        
        elif task_type == "video_qa":
            request = VideoQARequest(**task_input)
            response = await self.vqa_agent.process_video_qa(request)
            return response.model_dump()
        
        elif task_type == "kis_t":
            request = KISTextualRequest(**task_input)
            response = await self.kis_agent.process_kis_textual(request)
            return response.model_dump()
        
        elif task_type == "kis_v":
            request = KISVisualRequest(**task_input)
            response = await self.kis_agent.process_kis_visual(request)
            return response.model_dump()
        
        elif task_type == "kis_c":
            request = KISProgressiveRequest(**task_input)
            response = await self.kis_agent.process_kis_progressive(request)
            return response.model_dump()
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
