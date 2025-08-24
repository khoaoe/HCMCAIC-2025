"""
Multi-modal agent for HCMC AI Challenge 2025
Implements state-of-the-art techniques for video moment retrieval and QA
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from pathlib import Path
import os

from .enhanced_prompts import CompetitionPrompts
from .temporal_localization import TemporalLocalizer, ASRTemporalAligner
from .agent import VisualEventExtractor
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse
from schema.competition import MomentCandidate


from utils.video_utils import safe_convert_video_num


class QueryUnderstandingModule:
    """Advanced query understanding with entity extraction and temporal parsing"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.understanding_prompt = PromptTemplate(
            """
            Analyze the video search query to extract structured information for optimal retrieval.
            
            Query: {query}
            
            Extract and structure the following:
            
            1. **Entities**: People, objects, locations, proper nouns
            2. **Actions**: Verbs, activities, movements, interactions
            3. **Temporal Indicators**: Time references, sequence words, duration cues
            4. **Visual Attributes**: Colors, sizes, shapes, spatial relationships
            5. **Context**: Setting, environment, mood, style
            6. **Modality Hints**: Audio cues, speech, music, sound effects
            
            Return structured analysis that helps optimize multi-modal search:
            - Primary focus areas for embedding search
            - Secondary refinement criteria
            - Temporal constraints or preferences
            - Visual filter suggestions
            
            Format as JSON with clear categories for downstream processing.
            """
        )
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Deep query understanding for multi-modal search optimization"""
        prompt = self.understanding_prompt.format(query=query)
        
        try:
            response = await self.llm.acomplete(prompt)
            # Parse LLM response into structured format
            # For now, return basic structure - could enhance with structured output
            return {
                "original_query": query,
                "primary_focus": query,  # Enhanced analysis would go here
                "temporal_constraints": [],
                "visual_filters": [],
                "confidence": 0.8
            }
        except Exception as e:
            print(f"Warning: Query analysis failed: {e}")
            return {
                "original_query": query,
                "primary_focus": query,
                "temporal_constraints": [],
                "visual_filters": [],
                "confidence": 0.5
            }


class MultiModalRetriever:
    """Advanced retrieval with cross-modal fusion and reranking"""
    
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        temporal_localizer: TemporalLocalizer,
        asr_aligner: ASRTemporalAligner,
        objects_data: Dict[str, List[str]]
    ):
        self.llm = llm
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.temporal_localizer = temporal_localizer
        self.asr_aligner = asr_aligner
        self.objects_data = objects_data
        
        self.rerank_prompt = CompetitionPrompts.CROSS_MODAL_RERANKING
    
    async def retrieve_and_rank(
        self,
        query: str,
        top_k: int = 100,
        enable_reranking: bool = True,
        fusion_strategy: str = "late_fusion"
    ) -> List[MomentCandidate]:
        """
        Advanced multi-modal retrieval with fusion and reranking
        
        Args:
            query: Search query
            top_k: Number of results to return
            enable_reranking: Whether to use LLM reranking
            fusion_strategy: "early_fusion" or "late_fusion"
        """
        
        # Stage 1: Multi-query expansion for robust retrieval
        expanded_queries = await self._expand_query(query)
        
        # Stage 2: Parallel retrieval with different strategies
        all_keyframes = []
        for expanded_query in expanded_queries:
            embedding = self.model_service.embedding(expanded_query).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=top_k * 2,  # Retrieve more for fusion
                score_threshold=0.05
            )
            all_keyframes.extend(keyframes)
        
        # Stage 3: Deduplication and score fusion
        unique_keyframes = self._deduplicate_keyframes(all_keyframes)
        
        # Stage 4: Convert to moments with temporal clustering
        moments = self.temporal_localizer.create_moments_from_keyframes(
            keyframes=unique_keyframes,
            max_moments=top_k * 2
        )
        
        # Stage 5: Cross-modal enhancement
        enhanced_moments = await self._enhance_with_multimodal_context(moments)
        
        # Stage 6: LLM-based reranking if enabled
        if enable_reranking and len(enhanced_moments) > 1:
            reranked_moments = await self._llm_rerank(query, enhanced_moments[:20])
            return reranked_moments[:top_k]
        
        return enhanced_moments[:top_k]
    
    async def _expand_query(self, original_query: str) -> List[str]:
        """Generate query variations for robust retrieval"""
        expansion_prompt = PromptTemplate(
            """
            Generate 3-5 semantic variations of this video search query for robust retrieval.
            
            Original Query: {query}
            
            Create variations that:
            1. Use synonyms and alternative phrasings
            2. Focus on different aspects (visual, action, context)
            3. Vary specificity levels (broader and more specific)
            4. Consider alternative interpretations
            
            Return a list of query variations, each on a new line.
            Keep the original query as the first variation.
            """
        )
        
        try:
            prompt = expansion_prompt.format(query=original_query)
            response = await self.llm.acomplete(prompt)
            
            # Parse response into list of queries
            variations = [original_query]  # Always include original
            response_text = str(response).strip()
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line and line not in variations:
                    # Remove numbering and bullet points
                    cleaned = line.lstrip('123456789.- ').strip()
                    if cleaned and len(cleaned) > 10:  # Valid query length
                        variations.append(cleaned)
            
            return variations[:5]  # Limit to 5 variations
            
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            return [original_query]
    
    def _deduplicate_keyframes(
        self, 
        keyframes: List[KeyframeServiceReponse]
    ) -> List[KeyframeServiceReponse]:
        """Remove duplicate keyframes and fuse scores"""
        
        keyframe_map = {}
        for kf in keyframes:
            key = f"{kf.group_num}_{kf.video_num}_{kf.keyframe_num}"
            
            if key in keyframe_map:
                # Fuse scores - take maximum
                existing_kf = keyframe_map[key]
                if kf.confidence_score > existing_kf.confidence_score:
                    keyframe_map[key] = kf
            else:
                keyframe_map[key] = kf
        
        # Sort by confidence score
        unique_keyframes = list(keyframe_map.values())
        unique_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return unique_keyframes
    
    async def _enhance_with_multimodal_context(
        self, 
        moments: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Enhance moments with ASR and object detection context"""
        
        enhanced_moments = []
        for moment in moments:
            # Add ASR context
            asr_text = self.asr_aligner.get_asr_for_moment(
                moment.video_id, moment.start_time, moment.end_time
            )
            moment.asr_text = asr_text
            
            # Add object detection context
            objects = []
            for keyframe_num in moment.evidence_keyframes:
                keyframe_key = f"L{str(moment.group_num):0>2s}/L{str(moment.group_num):0>2s}_V{str(safe_convert_video_num(moment.video_num)):0>3s}/{int(keyframe_num):0>3d}.jpg"
                kf_objects = self.objects_data.get(keyframe_key, [])
                objects.extend(kf_objects)
            
            # Deduplicate objects
            moment.detected_objects = list(set(objects))
            enhanced_moments.append(moment)
        
        return enhanced_moments
    
    async def _llm_rerank(
        self, 
        query: str, 
        moments: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """LLM-based reranking for precision optimization"""
        
        if len(moments) <= 1:
            return moments
        
        # Prepare candidates information
        candidates_info = []
        for i, moment in enumerate(moments):
            info = f"""
            Candidate {i+1}:
            - Video: {moment.video_id}
            - Time: {moment.start_time:.1f}s - {moment.end_time:.1f}s
            - Confidence: {moment.confidence_score:.3f}
            - Objects: {', '.join(moment.detected_objects[:10]) if moment.detected_objects else 'None'}
            - ASR: {moment.asr_text[:200] + '...' if moment.asr_text and len(moment.asr_text) > 200 else moment.asr_text or 'None'}
            """
            candidates_info.append(info.strip())
        
        prompt = self.rerank_prompt.format(
            query=query,
            candidates_info='\n\n'.join(candidates_info)
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            
            # Parse LLM response and reorder based on scores
            reranked_moments = await self._parse_llm_reranking_response(response, moments)
            return reranked_moments
            
        except Exception as e:
            print(f"Warning: LLM reranking failed: {e}")
            return moments
    
    async def _parse_llm_reranking_response(
        self, 
        response, 
        moments: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """
        Parse LLM reranking response and reorder moments accordingly
        """
        
        try:
            response_text = str(response).strip()
            
            # Try to extract ranking information from LLM response
            # Look for patterns like "Candidate X is better than Candidate Y" or numbered rankings
            
            # Method 1: Look for explicit ranking numbers
            import re
            ranking_pattern = r'candidate\s*(\d+).*?(?:rank|score|better|best)'
            matches = re.findall(ranking_pattern, response_text.lower())
            
            if matches:
                # Extract candidate numbers and create ranking
                ranked_indices = []
                for match in matches:
                    try:
                        candidate_num = int(match) - 1  # Convert to 0-based index
                        if 0 <= candidate_num < len(moments):
                            ranked_indices.append(candidate_num)
                    except ValueError:
                        continue
                
                # Remove duplicates while preserving order
                seen = set()
                unique_ranked = []
                for idx in ranked_indices:
                    if idx not in seen:
                        unique_ranked.append(idx)
                        seen.add(idx)
                
                # Add any remaining moments that weren't explicitly ranked
                for i in range(len(moments)):
                    if i not in seen:
                        unique_ranked.append(i)
                
                # Reorder moments based on LLM ranking
                reranked_moments = [moments[i] for i in unique_ranked]
                print(f"LLM Reranking: Reordered {len(reranked_moments)} moments based on LLM analysis")
                return reranked_moments
            
            # Method 2: Look for confidence score adjustments
            confidence_pattern = r'confidence.*?(\d+\.?\d*)'
            confidence_matches = re.findall(confidence_pattern, response_text.lower())
            
            if confidence_matches and len(confidence_matches) >= len(moments):
                # Apply confidence adjustments
                adjusted_moments = []
                for i, moment in enumerate(moments):
                    if i < len(confidence_matches):
                        try:
                            new_confidence = float(confidence_matches[i])
                            # Create new moment with adjusted confidence
                            adjusted_moment = MomentCandidate(
                                video_id=moment.video_id,
                                group_num=moment.group_num,
                                video_num=moment.video_num,
                                start_time=moment.start_time,
                                end_time=moment.end_time,
                                confidence_score=new_confidence,
                                evidence_keyframes=moment.evidence_keyframes,
                                detected_objects=moment.detected_objects,
                                asr_text=moment.asr_text
                            )
                            adjusted_moments.append(adjusted_moment)
                        except ValueError:
                            adjusted_moments.append(moment)
                    else:
                        adjusted_moments.append(moment)
                
                # Sort by adjusted confidence
                adjusted_moments.sort(key=lambda x: x.confidence_score, reverse=True)
                print(f"LLM Reranking: Adjusted confidence scores for {len(adjusted_moments)} moments")
                return adjusted_moments
            
            # Method 3: Fallback - use semantic similarity with response
            print("LLM Reranking: Using semantic similarity fallback")
            return self._semantic_similarity_reranking(response_text, moments)
            
        except Exception as e:
            print(f"Error parsing LLM reranking response: {e}")
            return moments
    
    def _semantic_similarity_reranking(
        self, 
        response_text: str, 
        moments: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """
        Fallback reranking based on semantic similarity with LLM response
        """
        
        try:
            # Extract key terms from LLM response
            response_terms = set(response_text.lower().split())
            
            # Calculate similarity scores for each moment
            moment_scores = []
            for moment in moments:
                # Combine moment information into text
                moment_text = f"{moment.video_id} {' '.join(moment.detected_objects or [])} {moment.asr_text or ''}"
                moment_terms = set(moment_text.lower().split())
                
                # Calculate Jaccard similarity
                intersection = len(response_terms.intersection(moment_terms))
                union = len(response_terms.union(moment_terms))
                similarity = intersection / union if union > 0 else 0
                
                moment_scores.append((similarity, moment))
            
            # Sort by similarity score
            moment_scores.sort(key=lambda x: x[0], reverse=True)
            reranked_moments = [moment for _, moment in moment_scores]
            
            return reranked_moments
            
        except Exception as e:
            print(f"Error in semantic similarity reranking: {e}")
            return moments


class MultiModalAgent:
    """
    Multi-modal agent that combines all improvements for optimal competition performance
    """
    
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
        self.llm = llm
        self.data_folder = data_folder
        
        # Initialize core modules
        self.temporal_localizer = TemporalLocalizer(video_metadata_path)
        self.asr_aligner = ASRTemporalAligner(asr_data)
        
        # Initialize advanced modules
        self.query_understanding = QueryUnderstandingModule(llm)
        self.multimodal_retriever = MultiModalRetriever(
            llm, keyframe_service, model_service, self.temporal_localizer,
            self.asr_aligner, objects_data
        )
        
        # Feedback memory for interactive sessions
        self.interaction_memory: Dict[str, List[Dict[str, Any]]] = {}
    
    async def advanced_vcmr(
        self,
        query: str,
        top_k: int = 100,
        corpus_wide: bool = True
    ) -> List[MomentCandidate]:
        """
        Advanced VCMR with query understanding and multi-modal fusion
        """
        
        # Stage 1: Deep query understanding
        query_analysis = await self.query_understanding.analyze_query(query)
        
        # Stage 2: Multi-modal retrieval and ranking
        moments = await self.multimodal_retriever.retrieve_and_rank(
            query=query,
            top_k=top_k,
            enable_reranking=True
        )
        
        # Stage 3: Post-processing for competition compliance
        # Ensure temporal boundaries are valid and properly formatted
        validated_moments = []
        for moment in moments:
            if moment.end_time > moment.start_time and moment.confidence_score > 0:
                validated_moments.append(moment)
        
        return validated_moments
    
    async def advanced_video_qa(
        self,
        video_id: str,
        question: str,
        clip_range: Optional[Tuple[float, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Advanced Video QA with visual reasoning and evidence tracking
        
        Returns:
            (answer, evidence_list, confidence)
        """
        
        # Parse video identifier
        video_parts = video_id.split('/')
        if len(video_parts) >= 2:
            group_num = int(video_parts[0][1:]) if video_parts[0].startswith('L') else int(video_parts[0])
            video_num = safe_convert_video_num(video_parts[1][1:]) if video_parts[1].startswith('V') else safe_convert_video_num(video_parts[1])
        else:
            raise ValueError(f"Invalid video_id format: {video_id}")
        
        # Get relevant keyframes for the question
        embedding = self.multimodal_retriever.model_service.embedding(question).tolist()[0]
        
        if clip_range:
            # Convert time to frame range for targeted search
            start_frame = int(clip_range[0] * 25)  # Assuming 25 FPS
            end_frame = int(clip_range[1] * 25)
            range_queries = [(start_frame, end_frame)]
            
            keyframes = await self.multimodal_retriever.keyframe_service.search_by_text_range(
                text_embedding=embedding,
                top_k=30,
                score_threshold=0.1,
                range_queries=range_queries
            )
        else:
            # Search full video
            all_keyframes = await self.multimodal_retriever.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=50,
                score_threshold=0.1
            )
            # Filter to target video
            keyframes = [
                kf for kf in all_keyframes 
                if kf.group_num == group_num and kf.video_num == video_num
            ]
        
        # Create moments for context
        moments = self.temporal_localizer.create_moments_from_keyframes(keyframes[:20])
        enhanced_moments = await self.multimodal_retriever._enhance_with_multimodal_context(moments)
        
        # Generate answer with visual evidence
        answer, evidence, confidence = await self._generate_qa_answer(
            question, video_id, enhanced_moments, context
        )
        
        return answer, evidence, confidence
    
    async def _generate_qa_answer(
        self,
        question: str,
        video_id: str,
        moments: List[MomentCandidate],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Generate QA answer with evidence tracking"""
        
        # Prepare visual context with top moments
        chat_messages = []
        evidence_list = []
        
        for i, moment in enumerate(moments[:5]):  # Top 5 moments as evidence
            # Load representative keyframe images
            for keyframe_num in moment.evidence_keyframes[:2]:  # Max 2 keyframes per moment
                image_path = os.path.join(
                    self.data_folder,
                    f"L{str(moment.group_num):0>2s}/L{str(moment.group_num):0>2s}_V{str(safe_convert_video_num(moment.video_num)):0>3s}/{int(keyframe_num):0>3d}.jpg"
                )
                
                if os.path.exists(image_path):
                    context_text = f"""
                    Moment {i+1} ({moment.start_time:.1f}s - {moment.end_time:.1f}s):
                    - Objects: {', '.join(moment.detected_objects[:5]) if moment.detected_objects else 'None'}
                    - ASR: {moment.asr_text[:100] + '...' if moment.asr_text and len(moment.asr_text) > 100 else moment.asr_text or 'None'}
                    """
                    
                    message_content = [
                        ImageBlock(path=Path(image_path)),
                        TextBlock(text=context_text)
                    ]
                    
                    chat_messages.append(ChatMessage(
                        role=MessageRole.USER,
                        content=message_content
                    ))
                    
                    # Track evidence
                    evidence_list.append({
                        "start_time": moment.start_time,
                        "end_time": moment.end_time,
                        "confidence": moment.confidence_score,
                        "keyframe_num": keyframe_num,
                        "visual_path": image_path
                    })
        
        # Prepare context information
        context_info = []
        if context:
            if context.get("asr"):
                context_info.append(f"Additional ASR: {context['asr']}")
            if context.get("ocr"):
                context_info.append(f"OCR Text: {', '.join(context['ocr'])}")
            if context.get("metadata"):
                context_info.append(f"Metadata: {context['metadata']}")
        
        # Generate answer using enhanced prompts
        clip_range = "Full video"
        qa_prompt = CompetitionPrompts.VIDEO_QA_ANSWER
        
        final_prompt = qa_prompt.format(
            question=question,
            video_id=video_id,
            clip_range=clip_range,
            keyframes_info="See visual evidence above",
            context_info="\n".join(context_info) if context_info else "No additional context"
        )
        
        query_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)
        
        # Get LLM response
        response = await self.llm.achat(chat_messages)
        answer = str(response.message.content)
        
        # Calculate overall confidence
        if evidence_list:
            avg_confidence = sum(e["confidence"] for e in evidence_list) / len(evidence_list)
        else:
            avg_confidence = 0.5
        
        return answer, evidence_list, avg_confidence
    
    def _deduplicate_keyframes(
        self, 
        keyframes: List[KeyframeServiceReponse]
    ) -> List[KeyframeServiceReponse]:
        """Deduplicate keyframes with score fusion"""
        
        seen = {}
        for kf in keyframes:
            key = f"{kf.group_num}_{kf.video_num}_{kf.keyframe_num}"
            
            if key in seen:
                # Keep higher scoring keyframe
                if kf.confidence_score > seen[key].confidence_score:
                    seen[key] = kf
            else:
                seen[key] = kf
        
        return list(seen.values())
    
    async def handle_interactive_feedback(
        self,
        session_id: str,
        query: str,
        feedback: Dict[str, Any],
        previous_results: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Feedback integration for interactive tracks"""
        
        # Store feedback in memory
        if session_id not in self.interaction_memory:
            self.interaction_memory[session_id] = []
        
        self.interaction_memory[session_id].append({
            "query": query,
            "feedback": feedback,
            "previous_results": previous_results
        })
        
        # Generate refined query based on feedback
        feedback_prompt = CompetitionPrompts.FEEDBACK_INTEGRATION
        
        previous_result_summary = ""
        if previous_results:
            top_result = previous_results[0]
            previous_result_summary = f"Video: {top_result.video_id}, Time: {top_result.start_time:.1f}s-{top_result.end_time:.1f}s"
        
        prompt = feedback_prompt.format(
            original_query=query,
            previous_result=previous_result_summary,
            feedback=str(feedback)
        )
        
        try:
            response = await self.llm.acomplete(prompt)
            refined_query = str(response).strip()
            
            # Search with refined query
            return await self.multimodal_retriever.retrieve_and_rank(
                query=refined_query,
                top_k=10,
                enable_reranking=True
            )
            
        except Exception as e:
            print(f"Warning: Feedback integration failed: {e}")
            return previous_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring and optimization"""
        return {
            "total_sessions": len(self.interaction_memory),
            "avg_feedback_rounds": np.mean([len(session) for session in self.interaction_memory.values()]) if self.interaction_memory else 0,
            "system_status": "operational"
        }
