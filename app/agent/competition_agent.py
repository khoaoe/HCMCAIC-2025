"""
 Competition Agent for HCMC AI Challenge 2025
Implements advanced multimodal fusion, sophisticated temporal localization,
and optimized performance for all competition tasks
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import time
import numpy as np
import json
import re
import asyncio
from pathlib import Path
from collections import defaultdict
import os

from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole

from .prompts import CompetitionPrompts
from .temporal_localization import TemporalLocalizer, ASRTemporalAligner
from .performance_optimizer import PerformanceOptimizer, CompetitionModeOptimizer
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from schema.response import KeyframeServiceReponse
from schema.competition import (
    MomentCandidate, VCMRAutomaticRequest, VCMRAutomaticResponse, VCMRCandidate,
    VideoQARequest, VideoQAResponse, VideoQAEvidence,
    KISVisualRequest, KISTextualRequest, KISProgressiveRequest, KISResponse,
    VCMRFeedback, VCMRInteractiveCandidate
)

from utils.video_utils import safe_convert_video_num

class AdvancedQueryProcessor:
    """ query processing with semantic understanding and expansion"""
    
    def __init__(self, llm: LLM, model_service: ModelService):
        self.llm = llm
        self.model_service = model_service
        
        self.query_analysis_prompt = PromptTemplate(
            """
            Analyze this video search query and extract structured information for optimal retrieval.
            
            Query: {query}
            
            Extract:
            1. **Primary Visual Elements**: Main objects, people, scenes, settings
            2. **Actions & Interactions**: Specific movements, activities, behaviors
            3. **Temporal Indicators**: Sequence words, time references, duration cues
            4. **Spatial Relationships**: Locations, positions, directions
            5. **Descriptive Attributes**: Colors, sizes, styles, emotions
            6. **Context Clues**: Background information, situational context
            
            Return JSON with:
            - "primary_focus": Main search target (2-3 key terms)
            - "visual_elements": List of important visual components
            - "actions": List of actions/verbs
            - "temporal_cues": Time-related information
            - "context": Setting/environmental details
            - "complexity": "simple"/"moderate"/"complex"
            - "search_variations": 3-4 alternative query formulations
            """
        )
    
    async def analyze_and_expand_query(self, query: str) -> Dict[str, Any]:
        """Advanced query analysis with semantic expansion"""
        
        try:
            prompt = self.query_analysis_prompt.format(query=query)
            response = await self.llm.acomplete(prompt)
            
            # Parse LLM response - attempt JSON parsing with fallback
            response_text = str(response).strip()
            
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing if JSON fails
                analysis = self._parse_analysis_fallback(response_text, query)
            
            # Generate semantic variations using embeddings
            semantic_variations = await self._generate_semantic_variations(query, analysis)
            analysis["semantic_variations"] = semantic_variations
            
            return analysis
            
        except Exception as e:
            print(f"Warning: Query analysis failed: {e}")
            return {
                "primary_focus": query,
                "visual_elements": [],
                "actions": [],
                "temporal_cues": [],
                "context": "",
                "complexity": "moderate",
                "search_variations": [query],
                "semantic_variations": [query]
            }
    
    def _parse_analysis_fallback(self, response_text: str, original_query: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        
        # Simple keyword extraction as fallback
        words = original_query.lower().split()
        
        # Common action verbs
        action_words = {'walk', 'run', 'sit', 'stand', 'talk', 'drive', 'cook', 'eat', 'drink', 'play', 'write', 'read'}
        actions = [word for word in words if word in action_words]
        
        # Temporal indicators
        temporal_words = {'before', 'after', 'during', 'while', 'then', 'next', 'first', 'last', 'when'}
        temporal_cues = [word for word in words if word in temporal_words]
        
        return {
            "primary_focus": original_query,
            "visual_elements": words[:3],  # First 3 words as visual elements
            "actions": actions,
            "temporal_cues": temporal_cues,
            "context": original_query,
            "complexity": "moderate",
            "search_variations": [original_query]
        }
    
    async def _generate_semantic_variations(self, original_query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate semantic variations using different focus areas"""
        
        variations = [original_query]  # Always include original
        
        try:
            # Visual-focused variation
            if analysis.get("visual_elements"):
                visual_terms = " ".join(analysis["visual_elements"][:3])
                variations.append(f"{visual_terms} scene visual content")
            
            # Action-focused variation
            if analysis.get("actions"):
                action_terms = " ".join(analysis["actions"])
                variations.append(f"{action_terms} activity movement")
            
            # Context-focused variation
            if analysis.get("context") and analysis["context"] != original_query:
                variations.append(f"{analysis['context']} environment setting")
            
            # Add variations from LLM if available
            if analysis.get("search_variations"):
                variations.extend(analysis["search_variations"][:3])
            
            # Remove duplicates and limit
            unique_variations = []
            for var in variations:
                if var not in unique_variations and len(var.strip()) > 5:
                    unique_variations.append(var)
            
            return unique_variations[:5]  # Limit to 5 variations
            
        except Exception as e:
            print(f"Warning: Semantic variation generation failed: {e}")
            return [original_query]


class AdvancedVisualMatcher:
    """ visual similarity matching for KIS-V tasks"""
    
    def __init__(self, model_service: ModelService, data_folder: str):
        self.model_service = model_service
        self.data_folder = data_folder
    
    async def find_visual_matches(
        self, 
        query_clip_path: str, 
        keyframes: List[KeyframeServiceReponse],
        top_k: int = 50
    ) -> List[Tuple[KeyframeServiceReponse, float]]:
        """Find visual matches using advanced similarity metrics"""
        
        try:
            # For now, implement as  textual matching
            # In production, this would use visual feature extraction from query_clip_path
            
            # Extract representative frames from query clip (placeholder)
            # This would involve actual video processing in production
            query_description = await self._extract_visual_description(query_clip_path)
            
            # Use  embedding comparison
            query_embedding = self.model_service.embedding(query_description).tolist()[0]
            
            similarities = []
            for kf in keyframes:
                # Compute embedding similarity
                kf_description = f"keyframe {kf.keyframe_num} visual content"
                kf_embedding = self.model_service.embedding(kf_description).tolist()[0]
                
                # Cosine similarity
                similarity = np.dot(query_embedding, kf_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(kf_embedding)
                )
                
                similarities.append((kf, float(similarity)))
            
            # Sort by similarity and return top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Warning: Visual matching failed: {e}")
            # Fallback to confidence scores
            scored_kfs = [(kf, kf.confidence_score) for kf in keyframes]
            scored_kfs.sort(key=lambda x: x[1], reverse=True)
            return scored_kfs[:top_k]
    
    async def _extract_visual_description(self, clip_path: str) -> str:
        """Extract visual description from query clip (placeholder implementation)"""
        
        # In production, this would:
        # 1. Extract keyframes from the query clip
        # 2. Use vision models to generate descriptions
        # 3. Combine descriptions into a comprehensive query
        
        # For now, return a generic description
        return f"visual content from query clip {clip_path}"


class LLMReranker:
    """Sophisticated LLM-based reranking with actual parsing and reordering"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        
        self.reranking_prompt = PromptTemplate(
            """
            Rerank these video moment candidates based on relevance to the query.
            
            Query: {query}
            
            Candidates:
            {candidates_info}
            
            For each candidate, provide:
            1. Relevance score (0.0-1.0)
            2. Brief rationale (1-2 sentences)
            
            Consider:
            - Visual content alignment with query
            - Action/event matching
            - Temporal context appropriateness
            - ASR text relevance
            - Object detection support
            - Overall coherence and completeness
            
            Scoring Guidelines:
            - 0.9-1.0: Perfect match with all query elements
            - 0.7-0.8: Strong match with most query elements
            - 0.5-0.6: Moderate match with some query elements
            - 0.3-0.4: Weak match with few query elements
            - 0.0-0.2: Poor match or irrelevant content
            
            Return JSON array with format:
            [
                {{
                    "candidate_id": 1,
                    "relevance_score": 0.85,
                    "rationale": "Strong visual match with query elements"
                }},
                ...
            ]
            
            Sort by relevance_score (highest first).
            """
        )
    
    async def rerank_candidates(
        self, 
        query: str, 
        candidates: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Rerank candidates using LLM with actual parsing"""
        
        if len(candidates) <= 1:
            return candidates
        
        # Prepare candidate information with enhanced context
        candidates_info = []
        for i, candidate in enumerate(candidates):
            # Enhanced context information
            objects_info = ', '.join(candidate.detected_objects[:8]) if candidate.detected_objects else 'None detected'
            asr_info = candidate.asr_text[:200] + '...' if candidate.asr_text and len(candidate.asr_text) > 200 else candidate.asr_text or 'No ASR'
            
            info = f"""
            Candidate {i+1}:
            - Video: {candidate.video_id}
            - Time: {candidate.start_time:.1f}s - {candidate.end_time:.1f}s
            - Original Score: {candidate.confidence_score:.3f}
            - Detected Objects: {objects_info}
            - ASR Text: {asr_info}
            - Duration: {candidate.end_time - candidate.start_time:.1f}s
            """
            candidates_info.append(info.strip())
        
        try:
            prompt = self.reranking_prompt.format(
                query=query,
                candidates_info='\n\n'.join(candidates_info)
            )
            
            response = await self.llm.acomplete(prompt)
            response_text = str(response).strip()
            
            # Parse LLM response
            reranked_candidates = self._parse_reranking_response(response_text, candidates)
            
            return reranked_candidates
            
        except Exception as e:
            print(f"Warning: LLM reranking failed: {e}")
            return candidates
    
    def _parse_reranking_response(
        self, 
        response_text: str, 
        original_candidates: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Parse LLM reranking response and reorder candidates"""
        
        try:
            # Try to parse JSON response
            if '[' in response_text and ']' in response_text:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                json_text = response_text[start_idx:end_idx]
                
                rankings = json.loads(json_text)
                
                # Create reordered list based on rankings
                reordered = []
                for ranking in rankings:
                    candidate_id = ranking.get('candidate_id', 1)
                    relevance_score = ranking.get('relevance_score', 0.5)
                    
                    # Get corresponding candidate (1-indexed to 0-indexed)
                    if 1 <= candidate_id <= len(original_candidates):
                        candidate = original_candidates[candidate_id - 1]
                        # Update confidence score with LLM relevance
                        candidate.confidence_score = relevance_score
                        reordered.append(candidate)
                
                # If we successfully reordered all candidates, return reordered list
                if len(reordered) == len(original_candidates):
                    return reordered
            
            # Fallback: try to extract scores with regex
            return self._parse_scores_fallback(response_text, original_candidates)
            
        except Exception as e:
            print(f"Warning: Failed to parse reranking response: {e}")
            return original_candidates
    
    def _parse_scores_fallback(
        self, 
        response_text: str, 
        original_candidates: List[MomentCandidate]
    ) -> List[MomentCandidate]:
        """Fallback parsing using regex to extract scores"""
        
        try:
            # Look for patterns like "Candidate 1: 0.85" or "1. Score: 0.75"
            score_patterns = [
                r'[Cc]andidate\s*(\d+).*?(\d+\.?\d*)',
                r'(\d+)\..*?[Ss]core.*?(\d+\.?\d*)',
                r'(\d+).*?(\d+\.?\d*)'
            ]
            
            scores = {}
            for pattern in score_patterns:
                matches = re.findall(pattern, response_text)
                for match in matches:
                    try:
                        candidate_id = int(match[0])
                        score = float(match[1])
                        if 0 <= score <= 1 and 1 <= candidate_id <= len(original_candidates):
                            scores[candidate_id] = score
                    except (ValueError, IndexError):
                        continue
                
                if scores:
                    break
            
            if scores:
                # Update scores and sort
                for candidate_id, score in scores.items():
                    if 1 <= candidate_id <= len(original_candidates):
                        original_candidates[candidate_id - 1].confidence_score = score
                
                # Sort by updated scores
                original_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return original_candidates
            
        except Exception as e:
            print(f"Warning: Fallback score parsing failed: {e}")
            return original_candidates


class AdvancedTemporalRefiner:
    """ temporal boundary refinement"""
    
    def __init__(self, temporal_localizer: TemporalLocalizer, asr_aligner: ASRTemporalAligner):
        self.temporal_localizer = temporal_localizer
        self.asr_aligner = asr_aligner
    
    def refine_temporal_boundaries(
        self, 
        moment: MomentCandidate,
        query: str,
        refinement_strategy: str = "adaptive"
    ) -> MomentCandidate:
        """Refine temporal boundaries using multiple strategies"""
        
        if refinement_strategy == "adaptive":
            return self._adaptive_boundary_refinement(moment, query)
        elif refinement_strategy == "asr_aligned":
            return self._asr_aligned_refinement(moment)
        elif refinement_strategy == "action_focused":
            return self._action_focused_refinement(moment, query)
        else:
            return moment
    
    def _adaptive_boundary_refinement(self, moment: MomentCandidate, query: str) -> MomentCandidate:
        """Adaptive refinement based on content analysis"""
        
        try:
            # Analyze query for temporal hints
            duration_keywords = {
                'brief': 2.0, 'short': 3.0, 'quick': 2.0,
                'long': 10.0, 'extended': 15.0, 'detailed': 8.0
            }
            
            query_lower = query.lower()
            suggested_duration = None
            
            for keyword, duration in duration_keywords.items():
                if keyword in query_lower:
                    suggested_duration = duration
                    break
            
            if suggested_duration:
                # Adjust moment duration
                current_duration = moment.end_time - moment.start_time
                if abs(current_duration - suggested_duration) > 1.0:
                    center_time = (moment.start_time + moment.end_time) / 2
                    moment.start_time = max(0, center_time - suggested_duration / 2)
                    moment.end_time = center_time + suggested_duration / 2
            
            return moment
            
        except Exception as e:
            print(f"Warning: Adaptive refinement failed: {e}")
            return moment
    
    def _asr_aligned_refinement(self, moment: MomentCandidate) -> MomentCandidate:
        """Refine boundaries to align with ASR segments"""
        
        try:
            if not moment.asr_text:
                return moment
            
            # This would align with actual ASR timestamps in production
            # For now, return the moment as-is
            return moment
            
        except Exception as e:
            print(f"Warning: ASR alignment failed: {e}")
            return moment
    
    def _action_focused_refinement(self, moment: MomentCandidate, query: str) -> MomentCandidate:
        """Refine to focus on action sequences"""
        
        try:
            # Action-based duration mapping
            action_durations = {
                'walk': 5.0, 'run': 3.0, 'sit': 2.0, 'stand': 1.0,
                'drive': 8.0, 'cook': 10.0, 'eat': 6.0, 'talk': 4.0
            }
            
            query_lower = query.lower()
            for action, duration in action_durations.items():
                if action in query_lower:
                    current_duration = moment.end_time - moment.start_time
                    if current_duration > duration * 1.5:  # Too long
                        moment.end_time = moment.start_time + duration
                    break
            
            return moment
            
        except Exception as e:
            print(f"Warning: Action-focused refinement failed: {e}")
            return moment


class CompetitionAgent:
    """
     agent with advanced multimodal fusion and sophisticated processing
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
        self.performance_optimizer = PerformanceOptimizer()
        
        # Initialize  modules
        self.query_processor = AdvancedQueryProcessor(llm, model_service)
        self.visual_matcher = AdvancedVisualMatcher(model_service, data_folder)
        self.llm_reranker = LLMReranker(llm)
        self.temporal_refiner = AdvancedTemporalRefiner(self.temporal_localizer, self.asr_aligner)
        
        # Services
        self.keyframe_service = keyframe_service
        self.model_service = model_service
        self.objects_data = objects_data
        
        # Interactive state management
        self.session_state: Dict[str, Dict[str, Any]] = {}
    
    async def vcmr_automatic(
        self, 
        request: VCMRAutomaticRequest
    ) -> VCMRAutomaticResponse:
        """ VCMR automatic with advanced processing"""
        
        start_time = time.time()
        
        try:
            # Stage 1: Advanced query analysis and expansion
            query_analysis = await self.query_processor.analyze_and_expand_query(request.query)
            
            # Stage 2: Adaptive search parameters
            mode_settings = CompetitionModeOptimizer.get_automatic_mode_settings()
            perf_settings = self.performance_optimizer.optimize_top_k_strategy(
                query_analysis.get("complexity", "moderate"),
                interactive_mode=False
            )
            
            # Stage 3: Multi-query retrieval with semantic variations
            all_keyframes = []
            search_queries = query_analysis.get("semantic_variations", [request.query])
            
            for search_query in search_queries:
                # Check embedding cache first
                cached_embedding = self.performance_optimizer.get_cached_embedding(search_query)
                if cached_embedding:
                    embedding = cached_embedding
                    self.performance_optimizer.performance_stats["cache_hits"] += 1
                else:
                    embedding = self.model_service.embedding(search_query).tolist()[0]
                    self.performance_optimizer.cache_embedding(search_query, embedding)
                
                keyframes = await self.keyframe_service.search_by_text(
                    text_embedding=embedding,
                    top_k=perf_settings["initial_top_k"],
                    score_threshold=perf_settings["score_threshold"]
                )
                
                all_keyframes.extend(keyframes)
            
            # Stage 4: Deduplication and smart clustering
            unique_keyframes = self._deduplicate_keyframes(all_keyframes)
            moment_clusters = self.performance_optimizer.smart_temporal_clustering(
                unique_keyframes, target_moments=request.top_k * 2
            )
            
            # Stage 5: Create  moments with multimodal context
            moments = []
            for cluster in moment_clusters:
                moment = self.temporal_localizer.create_moment_from_keyframes(cluster)
                
                # Add ASR context
                asr_text = self.asr_aligner.get_asr_for_moment(
                    moment.video_id, moment.start_time, moment.end_time
                )
                moment.asr_text = asr_text
                
                # Add object detection context
                objects = []
                for kf_num in moment.evidence_keyframes:
                    keyframe_key = f"L{moment.group_num:02d}/L{moment.group_num:02d}_V{safe_convert_video_num(moment.video_num):03d}/{kf_num:03d}.jpg"
                    kf_objects = self.objects_data.get(keyframe_key, [])
                    objects.extend(kf_objects)
                moment.detected_objects = list(set(objects))
                
                # Apply temporal boundary refinement
                refined_moment = self.temporal_refiner.refine_temporal_boundaries(
                    moment, request.query, "adaptive"
                )
                
                moments.append(refined_moment)
            
            # Stage 6: Advanced LLM reranking
            if len(moments) > 1:
                reranked_moments = await self.llm_reranker.rerank_candidates(
                    request.query, moments[:perf_settings["rerank_top_k"]]
                )
                # Combine reranked top results with remaining moments
                final_moments = reranked_moments + moments[len(reranked_moments):]
            else:
                final_moments = moments
            
            # Stage 7: Convert to competition format
            candidates = []
            for moment in final_moments[:request.top_k]:
                candidates.append(VCMRCandidate(
                    video_id=moment.video_id,
                    start_time=moment.start_time,
                    end_time=moment.end_time,
                    score=moment.confidence_score
                ))
            
            # Generate  explanation
            notes = self._generate_result_explanation(request.query, final_moments[:3])
            
            # Track performance
            self.performance_optimizer.track_performance(
                start_time, request.query, len(candidates)
            )
            
            return VCMRAutomaticResponse(
                query=request.query,
                candidates=candidates,
                notes=notes
            )
            
        except Exception as e:
            print(f"Error in  VCMR: {e}")
            # Fallback to basic search
            return await self._fallback_vcmr(request)
    
    async def video_qa(
        self, 
        request: VideoQARequest
    ) -> VideoQAResponse:
        """ Video QA with comprehensive evidence tracking"""
        
        try:
            # Parse video identifier
            video_parts = request.video_id.split('/')
            if len(video_parts) >= 2:
                group_num = int(video_parts[0][1:]) if video_parts[0].startswith('L') else int(video_parts[0])
                video_num = safe_convert_video_num(video_parts[1][1:]) if video_parts[1].startswith('V') else safe_convert_video_num(video_parts[1])
            else:
                raise ValueError(f"Invalid video_id format: {request.video_id}")
            
            #  question analysis
            question_analysis = await self.query_processor.analyze_and_expand_query(request.question)
            
            # Adaptive search based on question type
            search_params = self._get_qa_search_parameters(request.question, question_analysis)
            
            # Multi-modal evidence gathering
            evidence_keyframes = await self._gather_qa_evidence(
                request, group_num, video_num, search_params
            )
            
            # Create comprehensive visual context
            visual_context = await self._create_visual_context(
                evidence_keyframes, request.question
            )
            
            # Generate answer with  prompting
            answer, confidence = await self._generate_qa_answer(
                request, visual_context, question_analysis
            )
            
            # Create detailed evidence list
            evidence = self._create_evidence_list(evidence_keyframes, confidence)
            
            return VideoQAResponse(
                video_id=request.video_id,
                question=request.question,
                answer=answer,
                evidence=evidence,
                confidence=confidence
            )
            
        except Exception as e:
            print(f"Error in  Video QA: {e}")
            return VideoQAResponse(
                video_id=request.video_id,
                question=request.question,
                answer="Unable to analyze video content",
                evidence=[],
                confidence=0.0
            )
    
    async def kis_textual(
        self, 
        request: KISTextualRequest
    ) -> KISResponse:
        """ KIS textual with precision optimization"""
        
        try:
            #  description analysis for exact matching
            description_analysis = await self.query_processor.analyze_and_expand_query(
                request.text_description
            )
            
            # Use precision-focused search parameters
            kis_settings = CompetitionModeOptimizer.get_kis_mode_settings()
            
            # Multiple search strategies for exact matching
            search_strategies = [
                ("exact", request.text_description),
                ("focused", description_analysis.get("primary_focus", request.text_description)),
                ("detailed", " ".join(description_analysis.get("visual_elements", [])))
            ]
            
            best_matches = []
            for strategy_name, search_query in search_strategies:
                if not search_query.strip():
                    continue
                
                embedding = self.model_service.embedding(search_query).tolist()[0]
                keyframes = await self.keyframe_service.search_by_text(
                    text_embedding=embedding,
                    top_k=kis_settings["top_k"],
                    score_threshold=kis_settings["score_threshold"]
                )
                
                if keyframes:
                    best_matches.extend(keyframes)
            
            if not best_matches:
                raise ValueError("No exact matches found for description")
            
            # Find the highest confidence match
            best_match = max(best_matches, key=lambda x: x.confidence_score)
            
            # Create tight temporal window for exact matching
            center_time = self.temporal_localizer.keyframe_to_timestamp(
                best_match.group_num, best_match.video_num, best_match.keyframe_num
            )
            
            # Very tight window for KIS (±0.5 seconds)
            start_time = max(0, center_time - 0.5)
            end_time = center_time + 0.5
            
            return KISResponse(
                video_id=f"L{best_match.group_num:02d}/L{best_match.group_num:02d}_V{safe_convert_video_num(best_match.video_num):03d}",
                start_time=start_time,
                end_time=end_time,
                match_confidence=best_match.confidence_score
            )
            
        except Exception as e:
            print(f"Error in  KIS textual: {e}")
            raise ValueError(f"KIS textual search failed: {e}")
    
    async def kis_visual(
        self, 
        request: KISVisualRequest
    ) -> KISResponse:
        """ KIS visual with advanced visual matching"""
        
        try:
            # First, get candidate keyframes using broad search
            broad_embedding = self.model_service.embedding("visual content").tolist()[0]
            candidate_keyframes = await self.keyframe_service.search_by_text(
                text_embedding=broad_embedding,
                top_k=200,
                score_threshold=0.1
            )
            
            # Apply advanced visual matching
            visual_matches = await self.visual_matcher.find_visual_matches(
                request.query_clip_uri, candidate_keyframes, top_k=10
            )
            
            if not visual_matches:
                raise ValueError("No visual matches found")
            
            # Get best visual match
            best_keyframe, similarity_score = visual_matches[0]
            
            # Create precise temporal window
            center_time = self.temporal_localizer.keyframe_to_timestamp(
                best_keyframe.group_num, best_keyframe.video_num, best_keyframe.keyframe_num
            )
            
            return KISResponse(
                video_id=f"L{best_keyframe.group_num:02d}/L{best_keyframe.group_num:02d}_V{safe_convert_video_num(best_keyframe.video_num):03d}",
                start_time=max(0, center_time - 0.5),
                end_time=center_time + 0.5,
                match_confidence=similarity_score
            )
            
        except Exception as e:
            print(f"Error in  KIS visual: {e}")
            raise ValueError(f"KIS visual search failed: {e}")
    
    async def kis_progressive(
        self, 
        request: KISProgressiveRequest,
        session_id: str,
        additional_hints: Optional[List[str]] = None
    ) -> KISResponse:
        """ KIS progressive with session state management"""
        
        try:
            # Initialize or update session state
            if session_id not in self.session_state:
                self.session_state[session_id] = {
                    "initial_hint": request.initial_hint,
                    "all_hints": [request.initial_hint],
                    "search_history": [],
                    "confidence_progression": []
                }
            
            session = self.session_state[session_id]
            
            # Add new hints
            if additional_hints:
                session["all_hints"].extend(additional_hints)
            
            # Combine all hints intelligently
            combined_query = await self._combine_progressive_hints(session["all_hints"])
            
            # Search with progressively refined query
            embedding = self.model_service.embedding(combined_query).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=100,
                score_threshold=0.15  # Moderate threshold for progressive search
            )
            
            if not keyframes:
                # Relax threshold if no results
                keyframes = await self.keyframe_service.search_by_text(
                    text_embedding=embedding,
                    top_k=50,
                    score_threshold=0.05
                )
            
            if not keyframes:
                raise ValueError("No matches found even with progressive hints")
            
            best_keyframe = keyframes[0]
            center_time = self.temporal_localizer.keyframe_to_timestamp(
                best_keyframe.group_num, best_keyframe.video_num, best_keyframe.keyframe_num
            )
            
            # Update session history
            search_result = {
                "query": combined_query,
                "hints_used": len(session["all_hints"]),
                "best_match": {
                    "video_id": f"L{best_keyframe.group_num:02d}/L{best_keyframe.group_num:02d}_V{safe_convert_video_num(best_keyframe.video_num):03d}",
                    "timestamp": center_time,
                    "confidence": best_keyframe.confidence_score
                }
            }
            session["search_history"].append(search_result)
            session["confidence_progression"].append(best_keyframe.confidence_score)
            
            return KISResponse(
                video_id=f"L{best_keyframe.group_num:02d}/L{best_keyframe.group_num:02d}_V{safe_convert_video_num(best_keyframe.video_num):03d}",
                start_time=max(0, center_time - 1.0),
                end_time=center_time + 1.0,
                match_confidence=best_keyframe.confidence_score
            )
            
        except Exception as e:
            print(f"Error in  KIS progressive: {e}")
            raise ValueError(f"KIS progressive search failed: {e}")
    
    # Helper methods
    
    def _deduplicate_keyframes(
        self, 
        keyframes: List[KeyframeServiceReponse]
    ) -> List[KeyframeServiceReponse]:
        """ deduplication with score fusion"""
        
        keyframe_map = {}
        for kf in keyframes:
            key = f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{safe_convert_video_num(kf.video_num):03d}/{kf.keyframe_num:03d}.jpg"
            
            if key in keyframe_map:
                # Use weighted average for score fusion
                existing_kf = keyframe_map[key]
                fused_score = (existing_kf.confidence_score + kf.confidence_score) / 2
                if kf.confidence_score > existing_kf.confidence_score:
                    keyframe_map[key] = kf
                    keyframe_map[key].confidence_score = fused_score
            else:
                keyframe_map[key] = kf
        
        unique_keyframes = list(keyframe_map.values())
        unique_keyframes.sort(key=lambda x: x.confidence_score, reverse=True)
        return unique_keyframes
    
    def _generate_result_explanation(
        self, 
        query: str, 
        top_moments: List[MomentCandidate]
    ) -> Optional[str]:
        """Generate explanation for top results"""
        
        if not top_moments:
            return None
        
        top_moment = top_moments[0]
        explanation = f"Top result in {top_moment.video_id} ({top_moment.start_time:.1f}s-{top_moment.end_time:.1f}s)"
        
        if top_moment.detected_objects:
            explanation += f" shows {', '.join(top_moment.detected_objects[:3])}"
        
        if top_moment.asr_text:
            explanation += f" with context: '{top_moment.asr_text[:100]}...'"
        
        return explanation
    
    async def _fallback_vcmr(self, request: VCMRAutomaticRequest) -> VCMRAutomaticResponse:
        """Fallback VCMR implementation"""
        
        try:
            embedding = self.model_service.embedding(request.query).tolist()[0]
            keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=min(request.top_k * 2, 200),
                score_threshold=0.1
            )
            
            moments = self.temporal_localizer.create_moments_from_keyframes(
                keyframes, max_moments=request.top_k
            )
            
            candidates = [
                VCMRCandidate(
                    video_id=moment.video_id,
                    start_time=moment.start_time,
                    end_time=moment.end_time,
                    score=moment.confidence_score
                )
                for moment in moments
            ]
            
            return VCMRAutomaticResponse(
                query=request.query,
                candidates=candidates,
                notes="Fallback search results"
            )
            
        except Exception as e:
            print(f"Error in fallback VCMR: {e}")
            return VCMRAutomaticResponse(
                query=request.query,
                candidates=[],
                notes="Search failed"
            )
    
    def _get_qa_search_parameters(
        self, 
        question: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimized search parameters for QA"""
        
        question_lower = question.lower()
        
        # Question type detection
        if any(word in question_lower for word in ['how many', 'count', 'number']):
            return {"top_k": 30, "score_threshold": 0.1, "focus": "counting"}
        elif any(word in question_lower for word in ['who', 'what', 'where', 'when']):
            return {"top_k": 20, "score_threshold": 0.15, "focus": "identification"}
        elif any(word in question_lower for word in ['yes', 'no', 'is', 'are', 'does', 'did']):
            return {"top_k": 15, "score_threshold": 0.2, "focus": "verification"}
        else:
            return {"top_k": 25, "score_threshold": 0.12, "focus": "general"}
    
    async def _gather_qa_evidence(
        self, 
        request: VideoQARequest,
        group_num: int,
        video_num: int,
        search_params: Dict[str, Any]
    ) -> List[KeyframeServiceReponse]:
        """Gather evidence keyframes for QA"""
        
        # Generate embedding for question
        embedding = self.model_service.embedding(request.question).tolist()[0]
        
        if request.clip:
            # Targeted search within clip
            start_frame = int(request.clip.start_time * 25)
            end_frame = int(request.clip.end_time * 25)
            range_queries = [(start_frame, end_frame)]
            
            keyframes = await self.keyframe_service.search_by_text_range(
                text_embedding=embedding,
                top_k=search_params["top_k"],
                score_threshold=search_params["score_threshold"],
                range_queries=range_queries
            )
        else:
            # Full video search with filtering
            all_keyframes = await self.keyframe_service.search_by_text(
                text_embedding=embedding,
                top_k=search_params["top_k"] * 2,
                score_threshold=search_params["score_threshold"]
            )
            
            # Filter to target video
            keyframes = [
                kf for kf in all_keyframes 
                if kf.group_num == group_num and kf.video_num == video_num
            ]
        
        return keyframes[:search_params["top_k"]]
    
    async def _create_visual_context(
        self, 
        keyframes: List[KeyframeServiceReponse],
        question: str
    ) -> List[Dict[str, Any]]:
        """Create rich visual context for QA"""
        
        visual_context = []
        
        for kf in keyframes[:10]:  # Limit for LLM context
            timestamp = self.temporal_localizer.keyframe_to_timestamp(
                kf.group_num, kf.video_num, kf.keyframe_num
            )
            
            # Get objects for this keyframe
            keyframe_key = f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{safe_convert_video_num(kf.video_num):03d}/{kf.keyframe_num:03d}.jpg"
            objects = self.objects_data.get(keyframe_key, [])
            
            # Image path
            image_path = os.path.join(
                self.data_folder,
                f"L{kf.group_num:02d}/L{kf.group_num:02d}_V{safe_convert_video_num(kf.video_num):03d}/{kf.keyframe_num:03d}.jpg"
            )
            
            visual_context.append({
                "timestamp": timestamp,
                "confidence": kf.confidence_score,
                "objects": objects,
                "image_path": image_path,
                "keyframe_num": kf.keyframe_num
            })
        
        return visual_context
    
    async def _generate_qa_answer(
        self, 
        request: VideoQARequest,
        visual_context: List[Dict[str, Any]],
        question_analysis: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Generate  QA answer with visual evidence"""
        
        # Prepare chat messages with visual content
        chat_messages = []
        
        for ctx in visual_context:
            if os.path.exists(ctx["image_path"]):
                context_text = f"""
                Frame at {ctx['timestamp']:.1f}s:
                - Objects detected: {', '.join(ctx['objects'][:5]) if ctx['objects'] else 'None'}
                - Relevance score: {ctx['confidence']:.3f}
                """
                
                message_content = [
                    ImageBlock(path=Path(ctx["image_path"])),
                    TextBlock(text=context_text)
                ]
                
                chat_messages.append(ChatMessage(
                    role=MessageRole.USER,
                    content=message_content
                ))
        
        # Prepare  QA prompt
        qa_prompt = CompetitionPrompts.VIDEO_QA_ANSWER
        
        # Context information
        context_info = []
        if request.context:
            if request.context.asr:
                context_info.append(f"ASR: {request.context.asr}")
            if request.context.ocr:
                context_info.append(f"OCR: {', '.join(request.context.ocr)}")
        
        clip_range = "Full video"
        if request.clip:
            clip_range = f"{request.clip.start_time:.1f}s - {request.clip.end_time:.1f}s"
        
        final_prompt = qa_prompt.format(
            question=request.question,
            video_id=request.video_id,
            clip_range=clip_range,
            keyframes_info="Visual evidence provided above",
            context_info="\n".join(context_info) if context_info else "No additional context"
        )
        
        query_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)
        
        # Generate answer
        response = await self.llm.achat(chat_messages)
        answer = str(response.message.content)
        
        # Calculate confidence based on visual evidence
        if visual_context:
            avg_confidence = sum(ctx["confidence"] for ctx in visual_context) / len(visual_context)
        else:
            avg_confidence = 0.5
        
        return answer, avg_confidence
    
    def _create_evidence_list(
        self, 
        keyframes: List[KeyframeServiceReponse],
        base_confidence: float
    ) -> List[VideoQAEvidence]:
        """Create detailed evidence list for QA"""
        
        evidence = []
        for kf in keyframes[:5]:  # Top 5 as evidence
            timestamp = self.temporal_localizer.keyframe_to_timestamp(
                kf.group_num, kf.video_num, kf.keyframe_num
            )
            
            evidence.append(VideoQAEvidence(
                start_time=timestamp,
                end_time=timestamp + 1.0,  # 1-second evidence window
                confidence=min(kf.confidence_score, base_confidence)
            ))
        
        return evidence
    
    async def _combine_progressive_hints(self, hints: List[str]) -> str:
        """Intelligently combine progressive hints"""
        
        if len(hints) <= 1:
            return hints[0] if hints else ""
        
        try:
            # Use LLM to combine hints intelligently
            hint_combination_prompt = PromptTemplate(
                """
                Combine these progressive hints into a comprehensive search query for exact video segment matching.
                
                Hints (in chronological order):
                {hints}
                
                Create a unified search query that:
                1. Integrates all relevant information
                2. Resolves any contradictions (prefer later hints)
                3. Maintains specificity for exact matching
                4. Focuses on unique, identifying features
                
                Return only the combined search query.
                """
            )
            
            hints_text = "\n".join([f"Hint {i+1}: {hint}" for i, hint in enumerate(hints)])
            prompt = hint_combination_prompt.format(hints=hints_text)
            
            response = await self.llm.acomplete(prompt)
            combined_query = str(response).strip()
            
            # Fallback to simple concatenation if LLM fails
            if not combined_query or len(combined_query) < 10:
                combined_query = " ".join(hints)
            
            return combined_query
            
        except Exception as e:
            print(f"Warning: Hint combination failed: {e}")
            return " ".join(hints)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        base_metrics = self.performance_optimizer.get_optimization_suggestions()
        
        metrics = {
            "session_count": len(self.session_state),
            "active_sessions": len([s for s in self.session_state.values() if s.get("search_history")]),
            "avg_hints_per_session": np.mean([len(s.get("all_hints", [])) for s in self.session_state.values()]) if self.session_state else 0
        }
        
        base_metrics.update(metrics)
        return base_metrics

    async def vcmr_automatic(
        self, 
        request: VCMRAutomaticRequest
    ) -> VCMRAutomaticResponse:
        """VCMR automatic - alias for vcmr_automatic for compatibility"""
        return await self.vcmr_automatic(request)

    async def video_qa(
        self, 
        request: VideoQARequest
    ) -> VideoQAResponse:
        """Video QA processing"""
        # Delegate to competition task processor
        return await self.task_processor.process_video_qa(request)

    async def kis_textual(
        self, 
        request: KISTextualRequest
    ) -> KISResponse:
        """KIS Textual processing"""
        # Delegate to competition task processor
        return await self.task_processor.process_kis_textual(request)

    async def kis_visual(
        self, 
        request: KISVisualRequest
    ) -> KISResponse:
        """KIS Visual processing"""
        # Delegate to competition task processor
        return await self.task_processor.process_kis_visual(request)

    async def kis_progressive(
        self, 
        request: KISProgressiveRequest
    ) -> KISResponse:
        """KIS Progressive processing"""
        # Delegate to competition task processor
        return await self.task_processor.process_kis_progressive(request)


# Export the  agent as the main competition agent
CompetitionAgent = CompetitionAgent
