"""
LLM-based Reranking Module
Reusable component for intelligent reranking of search results
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from schema.response import KeyframeServiceReponse


class LLMReranker:
    """LLM-based reranking for improved search result quality"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.reranking_prompt = PromptTemplate(
            """
            You are an expert video search result evaluator. Given a query and a set of keyframe descriptions, 
            rank them by relevance to the query.
            
            Query: {query}
            
            Keyframe candidates:
            {keyframe_descriptions}
            
            For each keyframe, provide:
            1. Relevance score (0.0-1.0) - how well it matches the query
            2. Brief reasoning for the score
            3. Confidence in your assessment (0.0-1.0)
            
            Return a JSON array with objects containing:
            - "keyframe_id": The keyframe identifier
            - "relevance_score": Float between 0.0 and 1.0
            - "reasoning": Brief explanation
            - "confidence": Float between 0.0 and 1.0
            
            Focus on:
            - Visual content relevance
            - Action/activity matching
            - Contextual appropriateness
            - Temporal coherence
            """
        )
    
    async def rerank_keyframes(
        self, 
        query: str, 
        keyframes: List[KeyframeServiceReponse],
        max_results: int = 20
    ) -> List[Tuple[KeyframeServiceReponse, float]]:
        """Rerank keyframes using LLM-based relevance scoring"""
        
        if not keyframes:
            return []
        
        try:
            # Prepare keyframe descriptions for LLM
            keyframe_descriptions = self._prepare_keyframe_descriptions(keyframes)
            
            # Get LLM reranking
            prompt = self.reranking_prompt.format(
                query=query,
                keyframe_descriptions=keyframe_descriptions
            )
            
            response = await self.llm.acomplete(prompt)
            reranking_scores = self._parse_reranking_response(response, keyframes)
            
            # Combine original scores with LLM scores
            combined_results = []
            for kf, llm_score in reranking_scores:
                # Weighted combination: 70% original score, 30% LLM score
                combined_score = 0.7 * kf.confidence_score + 0.3 * llm_score
                combined_results.append((kf, combined_score))
            
            # Sort by combined score and limit results
            combined_results.sort(key=lambda x: x[1], reverse=True)
            return combined_results[:max_results]
            
        except Exception as e:
            print(f"Warning: LLM reranking failed: {e}")
            # Fallback to original ordering
            return [(kf, kf.confidence_score) for kf in keyframes[:max_results]]
    
    def _prepare_keyframe_descriptions(self, keyframes: List[KeyframeServiceReponse]) -> str:
        """Prepare keyframe descriptions for LLM processing"""
        
        descriptions = []
        for i, kf in enumerate(keyframes[:50]):  # Limit to 50 for LLM processing
            desc = f"{i+1}. Keyframe {kf.key} (Video L{kf.group_num:02d}_V{kf.video_num:03d}, Frame {kf.keyframe_num})"
            
            # Add contextual information if available
            if hasattr(kf, 'objects') and kf.objects:
                desc += f" - Objects: {', '.join(kf.objects[:5])}"
            
            if hasattr(kf, 'asr_text') and kf.asr_text:
                desc += f" - Audio: {kf.asr_text[:100]}..."
            
            desc += f" - Original Score: {kf.confidence_score:.3f}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _parse_reranking_response(self, response: Any, keyframes: List[KeyframeServiceReponse]) -> List[Tuple[KeyframeServiceReponse, float]]:
        """Parse LLM reranking response"""
        
        try:
            import json
            response_text = str(response).strip()
            
            # Try to extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                reranking_data = json.loads(json_str)
                
                # Map LLM scores back to keyframes
                results = []
                for item in reranking_data:
                    keyframe_id = item.get('keyframe_id')
                    relevance_score = float(item.get('relevance_score', 0.5))
                    
                    # Find corresponding keyframe
                    for kf in keyframes:
                        if str(kf.key) == str(keyframe_id):
                            results.append((kf, relevance_score))
                            break
                
                return results
            
        except Exception as e:
            print(f"Warning: Failed to parse LLM reranking response: {e}")
        
        # Fallback: return original keyframes with default scores
        return [(kf, kf.confidence_score) for kf in keyframes]


class DiversityReranker:
    """Reranker that promotes diversity in search results"""
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def rerank_for_diversity(
        self, 
        keyframes: List[KeyframeServiceReponse],
        max_results: int = 20
    ) -> List[KeyframeServiceReponse]:
        """Rerank keyframes to promote diversity across videos and temporal segments"""
        
        if len(keyframes) <= max_results:
            return keyframes
        
        selected = []
        remaining = keyframes.copy()
        
        # Select highest scoring keyframe first
        if remaining:
            selected.append(remaining.pop(0))
        
        # Iteratively select diverse keyframes
        while len(selected) < max_results and remaining:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Calculate diversity penalty
                diversity_penalty = self._calculate_diversity_penalty(candidate, selected)
                
                # Final score = original score - diversity penalty
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
    
    def _calculate_diversity_penalty(self, candidate: KeyframeServiceReponse, selected: List[KeyframeServiceReponse]) -> float:
        """Calculate diversity penalty based on similarity to already selected keyframes"""
        
        penalty = 0.0
        
        for selected_kf in selected:
            # Penalize if from same video and similar temporal position
            if (candidate.video_num == selected_kf.video_num and 
                candidate.group_num == selected_kf.group_num):
                
                # Calculate temporal distance penalty
                temporal_diff = abs(int(candidate.keyframe_num) - int(selected_kf.keyframe_num))
                if temporal_diff < 50:  # Within 50 frames
                    penalty += 0.3
                elif temporal_diff < 100:  # Within 100 frames
                    penalty += 0.1
            
            # Penalize if from same video (temporal diversity)
            elif (candidate.video_num == selected_kf.video_num and 
                  candidate.group_num == selected_kf.group_num):
                penalty += 0.05
        
        return penalty * self.diversity_weight
