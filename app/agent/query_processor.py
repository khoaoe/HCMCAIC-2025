"""
 Query Processing Module
Reusable component for query analysis and expansion across different agents
"""

import json
import re
from typing import List, Dict, Any
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from service.model_service import ModelService


class QueryProcessor:
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
        """ query analysis with semantic expansion"""
        
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
        
        # Simple regex-based extraction
        analysis = {
            "primary_focus": original_query,
            "visual_elements": [],
            "actions": [],
            "temporal_cues": [],
            "context": "",
            "complexity": "moderate",
            "search_variations": [original_query]
        }
        
        # Extract visual elements (common object words)
        visual_pattern = r'\b(person|people|man|woman|child|car|building|room|table|chair|phone|computer|book|food|clothing|color|red|blue|green|yellow|black|white)\b'
        visual_elements = re.findall(visual_pattern, response_text.lower())
        analysis["visual_elements"] = list(set(visual_elements))
        
        # Extract actions (common verb patterns)
        action_pattern = r'\b(walking|running|sitting|standing|talking|eating|drinking|reading|writing|playing|working|sleeping|dancing|singing|laughing|crying)\b'
        actions = re.findall(action_pattern, response_text.lower())
        analysis["actions"] = list(set(actions))
        
        # Extract temporal cues
        temporal_pattern = r'\b(before|after|during|while|when|then|first|last|beginning|end|start|finish|long|short|quick|slow)\b'
        temporal_cues = re.findall(temporal_pattern, response_text.lower())
        analysis["temporal_cues"] = list(set(temporal_cues))
        
        return analysis
    
    async def _generate_semantic_variations(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate semantic variations using embeddings and analysis"""
        
        variations = [query]
        
        try:
            # Generate variations based on analysis
            if analysis.get("visual_elements"):
                visual_variation = f"{' '.join(analysis['visual_elements'][:3])} {analysis['primary_focus']}"
                variations.append(visual_variation)
            
            if analysis.get("actions"):
                action_variation = f"{analysis['primary_focus']} {' '.join(analysis['actions'][:2])}"
                variations.append(action_variation)
            
            # Add context-based variation
            if analysis.get("context"):
                context_variation = f"{analysis['context']} {analysis['primary_focus']}"
                variations.append(context_variation)
            
            # Ensure uniqueness
            variations = list(dict.fromkeys(variations))
            
        except Exception as e:
            print(f"Warning: Semantic variation generation failed: {e}")
        
        return variations[:4]  # Limit to 4 variations


class QueryUnderstandingModule:
    """ query understanding with entity extraction and temporal parsing"""
    
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
