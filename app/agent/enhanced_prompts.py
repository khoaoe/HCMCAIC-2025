"""
Enhanced prompts for competition tasks
Specialized prompts optimized for VCMR, VQA, and KIS performance
"""

from llama_index.core import PromptTemplate


class CompetitionPrompts:
    """Collection of optimized prompts for competition tasks"""
    
    # Enhanced Visual Event Extraction for VCMR
    VCMR_VISUAL_EXTRACTION = PromptTemplate(
        """
        You are an expert video moment retrieval system. Extract and optimize the query for semantic video search.
        
        COCO Objects Available: {coco}
        
        Original Query: {query}
        
        Your task:
        1. Extract key visual elements, actions, and temporal cues
        2. Rephrase for optimal embedding search (focus on concrete, visual terms)
        3. Identify relevant COCO objects that would help filter results
        4. Consider temporal relationships and action sequences
        
        Guidelines:
        - Prioritize visual, observable elements over abstract concepts
        - Include action verbs and spatial relationships
        - Consider lighting, setting, and environmental context
        - Extract person characteristics, object interactions, movement patterns
        
        Return:
        - refined_query: Optimized search query (focus on visual semantics)
        - list_of_objects: Relevant COCO objects for filtering (only if helpful for precision)
        
        Example:
        Query: "A woman places a picture and drives to store"
        Output: 
        - refined_query: "woman hanging framed picture on wall indoor setting, woman driving car vehicle outdoor"
        - list_of_objects: ["person", "car", "picture frame"] (if these help narrow results)
        """
    )
    
    # Video QA Answer Generation
    VIDEO_QA_ANSWER = PromptTemplate(
        """
        You are an expert video content analyzer. Answer the question precisely based on visual evidence.
        
        Question: {question}
        Video: {video_id}
        Time Range: {clip_range}
        
        Visual Evidence:
        {keyframes_info}
        
        Additional Context:
        {context_info}
        
        Instructions:
        1. Provide a direct, factual answer based on visual evidence
        2. For counting questions: Give exact numbers ("3 people", "2 cars")
        3. For identification: Provide specific names/labels if visible
        4. For yes/no questions: Be definitive based on evidence
        5. If uncertain, express confidence level
        6. Reference specific visual elements that support your answer
        
        Answer format: Concise, factual response (typically 1-3 sentences)
        """
    )
    
    # KIS Precision Matching
    KIS_PRECISION_SEARCH = PromptTemplate(
        """
        You are searching for an EXACT segment match. Analyze the description for unique identifiers.
        
        Target Description: {description}
        
        Extract the most unique, identifying features:
        1. Specific objects, colors, shapes
        2. Unique actions or interactions  
        3. Distinctive visual elements
        4. Spatial arrangements or compositions
        5. Notable details that distinguish this moment
        
        Create a precise search query that captures the uniqueness of this specific moment.
        Focus on features that would NOT appear in similar but different segments.
        
        Return a highly specific search query optimized for exact matching.
        """
    )
    
    # Interactive Feedback Integration
    FEEDBACK_INTEGRATION = PromptTemplate(
        """
        Integrate user feedback to refine video moment search.
        
        Original Query: {original_query}
        Previous Result: {previous_result}
        User Feedback: {feedback}
        
        Based on the feedback, modify the search strategy:
        
        For "not relevant" feedback:
        - What aspects should be avoided or filtered out?
        - What alternative interpretations might be more accurate?
        
        For refinement text:
        - How does this change the search focus?
        - What new constraints or requirements are added?
        
        For relevance scores:
        - What confidence threshold should be applied?
        - How should scoring criteria be adjusted?
        
        Return an updated search query and filtering strategy.
        """
    )
    
    # Cross-Modal Reranking
    CROSS_MODAL_RERANKING = PromptTemplate(
        """
        Rerank video moment candidates based on multi-modal evidence.
        
        Original Query: {query}
        
        Candidates to rank:
        {candidates_info}
        
        Ranking Criteria:
        1. Visual Content Alignment (40%):
           - Objects, scenes, actions match query
           - Visual composition and setting relevance
        
        2. Temporal Context (30%):
           - Action sequences and timing
           - Event progression and causality
        
        3. Textual Context (20%):
           - ASR speech content relevance
           - Spoken keywords and context
        
        4. Object/Scene Coherence (10%):
           - Detected objects support the query
           - Scene setting matches expectations
        
        For each candidate, provide:
        - Relevance score (0.0-1.0)
        - Brief rationale (2-3 sentences)
        - Key supporting evidence
        
        Return candidates ranked by overall relevance score.
        """
    )
    
    # Progressive Hint Integration for KIS-C
    PROGRESSIVE_HINT_INTEGRATION = PromptTemplate(
        """
        Integrate progressive hints to refine known-item search.
        
        Initial Hint: {initial_hint}
        Additional Hints: {additional_hints}
        Search History: {search_history}
        
        Combine all hints into an enhanced search strategy:
        
        1. Identify complementary information across hints
        2. Resolve any contradictions or conflicts
        3. Prioritize more specific/recent hints
        4. Build comprehensive target description
        
        Create a refined search query that integrates all available information
        while maintaining precision for exact segment matching.
        
        Consider:
        - Which hints provide the most distinctive information?
        - How do hints relate to each other temporally/spatially?
        - What unique combination of features emerges?
        
        Return optimized search query for exact target localization.
        """
    )
    
    # ASR-Visual Alignment
    ASR_VISUAL_ALIGNMENT = PromptTemplate(
        """
        Align ASR text with visual content for enhanced moment understanding.
        
        Visual Content: {visual_description}
        ASR Text: {asr_text}
        Query Context: {query}
        
        Analyze how speech and visual content work together:
        
        1. Content Synchronization:
           - Do spoken words describe visible actions?
           - Are there audio-visual correlations?
        
        2. Contextual Enhancement:
           - Does ASR provide names, locations, or details not visible?
           - Are there off-screen references that add context?
        
        3. Temporal Alignment:
           - Do spoken events match visual timeline?
           - Are there lag/lead relationships between audio and visual?
        
        Provide an integrated understanding that combines both modalities
        for more accurate moment retrieval and question answering.
        """
    )
    
    @classmethod
    def get_prompt(cls, prompt_name: str) -> PromptTemplate:
        """Get a specific prompt by name"""
        prompt_mapping = {
            "vcmr_visual_extraction": cls.VCMR_VISUAL_EXTRACTION,
            "video_qa_answer": cls.VIDEO_QA_ANSWER,
            "kis_precision_search": cls.KIS_PRECISION_SEARCH,
            "feedback_integration": cls.FEEDBACK_INTEGRATION,
            "cross_modal_reranking": cls.CROSS_MODAL_RERANKING,
            "progressive_hint_integration": cls.PROGRESSIVE_HINT_INTEGRATION,
            "asr_visual_alignment": cls.ASR_VISUAL_ALIGNMENT
        }
        
        if prompt_name not in prompt_mapping:
            raise ValueError(f"Unknown prompt: {prompt_name}. Available: {list(prompt_mapping.keys())}")
        
        return prompt_mapping[prompt_name]
