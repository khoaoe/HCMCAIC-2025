"""
Enhanced prompts for competition tasks
Specialized prompts optimized for VCMR, VQA, and KIS performance
"""

from llama_index.core import PromptTemplate


class CompetitionPrompts:
    """Collection of optimized prompts for competition tasks"""
    
    # Query Analysis for Adaptive Strategy
    QUERY_ANALYSIS_PROMPT = PromptTemplate(
        """
        Bạn là một chuyên gia phân tích truy vấn tìm kiếm video. Hãy phân tích truy vấn của người dùng và trả về một cấu trúc JSON.

        Danh sách các lớp đối tượng COCO có sẵn để tham khảo: {coco}

        Truy vấn người dùng: "{query}"

        Nhiệm vụ của bạn:
        1.  **Phân loại truy vấn (query_type)**:
            -   'object-centric': Nếu truy vấn tập trung chính vào việc tìm một hoặc nhiều đối tượng cụ thể (ví dụ: "tìm xe ô tô màu đỏ", "người đàn ông đội mũ bảo hiểm").
            -   'action-centric': Nếu truy vấn mô tả một hành động hoặc sự kiện là chính (ví dụ: "người đang đi bộ qua đường", "hai người đang nói chuyện").
            -   'scene-descriptive': Nếu truy vấn mô tả một cảnh quan hoặc bối cảnh chung (ví dụ: "cảnh hoàng hôn trên bãi biển", "đường phố đông đúc về đêm").
            -   'abstract': Nếu truy vấn chứa các khái niệm trừu tượng, cảm xúc (ví dụ: "khoảnh khắc gia đình hạnh phúc", "một cuộc trò chuyện căng thẳng").

        2.  **Trích xuất đối tượng khóa (key_objects)**:
            -   Chỉ liệt kê các đối tượng **quan trọng nhất** và **có trong danh sách COCO** được cung cấp.
            -   Nếu không có đối tượng nào rõ ràng hoặc truy vấn là trừu tượng, trả về một danh sách rỗng [].

        3.  **Đánh giá yêu cầu ngữ cảnh (requires_contextual_understanding)**:
            -   Trả về `true` nếu truy vấn thuộc loại 'action-centric', 'scene-descriptive', hoặc 'abstract'.
            -   Trả về `false` nếu truy vấn chỉ đơn thuần là 'object-centric' và không có yếu tố phức tạp nào khác.

        Chỉ trả về duy nhất một đối tượng JSON hợp lệ.
        """
    )
    
    # Visual Event Extraction for VCMR
    VCMR_VISUAL_EXTRACTION = PromptTemplate(
        """
        You are an expert video moment retrieval system. Extract and optimize the query for semantic video search.
        
        COCO Objects Available: {coco}
        
        Original Query: {query}
        
        Your task:
        1. Extract key visual elements, actions, and temporal cues from the ORIGINAL query
        2. Create a refined query that PRESERVES the user's intent while optimizing for search
        3. Generate semantic variations that EXPAND coverage without over-specifying
        4. Identify relevant COCO objects ONLY if they are explicitly mentioned or clearly implied
        
        CRITICAL GUIDELINES:
        - PRESERVE the user's original intent - do not add details not present in the query
        - Focus on VISUAL elements that are observable in video frames
        - Use the refined query as a SEARCH OPTIMIZATION, not a complete rewrite
        - Generate variations that are SEMANTICALLY RELATED but not overly specific
        - Only suggest objects that are EXPLICITLY mentioned or clearly implied
        - Avoid "hallucinating" details like specific settings, lighting, or environmental context unless stated
        
        Example Good Approach:
        Query: "a woman places a picture and drives to store"
        Refined: "woman placing picture frame, person driving car"
        Variations: 
        - "person handling picture frame object"
        - "woman driving vehicle transportation"
        - "picture frame placement activity"
        
        Example Bad Approach (AVOID):
        Query: "a woman places a picture and drives to store"
        Refined: "woman hanging framed picture on wall indoor home setting, woman driving car vehicle outdoor road"
        (This adds too many specific details not in the original query)
        
        Return:
        - refined_query: Optimized search query that preserves user intent
        - list_of_objects: Relevant COCO objects (only if explicitly mentioned or clearly implied)
        - query_variations: 3-4 semantic variations for comprehensive search coverage
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
        
        Combine all hints into an search strategy:
        
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
        Align ASR text with visual content for moment understanding.
        
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
            "query_analysis": cls.QUERY_ANALYSIS_PROMPT,
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
