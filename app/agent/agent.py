from typing import  cast
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from schema.agent import AgentResponse
from pathlib import Path

from typing import Dict, List
from schema.response import KeyframeServiceReponse
import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole



from utils.video_utils import safe_convert_video_num




COCO_CLASS = """
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
"""

class VisualEventExtractor:
    
    def __init__(self, llm: LLM):
        self.llm = llm
        # Import competition prompts
        from .prompts import CompetitionPrompts
        self.extraction_prompt = CompetitionPrompts.VCMR_VISUAL_EXTRACTION

    async def extract_visual_events(self, query: str) -> AgentResponse:
        prompt = self.extraction_prompt.format(query=query, coco=COCO_CLASS)
        response = await self.llm.as_structured_llm(AgentResponse).acomplete(prompt)
        obj = cast(AgentResponse, response.raw)
        
        # Generate query variations for robust search
        if obj.query_variations is None:
            obj.query_variations = await self._generate_query_variations(query, obj.refined_query)
        
        return obj
    
    async def _generate_query_variations(self, original_query: str, refined_query: str) -> List[str]:
        """Generate semantic variations of the query for better coverage"""
        
        expansion_prompt = PromptTemplate(
            """
            Generate 3-4 semantic variations of this video search query for comprehensive retrieval.
            
            Original Query: {original_query}
            Refined Query: {refined_query}
            
            Create variations that:
            1. Focus on different visual aspects (objects, actions, settings) from the ORIGINAL query
            2. Use alternative phrasings and synonyms that PRESERVE the original meaning
            3. Vary in specificity but stay within the scope of what was actually described
            4. Consider related concepts that might appear alongside the main elements
            5. Maintain focus on what is VISUALLY OBSERVABLE in video frames
            
            CRITICAL RULES:
            - DO NOT add details not present in the original query
            - DO NOT specify settings, lighting, or environmental context unless mentioned
            - DO NOT over-specify actions or add inferred details
            - Focus on SEMANTIC EXPANSION, not DETAIL ADDITION
            - Each variation should capture a different aspect of the SAME content
            
            Guidelines:
            - Use synonyms and alternative phrasings
            - Focus on different visual elements mentioned in the original
            - Consider broader and more specific versions of the same concept
            - Keep variations concise and focused on visual content
            
            Return 3-4 distinct query variations, each on a new line.
            Do not include the original or refined queries.
            
            Example Good Variations:
            Original: "a woman places a picture and drives to store"
            Variations:
            person handling picture frame object
            woman driving vehicle transportation
            picture frame placement activity
            automobile movement to building
            
            Example Bad Variations (AVOID):
            woman hanging framed picture on wall indoor home setting
            person driving car vehicle on road outdoor
            (These add too many specific details not in the original)
            """
        )
        
        try:
            prompt = expansion_prompt.format(
                original_query=original_query,
                refined_query=refined_query
            )
            response = await self.llm.acomplete(prompt)
            
            # Parse response into list of variations
            variations = [refined_query]  # Start with refined query
            response_text = str(response).strip()
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line and line not in variations:
                    # Remove numbering and bullet points
                    cleaned = line.lstrip('123456789.- ').strip()
                    if cleaned and len(cleaned) > 10:  # Valid query length
                        # Additional validation: check if variation is not too specific
                        if not self._is_over_specific(cleaned, original_query):
                            variations.append(cleaned)
            
            # Ensure we have the original refined query and variations
            if refined_query not in variations:
                variations.insert(0, refined_query)
            
            return variations[:5]  # Limit to 5 total variations
            
        except Exception as e:
            print(f"Warning: Query variation generation failed: {e}")
            return [refined_query]  # Fallback to just refined query
    
    def _is_over_specific(self, variation: str, original_query: str) -> bool:
        """Check if a variation adds too many specific details not in the original query"""
        # List of over-specific terms that shouldn't be added unless in original
        over_specific_terms = [
            'indoor', 'outdoor', 'home', 'office', 'kitchen', 'bedroom', 'living room',
            'road', 'street', 'highway', 'parking lot', 'garage', 'driveway',
            'wall', 'ceiling', 'floor', 'table', 'desk', 'shelf',
            'morning', 'afternoon', 'evening', 'night', 'daytime', 'nighttime',
            'bright', 'dark', 'lighting', 'shadow', 'sunlight', 'artificial light',
            'framed', 'hanging', 'mounted', 'displayed', 'positioned'
        ]
        
        original_lower = original_query.lower()
        variation_lower = variation.lower()
        
        # Check if variation contains over-specific terms not in original
        for term in over_specific_terms:
            if term in variation_lower and term not in original_lower:
                return True
        
        return False
    

    # REMOVED: calculate_video_scores function - This function implemented the flawed "best video only" 
    
class AnswerGenerator:
    """Generates final answers based on refined keyframes"""
    
    def __init__(self, llm: LLM, data_folder: str):
        self.data_folder = data_folder
        self.llm = llm
        self.answer_prompt = PromptTemplate(
            """
            Based on the user's query and the relevant keyframes found across multiple videos, generate a comprehensive answer.
            
            Original Query: {query}
            
            Search Results Context: {keyframes_context}
            
            Please provide a detailed answer that:
            1. Directly addresses the user's query with specific information from the keyframes
            2. Synthesizes information across multiple videos if relevant
            3. Identifies which video(s) contain the most relevant content
            4. Explains the temporal sequence or spatial relationships if applicable
            5. References specific keyframes and their confidence scores
            6. Mentions any detected objects or visual elements that support the answer
            
            If multiple videos contain relevant content:
            - Explain how the content relates across videos
            - Highlight the most important video(s) and why
            - Provide a comprehensive view of the event or scene
            
            Keep the answer informative, well-structured, and directly relevant to the query.
            """
        )
    
    async def generate_answer(
        self,
        original_query: str,
        final_keyframes: List[KeyframeServiceReponse],
        objects_data: Dict[str, List[str]],
        asr_data: str = ""
    ):
        # Group keyframes by video for better organization
        video_groups = {}
        for kf in final_keyframes:
            video_key = f"Video {kf.video_num} (Group {kf.group_num})"
            if video_key not in video_groups:
                video_groups[video_key] = []
            video_groups[video_key].append(kf)
        
        chat_messages = []
        
        # Add summary information about the search results
        summary_text = f"""
        Search Results Summary:
        - Total keyframes found: {len(final_keyframes)}
        - Videos containing relevant content: {len(video_groups)}
        - Videos: {', '.join(video_groups.keys())}
        
        Detailed keyframe analysis:
        """
        
        summary_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=summary_text)]
        )
        chat_messages.append(summary_message)
        
        # Process keyframes by video group
        for video_name, video_keyframes in video_groups.items():
            video_summary = f"\n=== {video_name} ==="
            video_message = ChatMessage(
                role=MessageRole.USER,
                content=[TextBlock(text=video_summary)]
            )
            chat_messages.append(video_message)
            
            for kf in video_keyframes:
                keyy = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
                objects = objects_data.get(keyy, [])

                image_path = os.path.join(self.data_folder, f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg")

                context_text = f"""
                Keyframe {kf.key} from {video_name} (Confidence: {kf.confidence_score:.3f}):
                - Frame number: {kf.keyframe_num}
                - Detected Objects: {', '.join(objects) if objects else 'None detected'}
                """

                if os.path.exists(image_path):
                    message_content = [
                        ImageBlock(path=Path(image_path)),
                        TextBlock(text=context_text)
                    ]   
                else:
                    message_content = [TextBlock(text=context_text + "\n(Image not available)")]
                
                user_message = ChatMessage(
                    role=MessageRole.USER,
                    content=message_content
                )

                chat_messages.append(user_message)

        
        # Include ASR context in the prompt
        keyframes_context = f"""
        Search Results Analysis:
        - Found {len(final_keyframes)} relevant keyframes across {len(video_groups)} videos
        - Keyframes are organized by video for better context
        - Each keyframe includes confidence score and detected objects
        - Results came from multi-variant semantic search for comprehensive coverage
        """
        
        if asr_data:
            keyframes_context += f"\n\nRelevant ASR Text: {asr_data}"
        
        final_prompt = self.answer_prompt.format(
            query=original_query,
            keyframes_context=keyframes_context
        ) 
        query_message = ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text=final_prompt)]
        )
        chat_messages.append(query_message)

        response = await self.llm.achat(chat_messages)
        return response.message.content







