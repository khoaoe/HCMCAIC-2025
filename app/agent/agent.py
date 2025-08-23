from typing import  cast
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from schema.agent import AgentResponse
from pathlib import Path

from typing import Dict, List
from schema.response import KeyframeServiceReponse
import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole



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
        # Import enhanced competition prompts
        from .enhanced_prompts import CompetitionPrompts
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
            1. Focus on different visual aspects (objects, actions, settings)
            2. Use alternative phrasings and synonyms
            3. Vary in specificity (broader and more specific)
            4. Consider temporal relationships and spatial arrangements
            5. Include related concepts that might appear in the same scene
            
            Guidelines:
            - Each variation should capture a different semantic aspect
            - Maintain focus on visual, observable elements
            - Consider what related content might appear alongside the main query
            - Keep variations concise but descriptive
            
            Return 3-4 distinct query variations, each on a new line.
            Do not include the original or refined queries.
            
            Example:
            Original: "a woman places a picture and drives to store"
            Variations:
            woman hanging artwork on wall indoor home setting
            person driving car vehicle on road outdoor
            female handling picture frame decoration object
            automobile transportation moving to building location
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
                        variations.append(cleaned)
            
            # Ensure we have the original refined query and variations
            if refined_query not in variations:
                variations.insert(0, refined_query)
            
            return variations[:5]  # Limit to 5 total variations
            
        except Exception as e:
            print(f"Warning: Query variation generation failed: {e}")
            return [refined_query]  # Fallback to just refined query
    

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







