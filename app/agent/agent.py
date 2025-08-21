import re
from typing import  cast
from llama_index.core.llms import LLM
from llama_index.core import PromptTemplate
from schema.agent import AgentResponse
from pathlib import Path

from typing import Dict, List, Tuple, Any
from collections import defaultdict
from schema.response import KeyframeServiceReponse
import os
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from dataclasses import dataclass


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


@dataclass
class AgentResponse:
    refined_query: str
    list_of_objects: List[str]


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
        return obj
    

    @staticmethod
    def calculate_video_scores(keyframes: List[KeyframeServiceReponse]) -> List[Tuple[float, List[KeyframeServiceReponse]]]:
        """
        Calculate average scores for each video and return sorted by score
        
        Returns:
            List of tuples: (video_num, average_score, keyframes_in_video)
        """
        video_keyframes: Dict[str, List[KeyframeServiceReponse]] = defaultdict(list)
        
        for keyframe in keyframes:
            video_keyframes[f"{keyframe.group_num}/{keyframe.video_num}"].append(keyframe)
        
        video_scores: List[Tuple[float, List[KeyframeServiceReponse]]] = []
        for _, video_keyframes_list in video_keyframes.items():
            avg_score = sum(kf.confidence_score for kf in video_keyframes_list) / len(video_keyframes_list)
            video_scores.append((avg_score, video_keyframes_list))
        
        video_scores.sort(key=lambda x: x[0], reverse=True)
        
        return video_scores
    



class AnswerGenerator:
    """Generates final answers based on refined keyframes"""
    
    def __init__(self, llm: LLM, data_folder: str):
        self.data_folder = data_folder
        self.llm = llm
        self.answer_prompt = PromptTemplate(
            """
            Based on the user's query and the relevant keyframes found, generate a comprehensive answer.
            
            Original Query and questions: {query}
            
            Relevant Keyframes:
            {keyframes_context}
            
            Please provide a detailed answer that:
            1. Directly addresses the user's query
            2. References specific information from the keyframes
            3. Synthesizes information across multiple keyframes if relevant
            4. Mentions which videos/keyframes contain the most relevant content
            
            Keep the answer informative but concise.
            """
        )
    
    async def generate_answer(
        self,
        original_query: str,
        final_keyframes: List[KeyframeServiceReponse],
        objects_data: Dict[str, List[str]],
        asr_data: str = ""
    ):
        chat_messages = []
        for kf in final_keyframes:
            keyy = f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg"
            objects = objects_data.get(keyy, [])

            image_path = os.path.join(self.data_folder, f"L{str(kf.group_num):0>2s}/L{str(kf.group_num):0>2s}_V{str(safe_convert_video_num(kf.video_num)):0>3s}/{str(kf.keyframe_num):0>3d}.jpg")

            context_text = f"""
            Keyframe {kf.key} from Video {kf.video_num} (Confidence: {kf.confidence_score:.3f}):
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
        keyframes_context = "See the keyframes and their context above"
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







