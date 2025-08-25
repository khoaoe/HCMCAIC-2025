import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)
sys.path.insert(0, ROOT_DIR)

from typing import Dict, List, Optional
from pathlib import Path
import json

from agent.agent import VisualEventExtractor
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from llama_index.core.llms import LLM


class AgentController:
     
    def __init__(
        self,
        llm: LLM,
        keyframe_service: KeyframeQueryService,
        model_service: ModelService,
        data_folder: str,
        objects_data: Optional[Dict] = None,
        asr_data: Optional[Dict] = None,
        top_k: int = 200
    ):
        
        objects_data = objects_data or {}
        asr_data = asr_data or {}

        self.agent = VisualEventExtractor(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            top_k=top_k
        )
    
    def _load_json_data(self, path: Path):
        return json.load(open(path))



    async def search_and_answer(self, user_query: str) -> str:
        return await self.agent.process_query(user_query)