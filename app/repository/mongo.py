"""
The implementation of Keyframe repositories. The following class is responsible for getting the keyframe by many ways
"""

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

from typing import Any
from models.keyframe import Keyframe
from common.repository import MongoBaseRepository
from schema.interface import KeyframeInterface


from utils.video_utils import safe_convert_video_num


class KeyframeRepository(MongoBaseRepository[Keyframe]):
    async def get_keyframe_by_list_of_keys(
        self, keys: list[int]
    ):
        result = await self.find({"key": {"$in": keys}})
        
        # Debug: Check for corrupted video_num in database
        for keyframe in result[:5]:  # Check first 5 keyframes
            if isinstance(keyframe.video_num, str) and '_V' in keyframe.video_num:
                print(f"WARNING: Corrupted video_num in database: {keyframe.video_num} for keyframe {keyframe.key}")
        
        return [
            KeyframeInterface(
                key=int(keyframe.key),
                video_num=safe_convert_video_num(keyframe.video_num),
                group_num=int(keyframe.group_num),
                keyframe_num=int(keyframe.keyframe_num),
                phash=keyframe.phash
            ) for keyframe in result

        ]

    async def get_keyframe_by_video_num(
        self, 
        video_num: int,
    ):
        result = await self.find({"video_num": video_num})
        return [
            KeyframeInterface(
                key=int(keyframe.key),
                video_num=safe_convert_video_num(keyframe.video_num),
                group_num=int(keyframe.group_num),
                keyframe_num=int(keyframe.keyframe_num),
                phash=keyframe.phash
            ) for keyframe in result
        ]

    async def get_keyframe_by_keyframe_num(
        self, 
        keyframe_num: int,
    ):
        result = await self.find({"keyframe_num": keyframe_num})
        return [
            KeyframeInterface(
                key=int(keyframe.key),
                video_num=safe_convert_video_num(keyframe.video_num),
                group_num=int(keyframe.group_num),
                keyframe_num=int(keyframe.keyframe_num),
                phash=keyframe.phash
            ) for keyframe in result
        ]   


