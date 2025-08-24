from beanie import Document, Indexed
from typing import Annotated, Optional
from pydantic import BaseModel, Field


class Keyframe(Document):
    key: Annotated[int, Indexed(unique=True)]
    video_num: Annotated[int, Indexed()]
    group_num: Annotated[int, Indexed()]
    keyframe_num: Annotated[int, Indexed()]
    phash: Optional[str] = Field(None, description="Perceptual hash of the keyframe image")

    class Settings:
        name = "keyframes"



    