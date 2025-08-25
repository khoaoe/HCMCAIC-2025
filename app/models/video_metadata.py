from beanie import Document, Indexed
from typing import List, Optional
from datetime import datetime


class VideoMetadata(Document):
    """Video metadata document for storing video information"""
    
    # Core identification fields
    video_id: Indexed(str, unique=True)  # Primary key, indexed
    group_num: int
    video_num: int
    
    # Content information
    title: str
    description: str
    keywords: List[str]
    author: str
    publish_date: str
    
    # Technical specifications
    duration: float  # in seconds
    total_frames: int
    fps: float
    
    # Metadata
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    
    class Settings:
        name = "video_metadata"
        indexes = [
            # Text index for optimized text search
            [
                ("title", "text"),
                ("description", "text"), 
                ("keywords", "text")
            ],
            # Compound indexes for common queries (ascending order)
            [("group_num", 1), ("video_num", 1)],
            [("author", 1), ("publish_date", 1)],
            [("keywords", 1)],
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "L21_V001",
                "group_num": 21,
                "video_num": 1,
                "title": "Sample Video Title",
                "description": "This is a sample video description",
                "keywords": ["sample", "video", "test"],
                "author": "Sample Author",
                "publish_date": "2024-01-01",
                "duration": 120.5,
                "total_frames": 3600,
                "fps": 30.0
            }
        }
