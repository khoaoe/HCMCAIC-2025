from typing import List, Optional, Dict, Any
from common.repository.base import MongoBaseRepository
from models.video_metadata import VideoMetadata


class VideoMetadataRepository(MongoBaseRepository[VideoMetadata]):
    """Repository for video metadata operations"""
    
    def __init__(self):
        super().__init__(VideoMetadata)
    
    async def search_by_text(
        self, 
        query: str, 
        filter_author: Optional[str] = None,
        filter_keywords: Optional[List[str]] = None,
        filter_publish_date: Optional[str] = None,
        limit: int = 100
    ) -> List[VideoMetadata]:
        """
        Search video metadata by text with optional filters
        
        Args:
            query: Text query to search in title, description, and keywords
            filter_author: Filter by specific author
            filter_keywords: Filter by specific keywords
            filter_publish_date: Filter by publish date
            limit: Maximum number of results to return
            
        Returns:
            List of VideoMetadata documents matching the criteria
        """
        # Build text search query
        text_search = {
            "$text": {
                "$search": query,
                "$caseSensitive": False,
                "$diacriticSensitive": False
            }
        }
        
        # Build filter conditions
        filter_conditions = {}
        
        if filter_author:
            filter_conditions["author"] = {"$regex": filter_author, "$options": "i"}
        
        if filter_keywords:
            filter_conditions["keywords"] = {"$in": filter_keywords}
        
        if filter_publish_date:
            filter_conditions["publish_date"] = filter_publish_date
        
        # Combine text search with filters
        if filter_conditions:
            search_query = {
                "$and": [
                    text_search,
                    filter_conditions
                ]
            }
        else:
            search_query = text_search
        
        # Execute search with text score sorting
        pipeline = [
            {"$match": search_query},
            {"$addFields": {"score": {"$meta": "textScore"}}},
            {"$sort": {"score": {"$meta": "textScore"}}},
            {"$limit": limit}
        ]
        
        return await self.find_pipeline(pipeline)
    
    async def get_by_video_id(self, video_id: str) -> Optional[VideoMetadata]:
        """
        Get video metadata by video_id
        
        Args:
            video_id: The video ID to search for
            
        Returns:
            VideoMetadata document if found, None otherwise
        """
        result = await self.collection.find_one({"video_id": video_id})
        return result
    
    async def get_by_group_and_video(self, group_num: int, video_num: int) -> Optional[VideoMetadata]:
        """
        Get video metadata by group number and video number
        
        Args:
            group_num: Group number
            video_num: Video number within the group
            
        Returns:
            VideoMetadata document if found, None otherwise
        """
        result = await self.collection.find_one({
            "group_num": group_num,
            "video_num": video_num
        })
        return result
    
    async def get_all_video_ids(self) -> List[str]:
        """
        Get all video IDs from the collection
        
        Returns:
            List of all video IDs
        """
        pipeline = [
            {"$project": {"video_id": 1}},
            {"$group": {"_id": None, "video_ids": {"$push": "$video_id"}}}
        ]
        
        result = await self.find_pipeline(pipeline)
        if result:
            return result[0].video_ids
        return []
    
    async def upsert_metadata(self, metadata: VideoMetadata) -> bool:
        """
        Insert or update video metadata
        
        Args:
            metadata: VideoMetadata document to upsert
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.collection.find_one_and_replace(
                {"video_id": metadata.video_id},
                metadata.dict(),
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error upserting metadata for {metadata.video_id}: {e}")
            return False
