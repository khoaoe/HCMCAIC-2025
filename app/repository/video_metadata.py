from typing import List, Optional, Dict, Any, Tuple
from common.repository.base import MongoBaseRepository
from models.video_metadata import VideoMetadata


class VideoMetadataRepository(MongoBaseRepository[VideoMetadata]):
    """Repository for video metadata operations"""
    
    def __init__(self):
        super().__init__(VideoMetadata)
    
    async def search_by_text(
        self, 
        query: Optional[str] = None, 
        filter_author: Optional[str] = None,
        filter_keywords: Optional[List[str]] = None,
        filter_publish_date: Optional[str] = None,
        limit: int = 100
    ) -> List[VideoMetadata]:
        """
        Search video metadata by text with optional filters. If query is None,
        performs filters-only search.
        """
        # Build match conditions
        conditions: List[Dict[str, Any]] = []
        
        if query and query.strip():
            conditions.append({
                "$text": {
                    "$search": query,
                    "$caseSensitive": False,
                    "$diacriticSensitive": False
                }
            })
        
        if filter_author:
            conditions.append({"author": {"$regex": filter_author, "$options": "i"}})
        
        if filter_keywords:
            conditions.append({"keywords": {"$in": filter_keywords}})
        
        if filter_publish_date:
            conditions.append({"publish_date": filter_publish_date})
        
        search_query: Dict[str, Any]
        if conditions:
            if len(conditions) == 1:
                search_query = conditions[0]
            else:
                search_query = {"$and": conditions}
        else:
            # No conditions: return empty to avoid scanning whole collection
            return []
        
        # Build pipeline with optional text score
        pipeline: List[Dict[str, Any]] = [{"$match": search_query}]
        if query and query.strip():
            pipeline += [
                {"$addFields": {"score": {"$meta": "textScore"}}},
                {"$sort": {"score": {"$meta": "textScore"}}},
            ]
        pipeline.append({"$limit": limit})
        
        return await self.find_pipeline(pipeline)
    
    async def search_by_filters(
        self,
        filter_author: Optional[str] = None,
        filter_keywords: Optional[List[str]] = None,
        filter_publish_date: Optional[str] = None,
        limit: int = 200
    ) -> List[Tuple[int, int]]:
        """Return (group_num, video_num) pairs matching filters only."""
        result_docs = await self.search_by_text(
            query=None,
            filter_author=filter_author,
            filter_keywords=filter_keywords,
            filter_publish_date=filter_publish_date,
            limit=limit,
        )
        pairs: List[Tuple[int, int]] = []
        for doc in result_docs:
            try:
                pairs.append((int(doc.group_num), int(doc.video_num)))
            except Exception:
                continue
        return pairs
    
    async def get_by_video_id(self, video_id: str) -> Optional[VideoMetadata]:
        result = await self.collection.find_one({"video_id": video_id})
        return result
    
    async def get_by_group_and_video(self, group_num: int, video_num: int) -> Optional[VideoMetadata]:
        result = await self.collection.find_one({
            "group_num": group_num,
            "video_num": video_num
        })
        return result
    
    async def get_all_video_ids(self) -> List[str]:
        pipeline = [
            {"$project": {"video_id": 1}},
            {"$group": {"_id": None, "video_ids": {"$push": "$video_id"}}}
        ]
        result = await self.find_pipeline(pipeline)
        if result:
            return result[0].video_ids
        return []
    
    async def upsert_metadata(self, metadata: VideoMetadata) -> bool:
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
