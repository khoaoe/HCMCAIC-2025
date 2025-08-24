from pydantic import BaseModel, Field
from typing import List

class AgentResponse(BaseModel):
    refined_query: str = Field(..., description="The rephrased response")
    list_of_objects: list[str] | None = Field(None, description="The list of objects for filtering(Object from COCO class), optional")
    query_variations: list[str] | None = Field(None, description="Semantic variations of the query for comprehensive retrieval")

class QueryAnalysisResult(BaseModel):
    query_type: str = Field(..., description="Phân loại truy vấn: 'object-centric', 'action-centric', 'scene-descriptive', hoặc 'abstract'.")
    key_objects: List[str] = Field(default_factory=list, description="Danh sách các đối tượng quan trọng, cụ thể có trong COCO class để ưu tiên.")
    requires_contextual_understanding: bool = Field(..., description="True nếu truy vấn yêu cầu hiểu biết về hành động, mối quan hệ, hoặc khái niệm trừu tượng.")

class QueryRefineResponse(BaseModel):
    translated_query: str = Field(..., description="Input translated to English (or original if already English)")
    enhanced_query: str = Field(..., description="Optimized English query for retrieval")
    
    
class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=2000)


class AgentQueryResponse(BaseModel):
    """Response model for agent queries"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")


