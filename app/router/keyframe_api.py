
from fastapi import APIRouter, Depends
from typing import List

from schema.request import KeyframeSearchRequest
from schema.response import KeyframeServiceReponse, SingleKeyframeDisplay, KeyframeDisplay
from controller.query_controller import QueryController
from core.dependencies import get_query_controller
from core.logger import SimpleLogger


logger = SimpleLogger(__name__)


router = APIRouter(
    prefix="/keyframe",
    tags=["keyframe"],
    responses={404: {"description": "Not found"}},
)


@router.post(
    "/search",
    response_model=KeyframeDisplay,
    summary="Unified keyframe search",
    description="Text search with optional filters: include/exclude groups/videos, time range, and hybrid metadata filters."
)
async def search_keyframes(
    request: KeyframeSearchRequest,
    controller: QueryController = Depends(get_query_controller)
):
    logger.info(f"Unified search: query='{request.query}', top_k={request.top_k}")

    results: List[KeyframeServiceReponse]

    # Route based on provided filters
    if request.include_groups or request.include_videos:
        results = await controller.search_with_selected_video_group(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            list_of_include_groups=request.include_groups or [],
            list_of_include_videos=request.include_videos or []
        )
    elif request.exclude_groups:
        results = await controller.search_text_with_exlude_group(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            list_group_exlude=request.exclude_groups
        )
    elif request.use_hybrid_search or any([request.filter_author, request.filter_keywords, request.filter_publish_date]):
        results = await controller.search_hybrid(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filter_author=request.filter_author,
            filter_keywords=request.filter_keywords,
            filter_publish_date=request.filter_publish_date,
            metadata_weight=request.metadata_weight
        )
    else:
        results = await controller.search_text(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )

    display_results = list(
        map(
            lambda pair: SingleKeyframeDisplay(path=pair[0], score=pair[1]),
            map(controller.convert_model_to_path, results)
        )
    )
    return KeyframeDisplay(results=display_results)

    



# Deprecated endpoints removed in favor of unified /search






# Deprecated endpoints removed in favor of unified /search

    


