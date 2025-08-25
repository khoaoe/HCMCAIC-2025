
from pathlib import Path
from fastapi import Depends, Request, HTTPException
from functools import lru_cache
import json

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)


from controller.query_controller import QueryController
from service import ModelService, KeyframeQueryService
from core.settings import KeyFrameIndexMilvusSetting, MongoDBSettings, AppSettings
from factory.factory import ServiceFactory
from controller.competition_controller import CompetitionController
from core.logger import SimpleLogger

from llama_index.llms.google_genai import GoogleGenAI
from controller.agent_controller import AgentController
from llama_index.core.llms import LLM

logger = SimpleLogger(__name__)

from dotenv import load_dotenv
load_dotenv()



@lru_cache
def get_llm(model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.0, max_tokens: int = 1000, timeout: float = 30.0, max_retries: int = 3) -> LLM:
    try:
        print("🤖 Getting LLM for Agent...") 
        return GoogleGenAI(
            model_name,
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )
    except Exception as e:
        logger.error(f"Failed to get LLM: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM initialization failed: {str(e)}"
        )




@lru_cache()
def get_app_settings():
    """Get MongoDB settings (cached)"""
    return AppSettings()


@lru_cache()
def get_milvus_settings():
    """Get Milvus settings (cached)"""
    return KeyFrameIndexMilvusSetting()


@lru_cache()
def get_mongo_settings():
    """Get MongoDB settings (cached)"""
    return MongoDBSettings()



def get_service_factory(request: Request) -> ServiceFactory:
    """Get ServiceFactory from app state"""
    service_factory = getattr(request.app.state, 'service_factory', None)
    if service_factory is None:
        logger.error("ServiceFactory not found in app state")
        raise HTTPException(
            status_code=503, 
            detail="Service factory not initialized. Please check application startup."
        )
    return service_factory



def get_agent_controller(
    service_factory = Depends(get_service_factory),
    app_settings: AppSettings = Depends(get_app_settings),
    request: Request = None
) -> AgentController:
    llm = get_llm()
    keyframe_service = service_factory.get_keyframe_query_service()
    model_service = service_factory.get_model_service()

    data_folder = app_settings.DATA_FOLDER
    
    # Get pre-loaded contextual data from app state
    objects_data = getattr(request.app.state, 'objects_data', {})
    asr_data = getattr(request.app.state, 'asr_data', {})

    return AgentController(
        llm=llm,
        keyframe_service=keyframe_service,
        model_service=model_service,
        data_folder=data_folder,
        objects_data=objects_data,
        asr_data=asr_data,
        top_k=50
    )




def get_model_service(service_factory: ServiceFactory = Depends(get_service_factory)) -> ModelService:
    try:
        model_service = service_factory.get_model_service()
        if model_service is None:
            logger.error("Model service not available from factory")
            raise HTTPException(
                status_code=503,
                detail="Model service not available"
            )
        return model_service
    except Exception as e:
        logger.error(f"Failed to get model service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Model service initialization failed: {str(e)}"
        )
    

def get_keyframe_service(service_factory: ServiceFactory = Depends(get_service_factory)) -> KeyframeQueryService:
    """Get keyframe query service from ServiceFactory"""
    try:
        keyframe_service = service_factory.get_keyframe_query_service()
        if keyframe_service is None:
            logger.error("Keyframe service not available from factory")
            raise HTTPException(
                status_code=503,
                detail="Keyframe service not available"
            )
        return keyframe_service
    except Exception as e:
        logger.error(f"Failed to get keyframe service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Keyframe service initialization failed: {str(e)}"
        )



def get_mongo_client(request: Request):
    """Get MongoDB client from app state"""
    mongo_client = getattr(request.app.state, 'mongo_client', None)
    if mongo_client is None:
        logger.error("MongoDB client not found in app state")
        raise HTTPException(
            status_code=503,
            detail="MongoDB client not initialized"
        )
    return mongo_client

async def check_mongodb_health(request: Request) -> bool:
    """Check MongoDB connection health"""
    try:
        mongo_client = get_mongo_client(request)
        await mongo_client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {str(e)}")
        return False



def get_milvus_repository(service_factory: ServiceFactory = Depends(get_service_factory)):
    """Get Milvus repository from ServiceFactory"""
    try:
        repository = service_factory.get_milvus_keyframe_repo()
        if repository is None:
            raise HTTPException(
                status_code=503,
                detail="Milvus repository not available"
            )
        return repository
    except Exception as e:
        logger.error(f"Failed to get Milvus repository: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Milvus repository initialization failed: {str(e)}"
        )


def get_query_controller(
    model_service: ModelService = Depends(get_model_service),
    keyframe_service: KeyframeQueryService = Depends(get_keyframe_service),
    app_settings: AppSettings = Depends(get_app_settings),
    request: Request = None
) -> QueryController:
    """Get query controller instance"""
    try:
        logger.info("Creating query controller...")
        
        data_folder = Path(app_settings.DATA_FOLDER)
        id2index_path = Path(app_settings.ID2INDEX_PATH)
        
        if not data_folder.exists():
            logger.warning(f"Data folder does not exist: {data_folder}")
            data_folder.mkdir(parents=True, exist_ok=True)
            
        if not id2index_path.exists():
            logger.warning(f"ID2Index file does not exist: {id2index_path}")
            id2index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(id2index_path, 'w') as f:
                json.dump({}, f)
        
        # Get pre-loaded objects data from app state
        objects_data = getattr(request.app.state, 'objects_data', {})
        
        controller = QueryController(
            data_folder=data_folder,
            id2index_path=id2index_path,
            model_service=model_service,
            keyframe_service=keyframe_service,
            llm=get_llm(),
            objects_data=objects_data,
            settings=app_settings
        )
        
        logger.info("Query controller created successfully")
        return controller
        
    except Exception as e:
        logger.error(f"Failed to create query controller: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Query controller initialization failed: {str(e)}"
        )


def get_competition_controller(
    service_factory = Depends(get_service_factory),
    app_settings: AppSettings = Depends(get_app_settings),
    request: Request = None
) -> CompetitionController:
    """Get competition controller instance for HCMC AI Challenge 2025 tasks"""
    try:
        llm = get_llm()
        keyframe_service = service_factory.get_keyframe_query_service()
        model_service = service_factory.get_model_service()

        data_folder = app_settings.DATA_FOLDER
        
        # Get pre-loaded contextual data from app state
        objects_data = getattr(request.app.state, 'objects_data', {})
        asr_data = getattr(request.app.state, 'asr_data', {})
        
        # Optional video metadata path for temporal mapping
        video_metadata_path = Path(app_settings.VIDEO_METADATA_PATH) if hasattr(app_settings, 'VIDEO_METADATA_PATH') else None

        return CompetitionController(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            video_metadata_path=video_metadata_path
        )
        
    except Exception as e:
        logger.error(f"Failed to create competition controller: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Competition controller initialization failed: {str(e)}"
        )