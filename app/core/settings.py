from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from pathlib import Path

# Ensure we always load the repository-root .env regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / '.env'
load_dotenv(dotenv_path=ENV_PATH)


class MongoDBSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding='utf-8', case_sensitive=False, extra='ignore'
    )
    MONGO_URI: str = Field(..., alias='MONGO_URI')
    MONGO_HOST: str = Field(..., alias='MONGO_HOST')
    MONGO_PORT: int = Field(..., alias='MONGO_PORT')
    MONGO_DB: str = Field(..., alias='MONGO_DB')
    MONGO_USER: str = Field(..., alias='MONGO_USER')
    MONGO_PASSWORD: str = Field(..., alias='MONGO_PASSWORD')


class IndexPathSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding='utf-8', case_sensitive=False, extra='ignore'
    )
    FAISS_INDEX_PATH: str | None  
    USEARCH_INDEX_PATH: str | None

class KeyFrameIndexMilvusSetting(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding='utf-8', case_sensitive=False, extra='ignore'
    )
    COLLECTION_NAME: str = "keyframe_embeddings"
    HOST: str = 'localhost'
    PORT: str = '19530'
    METRIC_TYPE: str = 'COSINE'
    INDEX_TYPE: str = 'FLAT'
    BATCH_SIZE: int =10000
    SEARCH_PARAMS: dict = {}
    
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding='utf-8', case_sensitive=False, extra='ignore'
    )
    # model configuration
    MODEL_NAME: str = "ViT-B-32"
    USE_PRETRAINED: bool = True  # Whether to use pretrained weights 
    PRETRAINED_WEIGHTS: str = "openai"  #  pretrained weights (only used if USE_PRETRAINED=True)
    # data folder
    DATA_FOLDER: str  = str(REPO_ROOT / 'resources' / 'keyframes')
    KEYFRAMES_FOLDER: str = str(REPO_ROOT / 'resources' / 'keyframes')
    ID2INDEX_PATH: str = str(REPO_ROOT / 'resources' / 'keyframes' / 'id2index.json')
    # FRAME2OBJECT: str = str(REPO_ROOT / 'resources' / 'detections.json')
    # ASR_FILE: str = str(REPO_ROOT / 'resources' / 'asr_proc.json')
    OBJECTS_FOLDER: str = str(REPO_ROOT / 'resources' / 'objects')
    VIDEO_METADATA_FOLDER: str = str(REPO_ROOT / 'resources' / 'metadata')
    OPTIMIZATION_PROFILE: str = 'balanced'


    # Defaults and algorithm parameters
    DEFAULT_FPS: float = 25.0

    # GRAB/ABTS Algorithm Parameters
    ABTS_LAMBDA_S: float = 0.7
    ABTS_LAMBDA_T: float = 0.3
    ABTS_SEARCH_WINDOW: int = 75
    ABTS_CONFIDENCE_THRESHOLD: float = 0.3

    # LLM Configuration
    LLM_MODEL_NAME: str = "gemini-2.5-flash-lite"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT: float = 30.0
