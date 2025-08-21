"""
Competition Factory for HCMC AI Challenge 2025
Creates and configures competition components with optimized settings
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

from agent.competition_agent import CompetitionAgent
from controller.competition_controller import CompetitionController
from router.competition_api import create_competition_router
from service.search_service import KeyframeQueryService
from service.model_service import ModelService
from core.settings import AppSettings
from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting
from factory.factory import ServiceFactory
from models.keyframe import Keyframe
from core.settings import AppSettings


class CompetitionFactory:
    """Factory for creating competition components"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._service_factory: Optional[ServiceFactory] = None
        self._llm: Optional[LLM] = None
        self._keyframe_service: Optional[KeyframeQueryService] = None
        self._model_service: Optional[ModelService] = None
        self._objects_data: Optional[Dict[str, List[str]]] = None
        self._asr_data: Optional[Dict[str, Any]] = None
        
    def create_llm(
        self, 
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> LLM:
        """Create optimized LLM for competition tasks"""
        
        if self._llm is None:
            # Competition-optimized settings
            self._llm = OpenAI(
                model=model_name,
                temperature=temperature,  # Low temperature for consistent results
                max_tokens=max_tokens,
                timeout=30.0,  # Reasonable timeout for competition
                max_retries=3  # Retry failed requests
            )
        
        return self._llm
    
    def create_keyframe_service(self) -> KeyframeQueryService:
        """Create keyframe query service with optimized settings"""
        
        if self._keyframe_service is None:
            # Initialize via the shared ServiceFactory to ensure proper dependencies
            service_factory = self._get_or_create_service_factory()
            self._keyframe_service = service_factory.get_keyframe_query_service()
            
            # Apply performance optimizations
            self._optimize_keyframe_service()
        
        return self._keyframe_service
    
    def create_model_service(self) -> ModelService:
        """Create model service with embedding capabilities"""
        
        if self._model_service is None:
            # Use ServiceFactory to construct a properly initialized ModelService
            service_factory = self._get_or_create_service_factory()
            self._model_service = service_factory.get_model_service()
            
            # Apply model optimizations
            self._optimize_model_service()
        
        return self._model_service
    
    def load_objects_data(
        self, 
        objects_file_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Load object detection data with caching"""
        
        if self._objects_data is None:
            objects_path = objects_file_path or self._get_default_objects_path()
            
            try:
                if os.path.exists(objects_path):
                    with open(objects_path, 'r') as f:
                        self._objects_data = json.load(f)
                    print(f"Loaded {len(self._objects_data)} object detection entries")
                else:
                    print(f"Objects file not found at {objects_path}, using empty data")
                    self._objects_data = {}
                    
            except Exception as e:
                print(f"Error loading objects data: {e}")
                self._objects_data = {}
        
        return self._objects_data
    
    def load_asr_data(
        self, 
        asr_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load ASR data with caching"""
        
        if self._asr_data is None:
            asr_path = asr_file_path or self._get_default_asr_path()
            
            try:
                if os.path.exists(asr_path):
                    with open(asr_path, 'r') as f:
                        self._asr_data = json.load(f)
                    print(f"Loaded ASR data with {len(self._asr_data)} entries")
                else:
                    print(f"ASR file not found at {asr_path}, using empty data")
                    self._asr_data = {}
                    
            except Exception as e:
                print(f"Error loading ASR data: {e}")
                self._asr_data = {}
        
        return self._asr_data
    
    def create_agent(
        self,
        data_folder: Optional[str] = None,
        video_metadata_path: Optional[str] = None,
        objects_file_path: Optional[str] = None,
        asr_file_path: Optional[str] = None
    ) -> CompetitionAgent:
        """Create competition agent with all components"""
        
        # Initialize core services
        llm = self.create_llm()
        keyframe_service = self.create_keyframe_service()
        model_service = self.create_model_service()
        
        # Load data
        objects_data = self.load_objects_data(objects_file_path)
        asr_data = self.load_asr_data(asr_file_path)
        
        # Resolve paths
        data_folder = data_folder or self._get_default_data_folder()
        video_metadata_path = Path(video_metadata_path) if video_metadata_path else None
        
        # Create agent
        agent = CompetitionAgent(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            video_metadata_path=video_metadata_path
        )
        
        return agent
    
    def create_controller(
        self,
        data_folder: Optional[str] = None,
        video_metadata_path: Optional[str] = None,
        objects_file_path: Optional[str] = None,
        asr_file_path: Optional[str] = None
    ) -> CompetitionController:
        """Create competition controller"""
        
        # Initialize core services
        llm = self.create_llm()
        keyframe_service = self.create_keyframe_service()
        model_service = self.create_model_service()
        
        # Load data
        objects_data = self.load_objects_data(objects_file_path)
        asr_data = self.load_asr_data(asr_file_path)
        
        # Resolve paths
        data_folder = data_folder or self._get_default_data_folder()
        video_metadata_path = Path(video_metadata_path) if video_metadata_path else None
        
        # Create controller
        controller = CompetitionController(
            llm=llm,
            keyframe_service=keyframe_service,
            model_service=model_service,
            data_folder=data_folder,
            objects_data=objects_data,
            asr_data=asr_data,
            video_metadata_path=video_metadata_path
        )
        
        return controller
    
    def create_api_router(
        self,
        data_folder: Optional[str] = None,
        video_metadata_path: Optional[str] = None,
        objects_file_path: Optional[str] = None,
        asr_file_path: Optional[str] = None
    ):
        """Create API router with all components"""
        
        # Create controller
        controller = self.create_controller(
            data_folder=data_folder,
            video_metadata_path=video_metadata_path,
            objects_file_path=objects_file_path,
            asr_file_path=asr_file_path
        )
        
        # Create router
        router = create_competition_router(controller)
        
        return router, controller
    
    def create_full_competition_system(
        self,
        data_folder: Optional[str] = None,
        video_metadata_path: Optional[str] = None,
        objects_file_path: Optional[str] = None,
        asr_file_path: Optional[str] = None,
        optimization_profile: str = "balanced"
    ) -> Dict[str, Any]:
        """Create complete competition system with all components"""
        
        # Apply optimization profile
        self._apply_optimization_profile(optimization_profile)
        
        # Create all components
        agent = self.create_agent(
            data_folder=data_folder,
            video_metadata_path=video_metadata_path,
            objects_file_path=objects_file_path,
            asr_file_path=asr_file_path
        )
        
        controller = self.create_controller(
            data_folder=data_folder,
            video_metadata_path=video_metadata_path,
            objects_file_path=objects_file_path,
            asr_file_path=asr_file_path
        )
        
        router = create_competition_router(controller)
        
        # Create system configuration
        system_config = {
            "optimization_profile": optimization_profile,
            "data_paths": {
                "data_folder": data_folder or self._get_default_data_folder(),
                "video_metadata": video_metadata_path,
                "objects_file": objects_file_path or self._get_default_objects_path(),
                "asr_file": asr_file_path or self._get_default_asr_path()
            },
            "performance_settings": self._get_current_performance_settings(),
            "available_tasks": [
                "vcmr_automatic", "vcmr_interactive",
                "video_qa", 
                "kis_textual", "kis_visual", "kis_progressive"
            ]
        }
        
        return {
            "agent": agent,
            "controller": controller,
            "router": router,
            "config": system_config,
            "services": {
                "llm": self._llm,
                "keyframe_service": self._keyframe_service,
                "model_service": self._model_service
            },
            "data": {
                "objects_data": self._objects_data,
                "asr_data": self._asr_data
            }
        }
    
    # Private helper methods
    def _get_or_create_service_factory(self) -> ServiceFactory:
        """Create or return a cached ServiceFactory configured from settings"""
        if self._service_factory is not None:
            return self._service_factory

        milvus_settings = KeyFrameIndexMilvusSetting()

        milvus_search_params = {
            "metric_type": milvus_settings.METRIC_TYPE,
            "params": milvus_settings.SEARCH_PARAMS
        }

        self._service_factory = ServiceFactory(
            milvus_collection_name=milvus_settings.COLLECTION_NAME,
            milvus_host=milvus_settings.HOST,
            milvus_port=milvus_settings.PORT,
            milvus_user="",
            milvus_password="",
            milvus_search_params=milvus_search_params,
            model_name=self.settings.MODEL_NAME,
            mongo_collection=Keyframe
        )

        return self._service_factory
    
    def _optimize_keyframe_service(self):
        """Apply performance optimizations to keyframe service"""
        # This would configure service-specific optimizations
        # For now, we assume the service handles its own optimization
        pass
    
    def _optimize_model_service(self):
        """Apply performance optimizations to model service"""
        # This would configure model-specific optimizations
        # For now, we assume the service handles its own optimization
        pass
    
    def _apply_optimization_profile(self, profile: str):
        """Apply system-wide optimization profile"""
        
        profiles = {
            "speed": {
                "llm_temperature": 0.0,
                "llm_max_tokens": 1024,
                "search_top_k": 50,
                "enable_reranking": False
            },
            "balanced": {
                "llm_temperature": 0.1,
                "llm_max_tokens": 2048,
                "search_top_k": 100,
                "enable_reranking": True
            },
            "precision": {
                "llm_temperature": 0.05,
                "llm_max_tokens": 4096,
                "search_top_k": 200,
                "enable_reranking": True
            }
        }
        
        if profile in profiles:
            settings = profiles[profile]
            
            # Apply LLM settings
            if self._llm is None:
                self._llm = self.create_llm(
                    temperature=settings["llm_temperature"],
                    max_tokens=settings["llm_max_tokens"]
                )
            
            # Store other settings for later use
            self._performance_settings = settings
        else:
            print(f"Unknown optimization profile: {profile}, using 'balanced'")
            self._apply_optimization_profile("balanced")
    
    def _get_current_performance_settings(self) -> Dict[str, Any]:
        """Get current performance settings"""
        return getattr(self, '_performance_settings', {
            "llm_temperature": 0.1,
            "llm_max_tokens": 2048,
            "search_top_k": 100,
            "enable_reranking": True
        })
    
    def _get_default_data_folder(self) -> str:
        """Get default data folder path"""
        # This should be configured based on your data structure
        return self.settings.DATA_FOLDER if hasattr(self.settings, 'DATA_FOLDER') else "/data/keyframes"
    
    def _get_default_objects_path(self) -> str:
        """Get default objects file path"""
        return self.settings.OBJECTS_FILE if hasattr(self.settings, 'OBJECTS_FILE') else "/data/objects.json"
    
    def _get_default_asr_path(self) -> str:
        """Get default ASR file path"""
        return self.settings.ASR_FILE if hasattr(self.settings, 'ASR_FILE') else "/data/asr.json"


# Utility functions for creating pre-configured systems

def create_competition_system_for_evaluation(
    data_folder: str,
    objects_file: str,
    asr_file: str,
    video_metadata_file: Optional[str] = None,
    optimization_profile: str = "precision"
) -> Dict[str, Any]:
    """Create competition system optimized for evaluation/testing"""
    
    from core.settings import AppSettings
    
    settings = AppSettings()
    factory = CompetitionFactory(settings)
    
    return factory.create_full_competition_system(
        data_folder=data_folder,
        video_metadata_path=video_metadata_file,
        objects_file_path=objects_file,
        asr_file_path=asr_file,
        optimization_profile=optimization_profile
    )


def create_competition_system_for_interactive(
    data_folder: str,
    objects_file: str,
    asr_file: str,
    video_metadata_file: Optional[str] = None
) -> Dict[str, Any]:
    """Create competition system optimized for interactive tasks"""
    
    return create_competition_system_for_evaluation(
        data_folder=data_folder,
        objects_file=objects_file,
        asr_file=asr_file,
        video_metadata_file=video_metadata_file,
        optimization_profile="speed"  # Prioritize speed for interactive tasks
    )


def create_competition_system_for_automatic(
    data_folder: str,
    objects_file: str,
    asr_file: str,
    video_metadata_file: Optional[str] = None
) -> Dict[str, Any]:
    """Create competition system optimized for automatic tasks"""
    
    return create_competition_system_for_evaluation(
        data_folder=data_folder,
        objects_file=objects_file,
        asr_file=asr_file,
        video_metadata_file=video_metadata_file,
        optimization_profile="precision"  # Prioritize precision for automatic tasks
    )


# Configuration validation

def validate_competition_setup(
    data_folder: str,
    objects_file: str,
    asr_file: str,
    video_metadata_file: Optional[str] = None
) -> Dict[str, Any]:
    """Validate competition system setup and data availability"""
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "data_stats": {}
    }
    
    # Check data folder (use provided path, not class attribute)
    if not data_folder:
        validation_result["valid"] = False
        validation_result["errors"].append("Data folder path is empty")
        validation_result["data_stats"]["keyframe_count"] = 0
    elif not os.path.isdir(data_folder):
        # Auto-create missing data folder and warn instead of failing hard
        try:
            os.makedirs(data_folder, exist_ok=True)
            validation_result["warnings"].append(f"Data folder did not exist and was created: {data_folder}")
            validation_result["data_stats"]["keyframe_count"] = 0
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Failed to create data folder '{data_folder}': {e}")
            validation_result["data_stats"]["keyframe_count"] = 0
    else:
        # Count keyframe files (.webp for L01-L20, .jpg/.jpeg common for L21-L30)
        keyframe_count = 0
        allowed_exts = {'.webp', '.jpg', '.jpeg'}
        for root, dirs, files in os.walk(data_folder):
            keyframe_count += sum(
                1 for f in files if os.path.splitext(f)[1].lower() in allowed_exts
            )
        validation_result["data_stats"]["keyframe_count"] = keyframe_count
    
    # Check objects file
    if not os.path.exists(objects_file):
        validation_result["warnings"].append(f"Objects file not found: {objects_file}")
        validation_result["data_stats"]["objects_available"] = False
    else:
        try:
            with open(objects_file, 'r') as f:
                objects_data = json.load(f)
            validation_result["data_stats"]["objects_count"] = len(objects_data)
            validation_result["data_stats"]["objects_available"] = True
        except Exception as e:
            validation_result["warnings"].append(f"Error reading objects file: {e}")
            validation_result["data_stats"]["objects_available"] = False
    
    # Check ASR file
    if not os.path.exists(asr_file):
        validation_result["warnings"].append(f"ASR file not found: {asr_file}")
        validation_result["data_stats"]["asr_available"] = False
    else:
        try:
            with open(asr_file, 'r') as f:
                asr_data = json.load(f)
            validation_result["data_stats"]["asr_entries"] = len(asr_data)
            validation_result["data_stats"]["asr_available"] = True
        except Exception as e:
            validation_result["warnings"].append(f"Error reading ASR file: {e}")
            validation_result["data_stats"]["asr_available"] = False
    
    # Check video metadata file
    if video_metadata_file:
        if not os.path.exists(video_metadata_file):
            validation_result["warnings"].append(f"Video metadata file not found: {video_metadata_file}")
            validation_result["data_stats"]["metadata_available"] = False
        else:
            validation_result["data_stats"]["metadata_available"] = True
    else:
        validation_result["data_stats"]["metadata_available"] = False
    
    return validation_result
