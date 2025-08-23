import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)

import logging
import torch

from repository.mongo import KeyframeRepository
from repository.milvus import KeyframeVectorRepository
from service import KeyframeQueryService, ModelService
from models.keyframe import Keyframe
import open_clip
from pymilvus import connections, Collection as MilvusCollection

logger = logging.getLogger(__name__)


class ServiceFactory:
    def __init__(
        self,
        milvus_collection_name: str,
        milvus_host: str,
        milvus_port: str ,
        milvus_user: str ,
        milvus_password: str ,
        milvus_search_params: dict,
        model_name: str ,
        pretrained_weights: str,
        use_pretrained: bool = True,
        milvus_db_name: str = "default",
        milvus_alias: str = "default",
        mongo_collection=Keyframe
    ):
        self._mongo_keyframe_repo = KeyframeRepository(collection=mongo_collection)
        self._milvus_keyframe_repo = self._init_milvus_repo(
            search_params=milvus_search_params,
            collection_name=milvus_collection_name,
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
            db_name=milvus_db_name,
            alias=milvus_alias
        )

        self._model_service = self._init_model_service(model_name, pretrained_weights, use_pretrained)

        self._keyframe_query_service = KeyframeQueryService(
            keyframe_mongo_repo=self._mongo_keyframe_repo,
            keyframe_vector_repo=self._milvus_keyframe_repo
        )

    def _init_milvus_repo(
        self,
        search_params: dict,
        collection_name: str,
        host: str,
        port: str,
        user: str,
        password: str,
        db_name: str = "default",
        alias: str = "default"
    ):
        if connections.has_connection(alias):
            connections.remove_connection(alias)

        conn_params = {
            "host": host,
            "port": port,
            "db_name": db_name
        }

        if user and password:
            conn_params["user"] = user
            conn_params["password"] = password

        connections.connect(alias=alias, **conn_params)
        collection = MilvusCollection(collection_name, using=alias)

        return KeyframeVectorRepository(collection=collection, search_params=search_params)

    def _init_model_service(self, model_name: str, pretrained_weights: str, use_pretrained: bool = True):
        # Load model with or without pretrained weights based on configuration
        try:
            if use_pretrained:
                logger.info(f"Initializing model: {model_name} with pretrained weights: {pretrained_weights}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained_weights
                )
            else:
                logger.info(f"Initializing model: {model_name} without pretrained weights")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=None
                )
            
            tokenizer = open_clip.get_tokenizer(model_name)
            
            # Validate model initialization
            if model is None:
                raise ValueError(f"Model {model_name} failed to initialize")
            
            # Test model with a simple input to ensure it's working
            test_text = ["test"]
            test_tokens = tokenizer(test_text)
            with torch.no_grad():
                test_embedding = model.encode_text(test_tokens)
            
            logger.info(f"Model {model_name} initialized successfully with shape: {test_embedding.shape}")
            return ModelService(model=model, preprocess=preprocess, tokenizer=tokenizer)
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name} with use_pretrained={use_pretrained}: {e}")
            # Try fallback model if available
            if model_name != "ViT-B-32":
                logger.info(f"Attempting fallback to ViT-B-32 model")
                return self._init_model_service("ViT-B-32", pretrained_weights, use_pretrained)
            else:
                raise

    def get_mongo_keyframe_repo(self):
        return self._mongo_keyframe_repo

    def get_milvus_keyframe_repo(self):
        return self._milvus_keyframe_repo

    def get_model_service(self):
        return self._model_service

    def get_keyframe_query_service(self):
        return self._keyframe_query_service
