"""
Embedding data type consistency utilities
Ensures consistent float32 precision across all embedding operations
"""

import numpy as np
from typing import Union, List, Any


def ensure_float32_embedding(embedding: Union[np.ndarray, List[float], Any]) -> np.ndarray:
    """
    Ensure embedding is in float32 format for consistency
    
    Args:
        embedding: Input embedding (numpy array, list, or other format)
        
    Returns:
        numpy array with float32 dtype
    """
    if isinstance(embedding, np.ndarray):
        if embedding.dtype != np.float32:
            return embedding.astype(np.float32)
        return embedding
    elif isinstance(embedding, list):
        return np.array(embedding, dtype=np.float32)
    else:
        # Convert to numpy array and ensure float32
        return np.array(embedding, dtype=np.float32)


def validate_embedding_shape(embedding: np.ndarray, expected_dim: int = 1024) -> bool:
    """
    Validate embedding shape and dimensions
    
    Args:
        embedding: Input embedding array
        expected_dim: Expected embedding dimension
        
    Returns:
        True if shape is valid, False otherwise
    """
    if embedding.ndim == 1:
        return embedding.shape[0] == expected_dim
    elif embedding.ndim == 2:
        return embedding.shape[1] == expected_dim
    else:
        return False


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length for cosine similarity
    
    Args:
        embedding: Input embedding array
        
    Returns:
        Normalized embedding array
    """
    embedding = ensure_float32_embedding(embedding)
    
    # Flatten if 2D
    if embedding.ndim == 2:
        embedding = embedding.flatten()
    
    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    else:
        return embedding


def get_embedding_info(embedding: np.ndarray) -> dict:
    """
    Get information about embedding for debugging
    
    Args:
        embedding: Input embedding array
        
    Returns:
        Dictionary with embedding information
    """
    embedding = ensure_float32_embedding(embedding)
    
    return {
        "shape": embedding.shape,
        "dtype": str(embedding.dtype),
        "min_value": float(embedding.min()),
        "max_value": float(embedding.max()),
        "mean_value": float(embedding.mean()),
        "std_value": float(embedding.std()),
        "norm": float(np.linalg.norm(embedding))
    }


def compare_embedding_dtypes(embedding1: np.ndarray, embedding2: np.ndarray) -> dict:
    """
    Compare data types and properties of two embeddings
    
    Args:
        embedding1: First embedding array
        embedding2: Second embedding array
        
    Returns:
        Dictionary with comparison results
    """
    info1 = get_embedding_info(embedding1)
    info2 = get_embedding_info(embedding2)
    
    return {
        "embedding1": info1,
        "embedding2": info2,
        "dtype_match": info1["dtype"] == info2["dtype"],
        "shape_match": info1["shape"] == info2["shape"],
        "both_float32": info1["dtype"] == "float32" and info2["dtype"] == "float32"
    }


def convert_embedding_batch(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert a batch of embeddings to consistent float32 format
    
    Args:
        embeddings: List of embedding arrays
        
    Returns:
        List of float32 embedding arrays
    """
    converted = []
    for i, embedding in enumerate(embeddings):
        try:
            converted_embedding = ensure_float32_embedding(embedding)
            converted.append(converted_embedding)
        except Exception as e:
            print(f"Warning: Failed to convert embedding {i}: {e}")
            # Return zero embedding as fallback
            converted.append(np.zeros(1024, dtype=np.float32))
    
    return converted


def log_embedding_consistency_check(embedding: np.ndarray, source: str = "unknown") -> None:
    """
    Log embedding consistency information for debugging
    
    Args:
        embedding: Input embedding array
        source: Source identifier for logging
    """
    info = get_embedding_info(embedding)
    
    print(f"[EMBEDDING] {source}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Dtype: {info['dtype']}")
    print(f"  Range: [{info['min_value']:.6f}, {info['max_value']:.6f}]")
    print(f"  Mean: {info['mean_value']:.6f}")
    print(f"  Std: {info['std_value']:.6f}")
    print(f"  Norm: {info['norm']:.6f}")
    
    # Check for potential issues
    if info['dtype'] != 'float32':
        print(f"  ⚠️  WARNING: Non-float32 dtype detected!")
    
    if info['norm'] == 0:
        print(f"  ⚠️  WARNING: Zero norm embedding detected!")
    
    if np.isnan(info['mean_value']) or np.isinf(info['mean_value']):
        print(f"  ⚠️  WARNING: Invalid values (NaN/Inf) detected!")
