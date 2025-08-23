import torch
import numpy as np


class ModelService:
    def __init__(
        self,
        model ,
        preprocess ,
        tokenizer ,
        device: str='cuda'
        ):
        # Select device with graceful fallback when CUDA is unavailable
        selected_device = device
        if device == 'cuda' and not torch.cuda.is_available():
            selected_device = 'cpu'

        self.model = model.to(selected_device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = selected_device
        self.model.eval()
    
    def embedding(self, query_text: str) -> np.ndarray:
        """
        Generate text embedding with consistent data type
        
        Args:
            query_text: Input text to embed
            
        Returns:
            numpy array of shape (1, 1024) with consistent dtype
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([query_text]).to(self.device)
            query_embedding = self.model.encode_text(text_tokens).cpu().detach().numpy()
            
            # Ensure consistent data type (float32 for maximum precision)
            query_embedding = query_embedding.astype(np.float32)
            
        return query_embedding
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Generate image embedding with consistent data type
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of shape (1024,) with consistent dtype
        """
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Generate image embedding
                image_embedding = self.model.encode_image(image_tensor).cpu().detach().numpy()
                
                # Ensure consistent data type (float32 for maximum precision)
                image_embedding = image_embedding.astype(np.float32)
                
                # Return as 1D array (1024,)
                return image_embedding.flatten()
                
        except Exception as e:
            print(f"Error embedding image {image_path}: {e}")
            # Return zero embedding as fallback
            return np.zeros(1024, dtype=np.float32)
    
    def get_embedding_dtype(self) -> np.dtype:
        """
        Get the consistent data type used for embeddings
        
        Returns:
            numpy dtype (np.float32 for maximum precision)
        """
        return np.float32

            