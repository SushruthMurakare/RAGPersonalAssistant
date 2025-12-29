import numpy as np
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class Embedding :
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded sucessfully")
            print(f"Dimensions of embedding {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e :
            print(f"Loading model failed: {e}")

    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        
        if not self.model:
            raise ValueError("Model not loaded")
        
        return self.model.encode(texts, show_progress_bar=True)
    
    
    

embedding_manager = Embedding()