import os
import chromadb
from chromadb.config import Settings
import numpy as np
import uuid
from typing import List, Dict, Any, Tuple


class VectorStore:
    
    def __init__(self, collection_name: str = "documents", persist_directory: str= "./data/vector_store"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            print("vector store initialized successfully")
            
        except Exception as e :
            print(f"Error while initialzing the vector store: {e}")
            
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        
        if len(documents) != len(embeddings):
            raise ValueError("Document length and Embeddings length are not equal")
        
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []
        
        
        
        for i , (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata["doc_index"] = doc_id
            metadata["doc_content_len"] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_texts.append(doc.page_content)
            
            embeddings_list.append(embedding.tolist())
            
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_texts
            )
            
                
        except Exception as e:
            print("Error adding documents")
         
        
vectorStore = VectorStore()

    
     
    
    
    