
from embedding import Embedding, embedding_manager
from vectorStore import VectorStore, vectorStore
from typing import List, Dict, Any, Tuple

class Retriever :
    
    def __init__(self, vector_store: VectorStore, embedding: Embedding):
        
        self.vector_store = vector_store
        self.embedding = embedding
        
        print("RAG retriever initialized sucessfully")
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        
        query_embedding = self.embedding.generate_embeddings([query])[0]

        try :
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )
            
            #print(results)
            
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "distance": distance,
                            "rank": i+1
                        })
                        
                        #print(f"Retrived {retrieved_docs} after filtering")
                if retrieved_docs:
                    print(f"Retrieved {len(retrieved_docs)} documents")
                    return retrieved_docs
                else:
                    print("No documents found above threshold")
                    return []
                
                    
        except Exception as e :
            print(f"Error while retrieving : {e}")
            return []
        
rag_retriever = Retriever(vectorStore, embedding_manager)
        
        
    