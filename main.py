import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from pathlib import Path


from langchain_community.document_loaders import PyMuPDFLoader , DirectoryLoader, CSVLoader, TextLoader, JSONLoader

from embedding import embedding_manager

from vectorStore import vectorStore

from llm_integration import rag_llm
from retriever import rag_retriever

from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager 
from fastapi.middleware.cors import CORSMiddleware



# Load all PDF files from the specified directory and its subdirectories


def process_all_documents(directory_path):
    all_documents = []
    folder_path = Path(directory_path)
    
    pdf_files = list(folder_path.glob("**/*.pdf"))
    csv_files = list(folder_path.glob("**/*.csv"))
    txt_files = list(folder_path.glob("**/*.txt"))
    json_files = list(folder_path.glob("**/*.json"))
    
    files = pdf_files + csv_files + txt_files + json_files
    
    
    
    for file in files:
        try:
            if file.suffix == ".pdf":
                loader = PyMuPDFLoader(str(file))
            elif file.suffix == ".csv":
                loader = CSVLoader(file_path=str(file))
            elif file.suffix == ".txt":
                loader = TextLoader(file_path=str(file))
            elif file.suffix == ".json":
                loader = JSONLoader(file_path=str(file), jq_schema=".")  
            else:
                continue
            
            documents = loader.load()

            
            
            for doc in documents:
                doc.metadata["source_file"] = str(file)
                doc.metadata["file_type"] = file.suffix

                
            all_documents.extend(documents)
            
        except Exception as e :
            print(f"Error loading {file}: {e}")
    return all_documents
    
def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"Document : {len(documents)} are Split into {len(split_documents)} chunks.")
    
    return split_documents

@asynccontextmanager 
async def lifespan(app: FastAPI):
    
    all_documents = process_all_documents("./data")
    chunks = split_documents_into_chunks(all_documents)
    texts  = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    
    vectorStore.add_documents(chunks, embeddings)
    yield
    
    # question = input("Enter your question: ")
    
    # while question != "DONE":
    #     answer = rag_llm(question, rag_retriever, top_k=3)
    #     print(answer)
    #     question = input("Enter your question: ")

app = FastAPI(title="RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(question: str = Body(..., embed=True)):
    try:
        answer = rag_llm(question, rag_retriever, top_k=3)
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    return {"message": "RAG API is running. Use POST /ask to ask questions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)