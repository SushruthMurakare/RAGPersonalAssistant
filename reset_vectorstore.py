import chromadb
import shutil
import os

# Option 1: Delete the entire vector store directory
persist_directory = "./data/vector_store"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Deleted {persist_directory}")
else:
    print(f"{persist_directory} does not exist")

print("Now run your main.py again to recreate the collection with correct settings")