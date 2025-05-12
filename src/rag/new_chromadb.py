import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import chromadb
from rag.document_reader import reader
from rag.enbedding import CustomEmbeddingFunction
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

STORAGE_PATH = "data/chroma_data"
if STORAGE_PATH is None:
    raise ValueError('STORAGE_PATH environment variable is not set')

# Initialisation du client Chroma
chromadb_client = chromadb.PersistentClient(path=STORAGE_PATH)

# Création d'une nouvelle instance de la fonction d'embedding
embedding_function = CustomEmbeddingFunction()

collection = chromadb_client.get_or_create_collection(
    name="documentsTest",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_function
)

documents = reader("data/documents_to_rag")
contents = [doc["content"] for doc in documents]
ids = [doc["id"] for doc in documents]
collection.add(documents=contents, ids=ids)

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path.endswith((".docx", ".txt", ".pdf")):
            print(f"New file detected: {file_path}")
            # Lire et ajouter le nouveau document
            new_documents = reader(os.path.dirname(file_path))
            collection.add(
                documents=[doc["content"] for doc in new_documents],
                ids=[doc["id"] for doc in new_documents]
            )
            print(f"Added new document to the collection: {file_path}")

def monitor_directory(directory):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def rag_pipeline(query: str) -> str:
    res = collection.query(query_texts=query, n_results=3)
    res_doc = res["documents"]
    res_ids = res["ids"]
    res_dist = res["distances"]
    if min(res_dist[0]) < 0.7:
        sorted_idex = np.argsort(res_dist[0])
        closet_idex = sorted_idex[0]
        best_doc: str = res_doc[0][closet_idex]
        best_id: str = res_ids[0][closet_idex]
        if best_doc is not None:
            context = "".join([j for i in best_doc for j in i])
            print(f'Context Found! {len(context)} from {best_id}')
            return context
    else:
        return ""

if __name__ == "__main__":
    monitor_directory("data/documents_to_rag")