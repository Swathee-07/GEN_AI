import chromadb
from chromadb.config import Settings
import json
import numpy as np

EMB_PATH = 'processed/embeddings.json'
COLLECTION_NAME = 'music_album_chunks'

def initialize_db():
    client = chromadb.PersistentClient(path="processed/chroma_db", settings=Settings())
    coll = client.get_or_create_collection(COLLECTION_NAME)
    with open(EMB_PATH, encoding='utf-8') as f:
        data = json.load(f)
    ids = [str(i) for i in range(len(data))]
    embeddings = [d['embedding'] for d in data]
    metas = [{"sentence": d['sentence']} for d in data]
    coll.add(ids=ids, embeddings=embeddings, metadatas=metas)
    print("Chroma DB initialized with chunks.")

def query_db(query_embedding, top_k=3):
    client = chromadb.PersistentClient(path="processed/chroma_db", settings=Settings())
    coll = client.get_collection(COLLECTION_NAME)
    results = coll.query(query_embeddings=[query_embedding], n_results=top_k)
    return [r['sentence'] for r in results['metadatas'][0]]

# Only run initialization once
if __name__ == "__main__":
    initialize_db()
