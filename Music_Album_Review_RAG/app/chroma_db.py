import chromadb
from chromadb.config import Settings
import json

EMB_PATH = 'processed/embeddings.json'
COLLECTION_NAME = 'music_album_chunks'
CHROMA_PATH = "processed/chroma_db"

_client = None  # keep a global client

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def initialize_db():
    client = get_client()
    coll = client.get_or_create_collection(COLLECTION_NAME)

    # Only add data if collection is empty
    if coll.count() == 0:
        with open(EMB_PATH, encoding='utf-8') as f:
            data = json.load(f)
        ids = [str(i) for i in range(len(data))]
        embeddings = [d['embedding'] for d in data]
        metas = [{"sentence": d['sentence']} for d in data]
        coll.add(ids=ids, embeddings=embeddings, metadatas=metas)
        print("Chroma DB initialized with chunks.")
    else:
        print("Chroma DB already populated.")

    return coll

def query_db(query_embedding, top_k=3):
    client = get_client()
    coll = client.get_collection(COLLECTION_NAME)
    results = coll.query(query_embeddings=[query_embedding], n_results=top_k)
    return [r['sentence'] for r in results['metadatas'][0]]
