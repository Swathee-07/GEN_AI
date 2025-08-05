from sentence_transformers import SentenceTransformer
import json

CHUNKS_PATH = 'processed/chunks.txt'
OUTPUT_JSON = 'processed/embeddings.json'

def embed_chunks():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with open(CHUNKS_PATH, encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    embeddings = model.encode(sentences, show_progress_bar=True)
    to_save = [{'sentence': s, 'embedding': emb.tolist()} for s, emb in zip(sentences, embeddings)]
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as out:
        json.dump(to_save, out, indent=2)
    print("Embeddings saved in:", OUTPUT_JSON)

if __name__ == "__main__":
    embed_chunks()
