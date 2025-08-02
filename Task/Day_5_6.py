import fitz  # PyMuPDF
import spacy
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


# 🔹 Load spaCy

nlp = spacy.load("en_core_web_sm")


# 📘 Load PDF & Extract Text

def load_pdf(file_path, max_pages=10):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(min(max_pages, len(doc))):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

file_path = r"D:\NT_Final_Task\Task\Doc.pdf"  
text = load_pdf(file_path)
print("📄 First 1000 characters:\n")
print(text[:1000])


# 🔹 1. Fixed-size Chunking

fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fixed_chunks = fixed_splitter.create_documents([text])
print(f"\n📦 Total Fixed Chunks: {len(fixed_chunks)}")


# 🔹 2. Recursive Chunking

recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
recursive_chunks = recursive_splitter.create_documents([text])
print(f"\n📦 Total Recursive Chunks: {len(recursive_chunks)}")


# 🔹 3. Sentence-based Chunking with spaCy

def sentence_chunker(text, max_words=100):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sent in doc.sents:
        words = sent.text.split()
        if current_len + len(words) <= max_words:
            current_chunk += " " + sent.text.strip()
            current_len += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent.text.strip()
            current_len = len(words)

    if current_chunk:
        chunks.append(current_chunk.strip())
    return [Document(page_content=chunk) for chunk in chunks]

sentence_documents = sentence_chunker(text)
print(f"\n📦 Total Sentence-based Chunks: {len(sentence_documents)}")


#  Sentence Embeddings

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# Create VectorStores

vector_db_fixed = Chroma.from_documents(fixed_chunks, embedding, collection_name="fixed_chunks")
vector_db_recursive = Chroma.from_documents(recursive_chunks, embedding, collection_name="recursive_chunks")
vector_db_sentence = Chroma.from_documents(sentence_documents, embedding, collection_name="sentence_chunks")


#  Forecasting User Queries

user_queries = [
    {
        "id": 1,
        "query": "What are the key steps involved in the forecasting process using multiple inputs?",
        "description": "This targets the entire forecasting workflow from problem definition to model deployment."
    },
    {
        "id": 2,
        "query": "How is missing data identified and handled in the forecasting pipeline?",
        "description": "This query helps retrieve the section on missing data types (MCAR, MAR, MNAR), imputation methods (mean, regression, hot/cold deck), and validation."
    },
    {
        "id": 3,
        "query": "Which imputation methods are described, and when should each be used?",
        "description": "This focuses specifically on the explanation of mean imputation, regression imputation, hot deck, cold deck, etc., and their use cases."
    },
    {
        "id": 4,
        "query": "What role does a centralized data warehouse play in the forecasting process?",
        "description": "This pulls in details about storing, cleaning, and integrating historical/internal/external data into a structured forecasting pipeline."
    },
    {
        "id": 5,
        "query": "How does the forecasting model incorporate real-time data updates for rolling forecasts?",
        "description": "This explores how live data influences forecasting accuracy, especially for perishable goods or time-sensitive inventory decisions."
    }
]


# Perform Similarity Search
# -------------------------------
for item in user_queries:
    query = item["query"]
    print("\n" + "="*30)
    print(f"🔍 QUERY {item['id']}: {query}")
    
    print("\n📌 Fixed Chunking Result:")
    result_fixed = vector_db_fixed.similarity_search(query, k=1)
    print(result_fixed[0].page_content.strip()[:500], "\n...")

    print("\n📌 Recursive Chunking Result:")
    result_recursive = vector_db_recursive.similarity_search(query, k=1)
    print(result_recursive[0].page_content.strip()[:500], "\n...")

    print("\n📌 Sentence-based Chunking Result:")
    result_sentence = vector_db_sentence.similarity_search(query, k=1)
    print(result_sentence[0].page_content.strip()[:500], "\n...")



#Day6

print("=========== DAY 6: SentenceTransformer Embeddings & Vector Store ===========")

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np

# === Step 1: Load your chunked docs from Day 5 ===
# Replace with your actual chunked content if available
recursive_chunks = [Document(page_content="Recursive chunked content about forecasting steps.")]
sentence_documents = [Document(page_content="Sentence split content explaining forecasting.")]

# === Step 2: Load embedding model ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# === Step 3: Extract texts (needed to convert to raw embeddings later) ===
recursive_texts = [doc.page_content for doc in recursive_chunks]
sentence_texts = [doc.page_content for doc in sentence_documents]

# === Step 4: Get raw embeddings as arrays ===
recursive_embeddings = embedding_model.embed_documents(recursive_texts)
sentence_embeddings = embedding_model.embed_documents(sentence_texts)

# === Step 5: Create vector stores ===
vectorstore_recursive = Chroma.from_documents(recursive_chunks, embedding_model, collection_name="rec_chunk")
vectorstore_sentence = Chroma.from_documents(sentence_documents, embedding_model, collection_name="sent_chunk")

# === Step 6: Perform test query ===
query = "What are the key steps involved in forecasting?"

print("\n🔍 Query:", query)
res1 = vectorstore_recursive.similarity_search(query, k=1)
print("📌 Recursive Chunk Result:\n", res1[0].page_content)

res2 = vectorstore_sentence.similarity_search(query, k=1)
print("📌 Sentence Chunk Result:\n", res2[0].page_content)

# === Step 7: Print embedding stats ===
recursive_embeddings_np = np.array(recursive_embeddings)
sentence_embeddings_np = np.array(sentence_embeddings)

print("\n--- Embeddings Generation Complete ---")
print(f"Recursive: {recursive_embeddings_np.shape}")
print(f"Sentence: {sentence_embeddings_np.shape}")

print(f"\nRecursive method: {recursive_embeddings_np.shape[0]} embeddings, each with {recursive_embeddings_np.shape[1]} dimensions.")
print(f"Sentence method: {sentence_embeddings_np.shape[0]} embeddings, each with {sentence_embeddings_np.shape[1]} dimensions.")

# Print a few embedding values
print("\nEmbedding example from Recursive:")
print(recursive_embeddings_np[0][:10], "...\n")

print("Embedding example from Sentence-based:")
print(sentence_embeddings_np[0][:10], "...")

# === Step 8: Check if embeddings are identical ===
if np.array_equal(recursive_embeddings_np, sentence_embeddings_np):
    print("\n✓ Both chunking methods produced identical embeddings.")
else:
    print("\n⚠ Different embeddings from different chunking methods.")

