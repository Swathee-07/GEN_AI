import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import re
import requests
from rouge_score import rouge_scorer
from collections import Counter
from dotenv import load_dotenv
from bert_score import score

# --- Load .env (Groq Key) ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Groq API Call ---
def groq_generate(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama3-70b-8192",  # Current Groq model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.text}"

# --- Utility Functions ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search(index, chunks, query_embedding, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return " ".join([chunks[i] for i in I[0]])

def split_into_sentence_chunks(text, max_sentences=3):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = ['. '.join(sentences[i:i + max_sentences]) + '.' 
              for i in range(0, len(sentences), max_sentences)]
    return chunks

def calculate_precision_recall_f1(predicted, reference):
    pred_words = set(re.findall(r'\b\w+\b', predicted.lower()))
    ref_words = set(re.findall(r'\b\w+\b', reference.lower()))
    if not ref_words or not pred_words:
        return 0.0, 0.0, 0.0
    intersection = pred_words & ref_words
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(ref_words)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(precision, 3), round(recall, 3), round(f1, 3)

def calculate_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(reference, predicted)['rougeL'].fmeasure
    return round(rouge_score, 3)

def calculate_bleu(predicted, reference):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    bleu_score = overlap / len(pred_tokens) if pred_tokens else 0
    return round(bleu_score, 3)

def calculate_cosine_similarity(predicted, reference):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb1 = embedding_model.encode(predicted, convert_to_tensor=True)
    emb2 = embedding_model.encode(reference, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
    return round(cosine_sim, 3)

def calculate_bert_score(predicted, reference):
    P, R, F1 = score([predicted], [reference], lang="en", verbose=False)
    return round(F1.mean().item(), 3)

# ---------- PROMPTING TECHNIQUES (STRICT CONTEXT) ----------
def chain_of_thought_prompt(context, query):
    return f"""You are a reasoning assistant. 
Use ONLY the provided context to answer. 
If the context does not have the information, say "Not enough information in context." 
Do not use external knowledge. 
Provide reasoning step-by-step and conclude.

Context:
{context}

Question:
{query}

Strict Contextual Answer with reasoning:
"""

def role_based_prompt(context, query):
    return f"""You are a senior forecasting analyst. 
Base your answer ONLY on the given context. 
If the answer is not present in context, respond: "Not enough information in context."

Context:
{context}

Question:
{query}

Domain Expert Answer:
"""

def step_back_prompt(context, query):
    return f"""Take a broader perspective, then narrow down to answer strictly from the context. 
If the context lacks details, respond: "Not enough information in context."

Context:
{context}

Question:
{query}

Step-Back Contextual Answer:
"""

def directional_stimulus_prompt(context, query):
    return f"""Focus strictly on technical insights from the context only. 
Do not use external knowledge. 
If context lacks details, respond: "Not enough information in context."

Context:
{context}

Question:
{query}

Focused Contextual Answer:
"""

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Model ", layout="wide")
st.title(" RAG Model with Advanced Metrics")

uploaded_file = st.file_uploader("📎 Upload your PDF file", type="pdf")

# --- Sample queries and ideal answers ---
sample_query_answers = {
    "What are the traditional forecasting methods and their limitations?":
        "Traditional forecasting methods include regression and ARIMA, but they assume linearity and may fail for complex patterns.",
    "Explain how ARIMA and LSTM differ in predictive analytics applications.":
        "ARIMA is statistical and linear, LSTM is deep learning and handles nonlinear patterns and long-term dependencies better.",
    "How do database management systems enhance forecasting scalability and real-time insights?":
        "DBMS like PostgreSQL or Cassandra allow scalable data storage and fast retrieval, enabling real-time forecasting pipelines.",
    "Describe the role of MongoDB and Cassandra in forecasting workflows.":
        "MongoDB handles flexible document data, Cassandra supports distributed high-speed writes; both improve forecast data flows.",
    "What are hybrid forecasting models and why are they important for modern industries?":
        "Hybrid models combine statistical and machine learning approaches to capture both short-term patterns and long-term trends."
}

query = st.selectbox("Select a sample query:", list(sample_query_answers.keys()))
ideal_answer = sample_query_answers[query]
st.text_area(" Ideal Answer (for evaluation):", ideal_answer, height=150)

chunking_method = st.selectbox("Select Chunking Method", ["Recursive", "Sentence-Based"])
top_k = st.slider("Select Top-K chunks to retrieve", 1, 5, 3)

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF Loaded Successfully!")

    model = SentenceTransformer("all-mpnet-base-v2", device='cpu') 
    query_embedding = model.encode(query)

    if chunking_method == "Recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text)
    else:
        chunks = split_into_sentence_chunks(text, max_sentences=3)

    embeddings = model.encode(chunks)
    index = build_faiss_index(embeddings)
    top_chunks = search(index, chunks, query_embedding, top_k=top_k)

    st.subheader(f"🔍 Retrieved Context (Top {top_k} Chunks Combined)")
    st.write(top_chunks)

    # --- Prompting and Answer Generation ---
    prompts = {
        "Chain-of-Thought": chain_of_thought_prompt(top_chunks, query),
        "Role-Based": role_based_prompt(top_chunks, query),
        "Step-Back": step_back_prompt(top_chunks, query),
        "Directional-Stimulus": directional_stimulus_prompt(top_chunks, query)
    }

    st.subheader(" Prompted Responses and Answers")
    for name, prompt in prompts.items():
        st.markdown(f"### 🔸 {name} Prompt")
        st.code(prompt, language="text")
        answer = groq_generate(prompt)
        st.markdown(f"**Final Answer:** {answer}")

        # --- Metrics ---
        precision, recall, f1 = calculate_precision_recall_f1(answer, ideal_answer)
        rouge = calculate_rouge(answer, ideal_answer)
        bleu = calculate_bleu(answer, ideal_answer)
        cosine_sim = calculate_cosine_similarity(answer, ideal_answer)
        bert_f1 = calculate_bert_score(answer, ideal_answer)

        combined_metric = round((rouge + bert_f1 + cosine_sim) / 3, 3)

        # Display Metrics in row
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("Precision", precision)
        col2.metric("Recall", recall)
        col3.metric("F1 Score", f1)
        col4.metric("ROUGE-L", rouge)
        col5.metric("BLEU", bleu)
        col6.metric("Cosine Sim", cosine_sim)
        col7.metric("Combined Metric", combined_metric)

        st.write(f"**BERTScore (F1):** {bert_f1}")
