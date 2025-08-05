import os
import time
import random
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from .chroma_db import query_db

# ── ENV ──────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "llama-3.1-8b-instant"          # Groq-supported model

# ── CACHED MODELS ───────────────────────────────────────────────
@st.cache_resource
def get_embedding_model():
    print("Loading embedding model …")
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_reranker():
    print("Loading reranker …")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── MAIN RAG FUNCTION ───────────────────────────────────────────
def rag_answer(user_query, return_context=False,
               prompt_mode="Direct Answering (Standard RAG)"):
    """
    Answer a question with single-query retrieval + Cross-Encoder re-ranking.
    """
    context_chunks, answer = [], "Error: Could not process query."

    try:
        # Simple client-side rate-limit (≥1.5 s between API calls)
        if hasattr(st.session_state, "last_api_call"):
            wait = 1.5 - (time.time() - st.session_state.last_api_call)
            if wait > 0:
                time.sleep(wait)

        # ── 1) RETRIEVE (single query) ─────────────────────────
        embedder  = get_embedding_model()
        query_emb = embedder.encode([user_query])[0]
        raw_chunks = query_db(query_emb, top_k=25)          # recall ↑

        # ── 2) RE-RANK  → keep best 12 chunks ─────────────────
        reranker = get_reranker()
        scores   = reranker.predict([[user_query, ch] for ch in raw_chunks])
        context_chunks = [
            ch for _, ch in sorted(zip(scores, raw_chunks), reverse=True)
        ][:12]

        # DEBUG
        print(f"\n=== DEBUG: Query: {user_query} ===")
        print(f"Retrieved {len(raw_chunks)} chunks; top 3 after re-rank:")
        for i, ch in enumerate(context_chunks[:3]):
            print(f"Ranked {i+1}: {ch[:200]}…")

        # ── 3) PROMPT MODES ───────────────────────────────────
        if prompt_mode == "Role-Based Answering (Advanced)":
            prompt = f"""
You are a music expert. Return the sentence that directly answers the question.
If absent, reply exactly: "No answer found in dataset."

TEXT:
{"\n\n".join(context_chunks)}

QUESTION: {user_query}

EXACT SENTENCE:
""".strip()
        else:  # Direct / Standard
            prompt = f"""
Answer with the exact sentence from the text below that answers the question.
If it is missing, say: "No answer found in dataset."

TEXT:
{"\n\n".join(context_chunks)}

Q: {user_query}
A:
""".strip()

        # ── 4) GROQ API CALL  (retry + debug) ────────────────
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}",
                   "Content-Type":  "application/json"}
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system",
                 "content": "Return only sentences that exactly appear in TEXT."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 120,
            "temperature": 0.0
        }

        for attempt in range(5):
            try:
                if attempt:
                    delay = min(1 * 2**attempt, 8) + random.uniform(0.1, 0.4)
                    print(f"Retrying after {delay:.1f}s (attempt {attempt+1})")
                    time.sleep(delay)

                resp = requests.post(api_url, headers=headers,
                                     json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if data.get("choices"):
                    raw = data["choices"][0]["message"]["content"].strip()
                    answer = clean_answer(raw)
                else:
                    answer = "Error: Invalid response from API."
                break                                     # success →

            except requests.exceptions.HTTPError as e:
                print(f"HTTP {resp.status_code} attempt {attempt+1}")
                print("BODY:", resp.text)
                # 500 → retry, 400 → stop
                if resp.status_code == 500 and attempt < 4:
                    continue
                answer = ("Error: Bad request – check model/prompt."
                          if resp.status_code == 400 else
                          f"Error: API returned {resp.status_code}")
                break
            except requests.exceptions.RequestException as e:
                print(f"Network error attempt {attempt+1}: {e}")
                if attempt == 4:
                    answer = "Error: Network problem – API unreachable."

        st.session_state.last_api_call = time.time()

    except Exception as err:
        print("rag_answer exception:", err)
        answer = f"Error: {err}"

    return (answer, context_chunks) if return_context else answer

# ── CLEAN ANSWER ───────────────────────────────────────────────
def clean_answer(text: str) -> str:
    fillers = [
        "Return the sentence that directly answers the question.",
        "If absent, reply exactly:",
        "Answer with the exact sentence from the text below that answers the question.",
        "If it is missing, say:",
        "TEXT:", "QUESTION:", "EXACT SENTENCE:",
        "Q:", "A:", "ANSWER:"
    ]
    for f in fillers:
        text = text.replace(f, "")
    text = text.strip(" \"'.,!?:;")
    return re.sub(r"\s+", " ", text)
