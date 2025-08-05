text
# 🎵 Music Album Review QA System

A Retrieval-Augmented Generation (RAG) Question-Answering system built for music album review documents. This project leverages sentence-level chunking, semantic embeddings, Chroma vector database, and Groq LLM to deliver accurate, dataset-grounded answers through an intuitive Streamlit interface.

##  Project Overview

This application enables users to ask questions about a curated dataset of music album reviews. The system retrieves relevant excerpts and generates answers strictly based on the dataset content, ensuring no hallucination or external knowledge injection.

##  Key Features

- **Smart Retrieval**: Semantic search using sentence transformers with re-ranking
- **Dual Prompt Modes**: Direct answering and role-based advanced prompting
- **Real-time Evaluation**: F1, Precision, Recall, Cosine Similarity, ROUGE-L, and LLM-as-a-Judge metrics
- **Interactive UI**: Dark/Light theme support with collapsible sidebar
- **Evidence Display**: Top 3 most relevant document chunks for transparency
- **Robust Error Handling**: Exponential backoff retry logic for API reliability
- **Performance Metrics**: Comprehensive evaluation dashboard with downloadable results

##  Technologies Used

- **Text Processing**: NLTK for sentence chunking
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers)
- **Vector Storage**: Chroma database for semantic search
- **Re-ranking**: Cross-encoder for context relevance scoring
- **LLM Inference**: Groq API (`llama-3.1-8b-instant`)
- **Frontend**: Streamlit with custom CSS styling
- **Evaluation**: F1 score, Precision, Recall, ROUGE-L, Cosine Similarity, LLM-as-a-Judge

##  Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see `requirements.txt`)

##  Features in Detail

### Smart Retrieval
- Retrieves top-25 candidate chunks
- Re-ranks using Cross-Encoder for relevance
- Selects top-12 most relevant contexts

### Error Handling
- Exponential backoff retry logic
- Graceful API failure handling
- Comprehensive error logging

### User Interface
- Responsive design with theme switching
- Chat history management
- Interactive evaluation dashboard
- Evidence transparency with top-3 chunk display

##  Acknowledgments

- Groq for LLM API services
- HuggingFace for embedding models
- Streamlit for the web framework
- The open-source community for various libraries used