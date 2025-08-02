import streamlit as st
from PIL import Image
import pytesseract
import os
import fitz  # PyMuPDF for PDF
from groq import Groq
from cryptography.fernet import Fernet

# Load secret key
with open("secret.key", "rb") as key_file:
    key = key_file.read()

# Load encrypted API key
with open("encrypted_api.key", "rb") as enc_file:
    encrypted_key = enc_file.read()

# Decrypt the API key
fernet = Fernet(key)
api_key = fernet.decrypt(encrypted_key).decode()

# Use decrypted API key
client = Groq(api_key=api_key)

# Prompt builder function
def build_prompt(text, technique):
    if technique == "Chain-of-Thought":
        return f"""
You are an intelligent assistant trained to break down complex information into logical steps.
Your task is to read the article below, decompose it into smaller reasoning steps (step-by-step),
and then provide a final summary based on your reasoning.

Instructions:
1. Analyze the article step-by-step.
2. Ensure each step builds toward understanding the whole.
3. End with a concise and clear summary.

Article:
{text}

Now begin your chain of thought and final summary:
"""

    elif technique == "Tree-of-Thought":
        return f"""
You are an AI assistant that processes information by exploring multiple lines of reasoning.
For the given article, generate a tree of ideas — think through different branches (e.g., causes, effects, viewpoints, examples).
Summarize insights from these branches into a cohesive final summary.

Instructions:
- Branch 1: Identify main themes or ideas.
- Branch 2: Examine consequences or implications.
- Branch 3: Explore any counterpoints or contrasts.
- Combine all insights into a final summary.

Article:
{text}

Think in branches and provide your final summary:
"""

    elif technique == "Role-based":
        return f"""
You are a professional journalist with a background in editorial writing.
Your job is to read the given article, understand its essence, and write a polished, objective,
and informative summary suitable for a national newspaper.

Instructions:
- Maintain professional and neutral tone.
- Eliminate fluff, focus on facts and key points.
- Aim for clarity and readability for a broad audience.

Article:
{text}

Now write your journalist-style summary:
"""

    elif technique == "ReAct":
        return f"""
You are a reasoning assistant that follows the ReAct (Reasoning + Acting) framework:
First, think through the problem (Thought), then take action (Action), then reflect on what you observed (Observation).

Your task is to:
1. Use reasoning to understand the article.
2. Identify and summarize key points.
3. Reflect on important observations and finalize a summary.

Format:
Thought: [Write your reasoning]
Action: [Summarize the key ideas]
Observation: [Reflect and conclude]
Final Summary: [Your complete summary]

Article:
{text}

Start your ReAct reasoning now:
"""

    elif technique == "Directional Stimulus":
        return f"""
You are an assistant trained to understand directional relationships such as cause-and-effect.
Read the article and identify:
- What actions or events caused what outcomes.
- Relationships like if-then, due to, as a result, which led to, etc.
Use these patterns to construct a cause-effect map and generate a summary.

Instructions:
1. Extract major causes and their effects.
2. Highlight logical sequences of change or progression.
3. Conclude with a summary emphasizing the causal structure.

Article:
{text}

Now analyze cause-effect and give your summary:
"""

    elif technique == "Step-Back":
        return f"""
You are an expert summarizer trained in reflective thinking.
Step back from the details and understand the overall message, purpose, and context of the article.
Then, write a clear and meaningful summary based on that high-level understanding.

Instructions:
1. Identify the central idea or theme.
2. Understand the purpose or goal of the content.
3. Avoid over-focusing on minor details.
4. Provide a big-picture summary.

Article:
{text}

Now step back and summarize the big picture:
"""

    else:
        return text

# Function to call Groq's LLM
def generate_summary(text, technique):
    model = "llama3-8b-8192"  # ✅ Model is fixed here
    prompt = build_prompt(text, technique)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# File handling
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "application/pdf":
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])
    elif "image" in file.type:
        image = Image.open(file)
        return pytesseract.image_to_string(image)
    else:
        return None

# Streamlit UI
st.set_page_config(page_title="Prompting Techniques", layout="centered")
st.title("🧠 GenAI Summarizer with Advanced Prompting Techniques")

uploaded_file = st.file_uploader("📄 Upload a .txt, .pdf or image file", type=["txt", "pdf", "png", "jpg", "jpeg"])

technique = st.selectbox("🧪 Choose a prompting technique", [
    "Chain-of-Thought",
    "Tree-of-Thought",
    "Role-based",
    "ReAct",
    "Directional Stimulus",
    "Step-Back"
])

# 👇 Removed the model selection — it's fixed internally to llama3-8b-8192

if uploaded_file and st.button("🔍 Generate Summary"):
    with st.spinner("Extracting and summarizing..."):
        raw_text = extract_text(uploaded_file)
        if raw_text:
            try:
                summary = generate_summary(raw_text, technique)
                st.subheader("📝 Generated Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Unsupported file type or unreadable content.")

st.markdown("---")
st.markdown("Built with ❤️ using Groq + Streamlit")
