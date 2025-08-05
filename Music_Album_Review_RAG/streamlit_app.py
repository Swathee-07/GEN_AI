import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from app.rag import rag_answer
from app.evaluation import all_metrics

# --- Page Configuration -------------------------------------------------------
st.set_page_config(page_title="Music Album RAG", page_icon="🎵", layout="centered")

# --------------- THEME --------------------------------------------------------
THEMES = {
    "light": {
        "primary": "#4A0D66", "bg": "#FFFFFF", "text": "#262730",
        "secondary_bg": "#F0E4F4", "sidebar_bg": "#F8F8F8",
        "box_bg": "#f7efff", "button_text": "#262730", "button_bg": "#FFFFFF"
    },
    "dark": {
        "primary": "#C39BD3", "bg": "#0E1117", "text": "#FAFAFA",
        "secondary_bg": "#2C1D38", "sidebar_bg": "#171420",
        "box_bg": "#321352", "button_text": "#FAFAFA", "button_bg": "#2C1D38"
    }
}

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "prompt_mode" not in st.session_state:
    st.session_state.prompt_mode = "Direct Answering (Standard RAG)"

def load_css(theme_name):
    theme = THEMES.get(theme_name.lower(), THEMES["dark"])
    st.markdown(f"""
    <style>
        html, body, [class*="st-"], [class*="css-"] {{
            font-family: 'Poppins', sans-serif;
            color: {theme['text']} !important;
        }}
        .stApp {{
            background-color: {theme['bg']} !important;
            color: {theme['text']} !important;
        }}
        [data-testid="stSidebar"] {{
            color: {theme['text']} !important;
            background-color: {theme['sidebar_bg']} !important;
            width: 240px !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {theme['text']} !important;
        }}
        
        /* FINAL FIX: Force button text visibility in light theme */
        [data-testid="stSidebar"] .stButton > button {{
            color: {theme['button_text']} !important;
            background-color: {theme['button_bg']} !important;
            border: 2px solid {theme['primary']} !important;
            font-weight: 600 !important;
        }}
        
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: {theme['primary']} !important;
            color: white !important;
        }}

        .ep-expander__container, .stTable, .stDataFrame {{
            color: {theme['text']} !important;
            background-color: {theme['bg']} !important;
        }}
        .answer-container {{
            background-color: {theme['box_bg']};
            border-left: 5px solid {theme['primary']};
            color: {theme['text']} !important;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .scrollable-chat {{
            max-height: 55vh;
            overflow-y: auto;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        .logo-container {{
            text-align: center;
            padding: 1rem 0;
            border-bottom: 1px solid {theme['primary']};
            margin-bottom: 1rem;
        }}
        [data-testid="collapsedControl"] {{
            background-color: {theme['primary']} !important;
            color: white !important;
        }}
    </style>
    """, unsafe_allow_html=True)

load_css(st.session_state.theme)

# --- Session State ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --------------- SIDEBAR (ARROW REMOVED) -------------------------------------
with st.sidebar:
    # Logo with styling
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image("./logo2.jpg", width=80)
    st.markdown("**Music RAG**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Sample Questions")
    sample_questions = [
        "When was the album Happier Than Ever by Billie Eilish released?",
        "What major British award did the song win in 2012?",
        "When was the song 'Hello' by Adele released?",
        "What musical styles does 'Dynamite' incorporate?",
        "Who wrote 'Thinking Out Loud' ?",
        "What musical genres are incorporated in Happier Than Ever?"
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.user_input = q
            st.rerun()

    st.divider()
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("### Settings")

    # Prompt-mode selector
    st.markdown(f"**Current Mode:** {st.session_state.prompt_mode}")
    new_prompt_mode = st.radio(
        "Select Prompting Technique:",
        ("Direct Answering (Standard RAG)", "Role-Based Answering (Advanced)"),
        index=0 if st.session_state.prompt_mode == "Direct Answering (Standard RAG)" else 1,
        key="prompt_selector"
    )
    if new_prompt_mode != st.session_state.prompt_mode:
        st.session_state.prompt_mode = new_prompt_mode
        st.rerun()

    # Theme switcher
    switches = {"Dark": "🌙", "Light": "☀️"}
    new_theme = st.radio(
        "Theme",
        list(switches.keys()),
        index=0 if st.session_state.theme == "dark" else 1,
        format_func=lambda v: f"{switches[v]} {v}",
        horizontal=True
    )
    if new_theme.lower() != st.session_state.theme:
        st.session_state.theme = new_theme.lower()
        st.rerun()

# --------------- MAIN APP -----------------------------------------------------
st.markdown("""
<h1 style='color: #C39BD3; font-weight: 800; font-size: 2.7rem; margin-bottom:8px;margin-top:0'>
<span style="font-size:2.6rem;vertical-align:middle;">🎵</span> 
<span style='color:#C39BD3'>Music Album</span> <span style="color:#9b59b6">Review <span style="color:#4A0D66">RAG</span></span>
</h1>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Ask AI", "📊 Evaluation Dashboard"])

with tab1:
    st.info(f"🤖 Currently using: **{st.session_state.prompt_mode}**")
    with st.container():
        st.markdown('<div class="scrollable-chat">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(f'<div class="answer-container">{chat["answer"]}</div>', unsafe_allow_html=True)
                # FIXED: Show only top 3 evidence chunks
                with st.expander("Show Evidence"):
                    top3 = chat["context"][:3]
                    st.info("\n\n".join(top3))
        st.markdown('</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask about an album review...", key="chat_widget"):
    st.session_state.user_input = prompt

if st.session_state.user_input:
    query = st.session_state.user_input
    st.session_state.user_input = ""

    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, context = rag_answer(query, return_context=True, prompt_mode=st.session_state.prompt_mode)
            metrics = all_metrics(query, answer)
            st.session_state.chat_history.append({
                "question": query, "answer": answer, "context": context, "metrics": metrics,
                "prompt_mode_used": st.session_state.prompt_mode
            })
    st.rerun()

with tab2:
    if not st.session_state.chat_history:
        st.info("Ask questions in the 'Ask AI' tab to see the evaluation here.")
    else:
        eval_data = [c for c in st.session_state.chat_history if c.get("metrics", {}).get("f1", 0) > 0]
        if not eval_data:
            st.warning("No questions with available ground truth have been asked yet.")
        else:
            from statistics import mean
            def compute_overall_metrics(chat_list):
                # UPDATED METRICS - Only F1, Precision, Recall, Cosine, LLM+F1, ROUGE-L
                keys = ['f1', 'precision', 'recall', 'cosine', 'f1_llm_combined', 'rougeL']
                scores = {k: [] for k in keys}
                for c in chat_list:
                    if c.get('metrics'):
                        for k in keys:
                            v = c['metrics'].get(k)
                            if v and v > 0: scores[k].append(v)
                return {k: round(mean(v), 3) if v else 0.0 for k, v in scores.items()}
            overall_metrics = compute_overall_metrics(eval_data)

            st.markdown("### 📊 Overall Performance")
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
            
            with col1: st.metric("**F1 Score**", f"{overall_metrics['f1']:.3f}")
            with col2: st.metric("**Precision**", f"{overall_metrics['precision']:.3f}")
            with col3: st.metric("**Recall**", f"{overall_metrics['recall']:.3f}")
            with col4: st.metric("**Cosine Similarity**", f"{overall_metrics['cosine']:.3f}")
            with col5: st.metric("**LLM + F1**", f"{overall_metrics['f1_llm_combined']:.3f}")
            with col6: st.metric("**ROUGE-L**", f"{overall_metrics['rougeL']:.3f}")

            st.markdown("---")
            st.markdown("### 📈 Performance Visualization")
            labels = {
                'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall',
                'cosine': 'Cosine', 'f1_llm_combined': 'LLM+F1', 'rougeL': 'ROUGE-L'
            }
            bar_df = pd.DataFrame({
                'Metric': [labels[k] for k in overall_metrics.keys()],
                'Score': list(overall_metrics.values())
            })
            st.bar_chart(bar_df.set_index('Metric'))

            with st.expander("📖 View Ground Truth References"):
                import json
                try:
                    with open('evaluation/queries.json', 'r', encoding='utf-8') as f:
                        ground_truth = json.load(f)
                    st.json(ground_truth[:5])
                except:
                    st.error("Could not load ground truth file")

            if st.button("💾 Download Evaluation Results"):
                df = pd.DataFrame([{
                    "Question": c["question"],
                    "Prompt Mode": c.get("prompt_mode_used", "N/A"),
                    **c["metrics"]
                } for c in eval_data])
                csv = df.to_csv(index=False).encode()
                st.download_button("Download CSV", csv, "evaluation_results.csv", "text/csv")

            st.markdown(f"**Total Questions Evaluated:** {len(eval_data)}")
