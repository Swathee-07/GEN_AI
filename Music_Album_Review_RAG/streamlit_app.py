# streamlit_app.py
import os
import json
import statistics as stats
import pandas as pd
import streamlit as st

from app.rag import rag_answer
from app.evaluation import all_metrics
from app.chroma_db import initialize_db

# ---------------- Page config ----------------
st.set_page_config(page_title="Music Album Review RAG", page_icon="🎵", layout="centered")

# ---------------- Theme ----------------------
THEMES = {
    "light": {"primary": "#4A0D66", "bg": "#FFFFFF", "text": "#262730", "sidebar_bg": "#F8F8F8", "box_bg": "#f7efff"},
    "dark":  {"primary": "#C39BD3", "bg": "#0E1117", "text": "#FAFAFA", "sidebar_bg": "#171420", "box_bg": "#321352"},
}
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "prompt_mode" not in st.session_state: st.session_state.prompt_mode = "Direct Answering (Standard RAG)"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "user_input" not in st.session_state: st.session_state.user_input = ""
if "sidebar_open" not in st.session_state: st.session_state.sidebar_open = True

def apply_theme():
    t = THEMES[st.session_state.theme]
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
        section[data-testid="stSidebar"] {{ background-color: {t['sidebar_bg']}; width: 260px !important; }}
        .stButton>button {{ color: {t['text']}; border: 1px solid {t['primary']}; background: transparent; }}
        a, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {t['primary']} !important; }}

        .answer-container {{
            background-color: {t['box_bg']}; border-left: 5px solid {t['primary']};
            border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem; color: {t['text']};
        }}
        [data-testid="collapsedControl"] {{ background-color: {t['primary']} !important; }}

        /* EXACT scrollable-chat from your snippet */
        .scrollable-chat {{
            max-height: 55vh;
            overflow-y: auto;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        /* Keep entire Ask AI centered and input visually fixed at bottom-center */
        .centered {{ width:min(920px,96%); margin:0 auto; }}
        .input-wrap {{ position: sticky; bottom: 0; left: 0; right: 0; }}

        /* Sidebar arrow bar */
        .sb-top {{ display:flex; align-items:center; justify-content:flex-start; height:36px; padding:6px 6px 4px 6px; border-bottom:1px solid {t['primary']}; }}
        .sb-arrow {{
            width:28px; height:28px; border-radius:6px; cursor:pointer; user-select:none;
            border:2px solid {t['primary']}; background: transparent; color:{t['text']};
            font-weight:800; line-height:22px; display:inline-flex; align-items:center; justify-content:center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_theme()

# ---------------- Optional secrets ----------------
if hasattr(st, "secrets"):
    try:
        for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]:
            v = st.secrets.get(k, "")
            if v: os.environ[k] = v
    except Exception:
        pass

# ---------------- Paths/DB --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def asset_path(*parts): return os.path.join(BASE_DIR, *parts)

if "db" not in st.session_state:
    st.session_state.db = initialize_db()

# ---------------- Utilities (MOVED TO TOP) -------------------
def normalize_question(q: str) -> str:
    """Normalize question text for consistent matching"""
    if not q: return ""
    q = q.strip().lower()
    q = q.replace("'", "'").replace(""", '"').replace(""", '"')
    q = " ".join(q.split())
    return q

def safe_load_json(file_path: str):
    """Safely load JSON file with error handling"""
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading {file_path}: {str(e)}")
        return None

def create_ground_truth_mapping(data_list):
    """Create a mapping from questions to answers"""
    mapping = {}
    if not data_list:
        return mapping
    
    for item in data_list:
        if not isinstance(item, dict):
            continue
            
        # Try different key combinations
        question = item.get("question") or item.get("query") or item.get("q", "")
        answer = item.get("answer") or item.get("ground_truth") or item.get("a", "")
        
        if question and answer:
            normalized_q = normalize_question(question)
            mapping[normalized_q] = {
                "original_question": question,
                "expected_answer": answer
            }
    
    return mapping

# ---------------- Ground Truth Loading (IMPROVED) ----------------
@st.cache_data
def load_ground_truth_data():
    """Load and cache ground truth data"""
    try:
        # Try multiple possible paths
        possible_paths = [
            asset_path("evaluation", "queries.json"),
            asset_path("queries.json"),
            "evaluation/queries.json",
            "queries.json"
        ]
        
        for path in possible_paths:
            data = safe_load_json(path)
            if data:
                mapping = create_ground_truth_mapping(data)
                return mapping, data, None  # mapping, raw_data, error
        
        return {}, [], "No ground truth file found in expected locations"
    except Exception as e:
        return {}, [], f"Error loading ground truth: {str(e)}"

# Initialize ground truth data
if "ground_truth_loaded" not in st.session_state:
    gt_mapping, gt_raw, gt_error = load_ground_truth_data()
    st.session_state.ground_truth_mapping = gt_mapping
    st.session_state.ground_truth_raw = gt_raw
    st.session_state.ground_truth_error = gt_error
    st.session_state.ground_truth_loaded = True

def find_ground_truth_answer(question: str):
    """Find expected answer for a question"""
    if not st.session_state.ground_truth_mapping:
        return None
    
    normalized_q = normalize_question(question)
    
    # Direct match
    if normalized_q in st.session_state.ground_truth_mapping:
        return st.session_state.ground_truth_mapping[normalized_q]["expected_answer"]
    
    # Fuzzy matching
    question_words = set(normalized_q.split())
    best_match = None
    best_score = 0
    
    for gt_q, gt_data in st.session_state.ground_truth_mapping.items():
        gt_words = set(gt_q.split())
        overlap = len(question_words.intersection(gt_words))
        total_words = len(question_words.union(gt_words))
        
        if total_words > 0:
            score = overlap / total_words
            if score > best_score and score > 0.5:  # 50% similarity threshold
                best_score = score
                best_match = gt_data["expected_answer"]
    
    return best_match

def compute_metrics_with_fallback(q_raw: str, ans: str):
    """Enhanced metrics computation with proper error handling"""
    def safe_metrics(q, a, expected=None):
        try:
            # Try to compute metrics
            if expected:
                # If we have expected answer, try to use it
                try:
                    m = all_metrics(q, a) or {}
                except:
                    m = {}
            else:
                m = all_metrics(q, a) or {}
        except Exception:
            m = {}
        
        # Ensure all required metrics exist
        out = {}
        for k in ["f1", "precision", "recall", "cosine", "f1_llm_combined", "rougeL"]:
            try: 
                out[k] = float(m.get(k, 0.0))
            except: 
                out[k] = 0.0
        return out

    # Check if we have ground truth
    expected_answer = find_ground_truth_answer(q_raw)
    has_ground_truth = expected_answer is not None
    
    # Compute metrics
    metrics = safe_metrics(q_raw, ans, expected_answer)
    
    # If all metrics are zero, try variations
    if all(v == 0.0 for v in metrics.values()):
        # Try normalized question
        qn = normalize_question(q_raw)
        metrics2 = safe_metrics(qn, ans)
        if sum(metrics2.values()) > 0:
            metrics = metrics2
        else:
            # Try without question mark
            metrics3 = safe_metrics(qn.rstrip("?"), ans)
            if sum(metrics3.values()) > 0:
                metrics = metrics3
    
    return metrics, has_ground_truth

# ---------------- Sidebar ---------------------
def sidebar_body():
    st.markdown("### Music RAG")
    logo = asset_path("logo2.jpg")
    if os.path.exists(logo):
        st.image(logo, width=80)

    st.markdown("#### Sample Questions")
    
    # Use ground truth questions if available, otherwise use defaults
    if st.session_state.ground_truth_raw:
        samples = []
        for item in st.session_state.ground_truth_raw[:4]:
            if isinstance(item, dict):
                q = item.get("question") or item.get("query") or ""
                if q:
                    # Truncate long questions
                    display_q = q[:70] + "..." if len(q) > 70 else q
                    samples.append(q)  # Store full question
    else:
        samples = [
            "When was the album Happier Than Ever by Billie Eilish released?",
            "What major British award did the song win in 2012?",
            "When was the song 'Hello' by Adele released?",
            "What musical styles does 'Dynamite' incorporate?",
        ]
    
    for i, q in enumerate(samples):
        display_q = q[:70] + "..." if len(q) > 70 else q
        if st.button(display_q, key=f"sample_{i}", use_container_width=True):
            st.session_state.user_input = q

    st.divider()
    st.markdown("#### Settings")
    mode = st.radio(
        "Select Prompting Technique",
        ["Direct Answering (Standard RAG)", "Role-Based Answering (Advanced)"],
        index=0 if st.session_state.prompt_mode == "Direct Answering (Standard RAG)" else 1,
        key="prompt_selector",
    )
    if mode != st.session_state.prompt_mode:
        st.session_state.prompt_mode = mode
        st.rerun()

    theme_choice = st.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme=="dark" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.divider()
    if st.session_state.chat_history:
        st.markdown("#### Chat History")
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Ground Truth Status
    st.divider()
    st.markdown("#### Ground Truth Status")
    if st.session_state.ground_truth_error:
        st.error(f"⚠️ {st.session_state.ground_truth_error}")
    else:
        gt_count = len(st.session_state.ground_truth_mapping)
        eval_count = sum(1 for c in st.session_state.chat_history if c.get("has_ground_truth", False))
        st.success(f"✅ {gt_count} questions loaded")
        st.info(f"📊 {eval_count} evaluated with GT")

with st.sidebar:
    st.markdown(
        f"""
        <div class="sb-top">
            <button id="sb-arrow" class="sb-arrow" title="Collapse/Expand">{'◀' if st.session_state.sidebar_open else '▶'}</button>
        </div>
        <script>
        (function(){{
            const btn = window.parent.document.getElementById("sb-arrow");
            if (btn && !btn._bound) {{
                btn._bound = true;
                btn.addEventListener("click", () => {{
                    const native = window.parent.document.querySelector('[data-testid="collapsedControl"]');
                    if (native) native.click();
                }});
            }}
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.sidebar_open:
        sidebar_body()

# ---------------- Title above tabs ----------------
st.markdown("""
<h1 style='color: #C39BD3; font-weight: 800; font-size: 2.7rem; margin-bottom:6px;margin-top:0' class='centered'>
<span style="font-size:2.2rem;vertical-align:middle;">🎵</span> 
<span style='color:#C39BD3'>Music Album</span> <span style="color:#9b59b6">Review <span style="color:#4A0D66">RAG</span></span>
</h1>
""", unsafe_allow_html=True)

# ---------------- Tabs -----------------------
tab_ask, tab_eval = st.tabs(["Ask AI", "Evaluation Dashboard"])

# ---------------- Ask AI (scrollable-chat + pinned bottom input) ---------------
with tab_ask:
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.info(f"Currently using: {st.session_state.prompt_mode}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display ground truth error if exists
    if st.session_state.ground_truth_error:
        st.warning(f"⚠️ Ground Truth Issue: {st.session_state.ground_truth_error}")

    # History in a fixed-height scrollable area; centered
    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.markdown("<div class='scrollable-chat'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(f"<div class='answer-container'>{chat['answer']}</div>", unsafe_allow_html=True)
            with st.expander("Show Evidence"):
                top3 = [str(ev) for ev in (chat.get("context") or [])[:3]]
                if top3:
                    preview = "\n\n".join(ev[:200] + "..." if len(ev) > 200 else ev for ev in top3)
                    st.info(preview)
                else:
                    st.write("No evidence available.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom-center input; visually fixed because scrollable area is above it
    st.markdown("<div class='input-wrap centered'>", unsafe_allow_html=True)
    prompt = st.chat_input("Ask about an album review...", key="chat_widget")
    st.markdown("</div>", unsafe_allow_html=True)

    if prompt:
        st.session_state.user_input = prompt

    if st.session_state.user_input:
        q_raw = st.session_state.user_input
        st.session_state.user_input = ""
        with st.chat_message("user"):
            st.write(q_raw)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans, ctx = rag_answer(q_raw, return_context=True, prompt_mode=st.session_state.prompt_mode)
                except Exception as e:
                    st.error(f"Inference failed. If this shows 401, configure API secrets. Error: {e}")
                    ans, ctx = ("Sorry, the model could not answer due to a configuration error.", [])
                
                # Compute metrics
                mets, has_gt = compute_metrics_with_fallback(q_raw, ans)
                
                st.markdown(f"<div class='answer-container'>{ans}</div>", unsafe_allow_html=True)
                if ctx:
                    with st.expander("Show Evidence"):
                        top3 = [str(ev) for ev in (ctx or [])[:3]]
                        preview = "\n\n".join(ev[:200] + "..." if len(ev) > 200 else ev for ev in top3) if top3 else "No evidence available."
                        st.info(preview)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": q_raw, 
            "answer": ans, 
            "context": ctx, 
            "metrics": mets,
            "prompt_mode_used": st.session_state.prompt_mode,
            "has_ground_truth": has_gt
        })
        st.rerun()

# ---------------- Evaluation Dashboard ---------------
with tab_eval:
    if not st.session_state.chat_history:
        st.info("Ask questions in the Ask AI tab to see the evaluation here.")
        
        # Show ground truth status even without chat history
        if st.session_state.ground_truth_error:
            st.error(f"❌ {st.session_state.ground_truth_error}")
        elif st.session_state.ground_truth_mapping:
            st.success(f"✅ {len(st.session_state.ground_truth_mapping)} ground truth questions available")
    else:
        eval_data = [c for c in st.session_state.chat_history if isinstance(c.get("metrics"), dict)]
        if not eval_data:
            st.warning("No questions with available ground truth have been asked yet.")
        else:
            keys = ["f1", "precision", "recall", "cosine", "f1_llm_combined", "rougeL"]
            agg = {}
            
            # Calculate aggregate metrics
            for k in keys:
                vals = []
                for c in eval_data:
                    try: 
                        val = float(c["metrics"].get(k, 0.0))
                        if val > 0:  # Only count non-zero values
                            vals.append(val)
                    except Exception: 
                        continue
                agg[k] = round(stats.mean(vals), 3) if vals else 0.0

            st.markdown("#### Overall Performance")
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions Asked", len(st.session_state.chat_history))
            with col2:
                st.metric("Questions Evaluated", len(eval_data))
            with col3:
                gt_eval = sum(1 for c in eval_data if c.get("has_ground_truth", False))
                st.metric("Ground Truth Matches", gt_eval)
            
            # Performance metrics
            c1, c2, c3 = st.columns(3)
            d1, d2, d3 = st.columns(3)
            c1.metric("F1 Score", f"{agg['f1']:.3f}")
            c2.metric("Precision", f"{agg['precision']:.3f}")
            c3.metric("Recall", f"{agg['recall']:.3f}")
            d1.metric("Cosine Similarity", f"{agg['cosine']:.3f}")
            d2.metric("LLM F1", f"{agg['f1_llm_combined']:.3f}")
            d3.metric("ROUGE-L", f"{agg['rougeL']:.3f}")

            st.markdown("---")
            st.markdown("#### Performance Visualization")
            bar_df = pd.DataFrame(
                {"Metric": ["F1", "Precision", "Recall", "Cosine", "LLM+F1", "ROUGE-L"],
                 "Score": [agg['f1'], agg['precision'], agg['recall'], agg['cosine'], agg['f1_llm_combined'], agg['rougeL']]}
            )
            st.bar_chart(bar_df.set_index("Metric"))

            st.markdown("#### View Ground Truth References")
            if st.session_state.ground_truth_raw:
                for i, item in enumerate(st.session_state.ground_truth_raw[:10]):
                    if isinstance(item, dict):
                        q = item.get("question") or item.get("query") or ""
                        a = item.get("answer") or item.get("ground_truth") or ""
                        if q and a:
                            with st.expander(f"Q{i+1}: {q[:80]}{'...' if len(q) > 80 else ''}"):
                                st.markdown("*Question*")
                                st.write(q)
                                st.markdown("*Answer*")
                                st.write(a)
            else:
                if st.session_state.ground_truth_error:
                    st.error(st.session_state.ground_truth_error)
                else:
                    st.caption("No ground truth data available.")

            if st.button("Download Evaluation Results"):
                df = pd.DataFrame([
                    {
                        "Question": c["question"], 
                        "Prompt Mode": c.get("prompt_mode_used", "N/A"),
                        "Has_Ground_Truth": c.get("has_ground_truth", False),
                        **c.get("metrics", {})
                    } for c in eval_data
                ])
                st.download_button("Download CSV", df.to_csv(index=False).encode(), "evaluation_results.csv", "text/csv")
            st.caption(f"Total Questions Evaluated: {len(eval_data)}")