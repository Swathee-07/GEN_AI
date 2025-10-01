# streamlit_app.py
import os
import json
import statistics as stats
import pandas as pd
import streamlit as st
import traceback

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
        
        /* Debug info styling */
        .debug-info {{
            background: {t['box_bg']};
            border: 1px solid {t['primary']};
            border-radius: 6px;
            padding: 0.5rem;
            margin: 0.5rem 0;
            font-size: 0.8rem;
            font-family: monospace;
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

# ---------------- Utilities -------------------
def normalize_question(q: str) -> str:
    if not q: return ""
    q = q.strip().lower()
    q = q.replace("'", "'").replace(""", '"').replace(""", '"')
    q = " ".join(q.split())
    return q

def get_file_debug_info():
    """Get debug information about file paths and existence"""
    debug_info = []
    
    # Check current working directory
    debug_info.append(f"Current working directory: {os.getcwd()}")
    debug_info.append(f"__file__ location: {__file__ if '__file__' in globals() else 'Not available'}")
    debug_info.append(f"BASE_DIR: {BASE_DIR}")
    
    # Check possible paths for queries.json
    possible_paths = [
        asset_path("evaluation", "queries.json"),
        asset_path("queries.json"),
        os.path.join(os.getcwd(), "evaluation", "queries.json"),
        os.path.join(os.getcwd(), "queries.json"),
        "evaluation/queries.json",
        "queries.json"
    ]
    
    for path in possible_paths:
        exists = os.path.exists(path)
        debug_info.append(f"Path: {path} - Exists: {exists}")
        if exists:
            try:
                size = os.path.getsize(path)
                debug_info.append(f"  Size: {size} bytes")
            except:
                debug_info.append(f"  Size: Unable to get size")
    
    # List all files in current directory and subdirectories
    try:
        debug_info.append("\nFiles in current directory:")
        for root, dirs, files in os.walk("."):
            if len(files) > 0:
                debug_info.append(f"  {root}: {files}")
                if len(debug_info) > 50:  # Limit output
                    debug_info.append("  ... (truncated)")
                    break
    except Exception as e:
        debug_info.append(f"Error listing files: {e}")
    
    return debug_info

def load_ground_truth_cloud_safe():
    """Load ground truth data with cloud-specific handling"""
    debug_info = []
    
    try:
        # Get debug information first
        debug_info.extend(get_file_debug_info())
        
        # Try multiple possible paths with cloud-specific adjustments
        possible_paths = [
            # Standard paths
            asset_path("evaluation", "queries.json"),
            asset_path("queries.json"),
            # Cloud-specific paths
            os.path.join(os.getcwd(), "evaluation", "queries.json"),
            os.path.join(os.getcwd(), "queries.json"),
            # Relative paths
            "./evaluation/queries.json",
            "./queries.json",
            "evaluation/queries.json",
            "queries.json"
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                debug_info.append(f"Found queries.json at: {path}")
                break
        
        if not found_path:
            return {}, [], f"queries.json not found in any location. Checked: {possible_paths}", debug_info
        
        # Try to read the file
        try:
            with open(found_path, "r", encoding="utf-8") as f:
                content = f.read()
                debug_info.append(f"File content length: {len(content)} characters")
                
                if not content.strip():
                    return {}, [], "queries.json file is empty", debug_info
                
                data = json.loads(content)
                debug_info.append(f"JSON parsed successfully, {len(data)} items")
                
        except json.JSONDecodeError as e:
            return {}, [], f"Invalid JSON in {found_path}: {str(e)}", debug_info
        except Exception as e:
            return {}, [], f"Error reading {found_path}: {str(e)}", debug_info
        
        if not data:
            return {}, [], "JSON file contains no data", debug_info
            
        # Process the data
        mapping = {}
        valid_items = []
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                debug_info.append(f"Item {i} is not a dictionary: {type(item)}")
                continue
                
            # Get question and answer with multiple possible keys
            question = (item.get("question") or 
                       item.get("query") or 
                       item.get("q") or "")
            
            answer = (item.get("answer") or 
                     item.get("ground_truth") or 
                     item.get("a") or 
                     item.get("expected_answer") or "")
            
            if question and answer:
                normalized_q = normalize_question(question)
                mapping[normalized_q] = {
                    "original_question": question,
                    "expected_answer": answer,
                    "raw_item": item
                }
                valid_items.append(item)
            else:
                debug_info.append(f"Item {i} missing question or answer: {item}")
        
        debug_info.append(f"Successfully processed {len(mapping)} question-answer pairs")
        
        if not mapping:
            return {}, data, f"No valid question-answer pairs found in {found_path}", debug_info
            
        return mapping, valid_items, None, debug_info
        
    except Exception as e:
        error_trace = traceback.format_exc()
        debug_info.append(f"Unexpected error: {error_trace}")
        return {}, [], f"Unexpected error loading ground truth: {str(e)}", debug_info

def refresh_ground_truth():
    """Refresh ground truth data with enhanced debugging"""
    gt_mapping, gt_raw, gt_error, debug_info = load_ground_truth_cloud_safe()
    
    st.session_state.ground_truth_mapping = gt_mapping
    st.session_state.ground_truth_raw = gt_raw
    st.session_state.ground_truth_error = gt_error
    st.session_state.ground_truth_debug = debug_info
    
    return len(gt_mapping)

# Load ground truth on first run
if "ground_truth_mapping" not in st.session_state:
    refresh_ground_truth()

def find_ground_truth_answer(question: str):
    """Find expected answer for a question"""
    if not st.session_state.ground_truth_mapping:
        return None
    
    normalized_q = normalize_question(question)
    
    # Exact match first
    if normalized_q in st.session_state.ground_truth_mapping:
        return st.session_state.ground_truth_mapping[normalized_q]["expected_answer"]
    
    # Fuzzy matching
    question_words = set(normalized_q.split())
    best_match = None
    best_score = 0
    
    for gt_q, gt_data in st.session_state.ground_truth_mapping.items():
        gt_words = set(gt_q.split())
        if not gt_words:
            continue
            
        overlap = len(question_words.intersection(gt_words))
        total_words = len(question_words.union(gt_words))
        
        if total_words > 0:
            score = overlap / total_words
            if score > best_score and score > 0.6:
                best_score = score
                best_match = gt_data["expected_answer"]
    
    return best_match

def compute_metrics_with_fallback(q_raw: str, ans: str):
    """Enhanced metrics computation with cloud error handling"""
    def safe_metrics(q, a):
        try:
            # Try with timeout for cloud environment
            m = all_metrics(q, a) or {}
        except Exception as e:
            # Log the error for debugging
            if "metrics_errors" not in st.session_state:
                st.session_state.metrics_errors = []
            st.session_state.metrics_errors.append(f"Metrics error for '{q[:50]}...': {str(e)}")
            m = {}
        
        # Ensure all required metrics exist with default values
        out = {}
        for k in ["f1","precision","recall","cosine","f1_llm_combined","rougeL"]:
            try: 
                val = m.get(k, 0.0)
                out[k] = float(val) if val is not None else 0.0
            except: 
                out[k] = 0.0
        return out

    # Check for ground truth
    expected_answer = find_ground_truth_answer(q_raw)
    has_ground_truth = expected_answer is not None
    
    # Compute metrics with fallbacks
    m = safe_metrics(q_raw, ans)
    
    # If metrics are all zero, try variations
    if all(v == 0.0 for v in m.values()):
        qn = normalize_question(q_raw)
        m2 = safe_metrics(qn, ans)
        if sum(m2.values()) > 0:
            m = m2
        else:
            m3 = safe_metrics(qn.rstrip("?"), ans)
            if sum(m3.values()) > 0:
                m = m3
    
    return m, has_ground_truth

# ---------------- Sidebar ---------------------
def sidebar_body():
    st.markdown("### Music RAG")
    logo = asset_path("logo2.jpg")
    if os.path.exists(logo):
        st.image(logo, width=80)

    st.markdown("#### Sample Questions")
    
    # Use ground truth questions if available
    if st.session_state.ground_truth_raw and len(st.session_state.ground_truth_raw) > 0:
        samples = []
        for item in st.session_state.ground_truth_raw[:4]:
            if isinstance(item, dict):
                q = item.get("question") or item.get("query") or item.get("q", "")
                if q:
                    samples.append(q)
        
        if len(samples) < 4:
            defaults = [
                "When was the album Happier Than Ever by Billie Eilish released?",
                "What major British award did the song win in 2012?", 
                "When was the song 'Hello' by Adele released?",
                "What musical styles does 'Dynamite' incorporate?",
            ]
            samples.extend(defaults[len(samples):])
    else:
        samples = [
            "When was the album Happier Than Ever by Billie Eilish released?",
            "What major British award did the song win in 2012?",
            "When was the song 'Hello' by Adele released?",
            "What musical styles does 'Dynamite' incorporate?",
        ]
    
    for i, q in enumerate(samples):
        display_q = q[:65] + "..." if len(q) > 65 else q
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
        st.session_state.prompt_mode = mode; st.rerun()

    theme_choice = st.radio("Theme", ["dark", "light"], index=0 if st.session_state.theme=="dark" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice; st.rerun()

    st.divider()
    if st.session_state.chat_history:
        st.markdown("#### Chat History")
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []; st.rerun()
    
    # Ground truth status with enhanced debugging
    st.divider()
    st.markdown("#### Ground Truth Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", help="Reload queries.json"):
            count = refresh_ground_truth()
            if count > 0:
                st.success(f"Loaded {count} questions")
            else:
                st.error("Failed to load questions")
            st.rerun()
    with col2:
        gt_count = len(st.session_state.ground_truth_mapping)
        st.metric("Questions", gt_count)
    
    # Enhanced status display
    if st.session_state.ground_truth_error:
        st.error(f"⚠️ {st.session_state.ground_truth_error}")
    elif gt_count > 0:
        st.success(f"✅ {gt_count} questions loaded")
        # Show metrics errors if any
        if hasattr(st.session_state, 'metrics_errors') and st.session_state.metrics_errors:
            with st.expander("⚠️ Metrics Issues"):
                for error in st.session_state.metrics_errors[-3:]:  # Show last 3 errors
                    st.warning(error)
    else:
        st.info("No ground truth data loaded")

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
        with st.chat_message("user"): st.write(q_raw)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ans, ctx = rag_answer(q_raw, return_context=True, prompt_mode=st.session_state.prompt_mode)
                except Exception as e:
                    st.error(f"Inference failed. If this shows 401, configure API secrets. Error: {e}")
                    ans, ctx = ("Sorry, the model could not answer due to a configuration error.", [])
                mets, has_gt = compute_metrics_with_fallback(q_raw, ans)
                st.markdown(f"<div class='answer-container'>{ans}</div>", unsafe_allow_html=True)
                if ctx:
                    with st.expander("Show Evidence"):
                        top3 = [str(ev) for ev in (ctx or [])[:3]]
                        preview = "\n\n".join(ev[:200] + "..." if len(ev) > 200 else ev for ev in top3) if top3 else "No evidence available."
                        st.info(preview)
        st.session_state.chat_history.append(
            {"question": q_raw, "answer": ans, "context": ctx, "metrics": mets, "prompt_mode_used": st.session_state.prompt_mode, "has_ground_truth": has_gt}
        )
        st.rerun()

# ---------------- Enhanced Evaluation Dashboard ---------------
with tab_eval:
    # Show debug information if needed
    if st.session_state.ground_truth_error and len(st.session_state.ground_truth_mapping) == 0:
        with st.expander("🔧 Debug Information (Click to expand)", expanded=False):
            st.markdown("**File System Debug Info:**")
            if hasattr(st.session_state, 'ground_truth_debug'):
                for info in st.session_state.ground_truth_debug:
                    st.code(info, language=None)
    
    if not st.session_state.chat_history:
        st.info("Ask questions in the Ask AI tab to see the evaluation here.")
        if st.session_state.ground_truth_error:
            st.error(f"❌ {st.session_state.ground_truth_error}")
        elif len(st.session_state.ground_truth_mapping) > 0:
            st.success(f"✅ {len(st.session_state.ground_truth_mapping)} ground truth questions loaded")
        else:
            st.warning("⚠️ No ground truth data found. Upload evaluation/queries.json file.")
    else:
        eval_data = [c for c in st.session_state.chat_history if isinstance(c.get("metrics"), dict)]
        
        if not eval_data:
            st.warning("No questions have been evaluated yet.")
            st.info("💡 Ask some questions first to see evaluation metrics.")
        else:
            # Show evaluation statistics
            st.markdown("#### 📊 Evaluation Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Questions", len(st.session_state.chat_history))
            with col2:
                st.metric("Evaluated", len(eval_data))
            with col3:
                gt_count = sum(1 for c in eval_data if c.get("has_ground_truth", False))
                st.metric("With Ground Truth", gt_count)
            with col4:
                st.metric("GT Available", len(st.session_state.ground_truth_mapping))
            
            # Calculate and display metrics
            keys = ["f1", "precision", "recall", "cosine", "f1_llm_combined", "rougeL"]
            agg = {}
            
            # Enhanced metrics calculation with validation
            for k in keys:
                vals = []
                for c in eval_data:
                    try: 
                        val = float(c["metrics"].get(k, 0.0))
                        if val is not None and not (val != val):  # Check for NaN
                            vals.append(val)
                    except: 
                        continue
                agg[k] = round(stats.mean(vals), 3) if vals else 0.0

            st.markdown("#### 🎯 Overall Performance")
            
            # Only show metrics if we have valid data
            if any(v > 0 for v in agg.values()):
                c1, c2, c3 = st.columns(3)
                d1, d2, d3 = st.columns(3)
                c1.metric("F1 Score", f"{agg['f1']:.3f}")
                c2.metric("Precision", f"{agg['precision']:.3f}")
                c3.metric("Recall", f"{agg['recall']:.3f}")
                d1.metric("Cosine Similarity", f"{agg['cosine']:.3f}")
                d2.metric("LLM F1", f"{agg['f1_llm_combined']:.3f}")
                d3.metric("ROUGE-L", f"{agg['rougeL']:.3f}")

                st.markdown("---")
                st.markdown("#### 📈 Performance Visualization")
                bar_df = pd.DataFrame(
                    {"Metric": ["F1", "Precision", "Recall", "Cosine", "LLM+F1", "ROUGE-L"],
                     "Score": [agg['f1'], agg['precision'], agg['recall'], agg['cosine'], agg['f1_llm_combined'], agg['rougeL']]}
                )
                st.bar_chart(bar_df.set_index("Metric"))
            else:
                st.warning("⚠️ No valid metrics computed yet.")
                st.info("This might indicate:")
                st.info("• Your evaluation function needs debugging")
                st.info("• Questions don't match ground truth data")
                st.info("• Missing dependencies in cloud environment")
                
                # Show raw metrics data for debugging
                with st.expander("🔧 Raw Metrics Debug"):
                    for i, chat in enumerate(eval_data[:3]):  # Show first 3
                        st.write(f"**Question {i+1}:** {chat['question'][:50]}...")
                        st.write(f"**Metrics:** {chat.get('metrics', {})}")
                        st.write(f"**Has GT:** {chat.get('has_ground_truth', False)}")
                        st.write("---")

            # Ground Truth References section
            st.markdown("#### 📚 Available Ground Truth Questions")
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
                st.warning("📁 No ground truth data loaded.")
                st.info("Make sure your `evaluation/queries.json` file is uploaded to your repository.")

            # Download results
            if eval_data:
                if st.button("📥 Download Evaluation Results"):
                    df = pd.DataFrame([
                        {
                            "Question": c["question"], 
                            "Answer": c.get("answer", ""),
                            "Prompt Mode": c.get("prompt_mode_used", "N/A"),
                            "Has Ground Truth": c.get("has_ground_truth", False),
                            **c.get("metrics", {})
                        } for c in eval_data
                    ])
                    st.download_button("Download CSV", df.to_csv(index=False).encode(), "evaluation_results.csv", "text/csv")
                st.caption(f"📊 Total Evaluated: {len(eval_data)} | With GT: {sum(1 for c in eval_data if c.get('has_ground_truth', False))}")