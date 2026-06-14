import streamlit as st

import sys
import os

# ── Fix paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Change working directory to src so relative paths work
os.chdir(SRC_DIR)

from rag_pipeline import load_store, ask



# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaBot AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 1px solid #00d4ff33;
    }

    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .bot-message {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        color: white;
        border: 1px solid #00d4ff33;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }

    /* Input box */
    .stTextInput input {
        background: #1a1a2e !important;
        color: white !important;
        border: 2px solid #00d4ff !important;
        border-radius: 25px !important;
        padding: 15px 20px !important;
        font-size: 16px !important;
    }

    /* Send button */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 30px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea33, #764ba233);
        border-radius: 15px;
        border: 1px solid #667eea44;
        margin-bottom: 20px;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 5px 0;
    }

    /* Divider */
    hr {
        border-color: #00d4ff22 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load FAISS store once ────────────────────────────────────────
@st.cache_resource
def initialize_store():
    load_store()
    return True

initialize_store()

initialize_store()


# ── Session State ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am **PharmaBot AI**. Ask me anything about the pharma molecule **Sitagliptin** — including its mechanism of action, clinical efficacy, cardiac effects, drug formulation, and more!"
        }
    ]

if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 60px;'>💊</div>
        <h2 style='color: #00d4ff; margin: 0;'>PharmaBot AI</h2>
        <p style='color: #888; font-size: 12px;'>Pharmaceutical Research Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div style='font-size: 24px; color: #667eea;'>{st.session_state.question_count}</div>
            <div style='font-size: 11px; color: #888;'>Questions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div style='font-size: 24px; color: #00d4ff;'>{len(st.session_state.messages)}</div>
            <div style='font-size: 11px; color: #888;'>Messages</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sample Questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the mechanism of action of sitagliptin?",
        "How does sitagliptin affect JAK/STAT pathway?",
        "What is the role of metformin in diabetes?",
        "How does sitagliptin help in cardiomyopathy?",
        "What is DPP-4 inhibitor?",
        "What are the side effects of sitagliptin?",
        "How does sitagliptin affect cardiac function?"
    ]

    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.pending_question = q

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I am **PharmaBot AI**. Ask me anything about the pharma molecule **Sitagliptin**!"
            }
        ]
        st.session_state.question_count = 0
        st.rerun()

    st.markdown("""
    <div style='text-align: center; padding: 20px 0; color: #444; font-size: 11px;'>
        Powered by LLaMA 3 + FAISS<br>
        Research Papers: Sitagliptin Studies
    </div>
    """, unsafe_allow_html=True)


# ── Main Area ────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='color: #00d4ff; margin: 0; font-size: 32px;'>
        💊 PharmaBot AI
    </h1>
    <p style='color: #888; margin: 5px 0 0 0;'>
        Intelligent Pharmaceutical Research Assistant | Sitagliptin Studies
    </p>
</div>
""", unsafe_allow_html=True)

# ── Chat Messages ────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class='user-message'>
                <span style='font-size: 18px;'>👤</span> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='bot-message'>
                <span style='font-size: 18px;'>🤖</span> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Input Area ───────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        label="",
        placeholder="💬 Ask anything about pharma molecule Sitagliptin...",
        key="user_input",
        label_visibility="collapsed"
    )

with col2:
    send_clicked = st.button("Send 🚀")

# ── Handle pending question from sidebar ─────────────────────────
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
    send_clicked = True

# ── Process Input ────────────────────────────────────────────────
if send_clicked and user_input.strip():
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.question_count += 1

    with st.spinner("🔬 Searching research papers..."):
        answer = ask(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()