# app.py
import os
import shutil  # <-- added
import streamlit as st
from dotenv import load_dotenv

from engines.base import ChatEngine
from engines.rag_engine import RagEngine
from rag.ingest import ingest_pdfs
from rag.pipeline import build_chain

# Load .env for GROQ_API_KEY
load_dotenv()

PERSIST_DIR = ".chroma/student-rag"
COLLECTION = "pdf-chat"

st.set_page_config(page_title="Conversational RAG Q&A", page_icon="💬", layout="wide")

# --- NEW: ensure the vector store starts empty on first load this session
if "cleared_on_start" not in st.session_state:
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    st.session_state.cleared_on_start = True

st.title("📚 PDF Conversational Assistant")
st.caption("Upload PDFs and chat with them. Answers are grounded in your documents.")

# -----------------------------
# API KEY HANDLING
# -----------------------------
def ensure_api_key() -> bool:
    key = os.getenv("GROQ_API_KEY")
    if key:
        st.sidebar.success("🔑 GROQ_API_KEY loaded from environment")
        return True

    with st.sidebar:
        st.warning("No GROQ_API_KEY found in .env. Paste it here (kept only in memory).")
        pasted = st.text_input("GROQ API key", type="password")
        if pasted:
            os.environ["GROQ_API_KEY"] = pasted
            st.sidebar.success("API key set for this session")
            return True
    return False

api_ready = ensure_api_key()

# -----------------------------
# SIDEBAR LAYOUT
# -----------------------------
with st.sidebar:
    st.header("⚡ Quick Actions")
    if st.button("Reset Chat"):
        st.session_state.history_store.clear()
        st.session_state.chat_history_ui = []
        st.success("Chat reset!")

    # --- NEW: clear all PDFs (vector store) on demand
    if st.button("🗑️ Clear All PDFs"):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        st.success("All PDFs cleared from the vector store.")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    bubble_color = st.color_picker("Chat bubble color", "#E6F7E6")

# -----------------------------
# SESSION HANDLING
# -----------------------------
session_id = st.text_input("Session ID", placeholder="e.g., demo-1")
if "chat_history_ui" not in st.session_state:
    st.session_state.chat_history_ui = []

# -----------------------------
# PDF UPLOAD & AUTO-INGEST
# -----------------------------
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

def _save_uploads(files):
    paths = []
    if not files:
        return paths
    os.makedirs("uploaded", exist_ok=True)
    for f in files:
        p = os.path.join("uploaded", f.name)
        with open(p, "wb") as out:
            out.write(f.read())
        paths.append(p)
    return paths

if uploaded_files:
    paths = _save_uploads(uploaded_files)
    try:
        ingest_pdfs(paths, persist_directory=PERSIST_DIR, collection_name=COLLECTION)
        st.sidebar.success("📚 PDFs ingested automatically")
    except Exception as e:
        st.sidebar.error(f"Ingestion failed: {e}")

# -----------------------------
# BUILD CHAIN ONCE
# -----------------------------
if "rag_chain" not in st.session_state:
    chain, store = build_chain(
        model_name="llama3-8b-8192",
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )
    st.session_state.rag_chain = chain
    st.session_state.history_store = store
    st.session_state.engine: ChatEngine = RagEngine(chain, store)

# -----------------------------
# CHAT BUBBLE RENDERER
# -----------------------------
def chat_bubble(role, text, color="#E6F7E6"):
    if role == "user":
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:10px; border-radius:10px; margin:5px 0; text-align:right">
                <b>You:</b> {text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#F0F0F0; padding:10px; border-radius:10px; margin:5px 0; text-align:left">
                <b>Bot:</b> {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# MAIN CHAT
# -----------------------------
st.divider()
st.subheader("💬 Chat")

if not session_id:
    st.info("Enter a Session ID to start chatting.")
else:
    user_q = st.chat_input("Ask about your PDFs…")
    if user_q:
        if not api_ready:
            st.error("No API key found. Add GROQ_API_KEY to your .env or sidebar.")
        else:
            with st.spinner("Thinking…"):
                ans = st.session_state.engine.answer(session_id, user_q)

            # Save to UI history
            st.session_state.chat_history_ui.append(("user", user_q))
            st.session_state.chat_history_ui.append(("bot", ans))

    # Render conversation
    for role, msg in st.session_state.chat_history_ui:
        chat_bubble(role, msg, color=bubble_color if role == "user" else "#F0F0F0")
