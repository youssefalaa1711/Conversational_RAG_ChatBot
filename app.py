# app.py 
import os
import uuid
import shutil
import streamlit as st
from dotenv import load_dotenv

from engines.base import ChatEngine
from engines.rag_engine import RagEngine
from rag.ingest import ingest_pdfs
from rag.pipeline import build_chain
from langchain_groq import ChatGroq  # for auto-titling

# -----------------------------
# ENV / CONSTANTS
# -----------------------------
load_dotenv()
BASE_CHROMA_DIR = ".chroma"            # base dir for all chats (per-chat subfolders)
DEFAULT_COLLECTION = "pdf-chat"        # same name OK; folders isolate per chat
TITLE_MODEL = "llama3-8b-8192"

st.set_page_config(page_title="Conversational RAG Q&A", page_icon="💬", layout="wide")

# ✅ default user bubble color (used at render)
bubble_color = "#E6F7E6"

# -----------------------------
# Minimal CSS to resemble ChatGPT sidebar
# -----------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] > div { padding-top: .4rem !important; }
.app-brand { font-weight:700; font-size:1.1rem; margin:.25rem .25rem .5rem .25rem; }
.chat-list { height: 48vh; overflow-y: auto; padding-right:.25rem; }
.chat-item { border-radius:8px; padding:.5rem .6rem; margin-bottom:.15rem; cursor:pointer; }
.chat-item:hover { background:#f5f5f5; }
.chat-item.active { background:#e9f0ff; border:1px solid #cfe0ff; }
.chat-title { font-weight:600; font-size:.95rem; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.small { font-size:.8rem; color:#666; }
hr { margin: .6rem 0 .6rem 0; }
.tag { display:inline-block; padding:.18rem .45rem; border-radius:999px; border:1px solid #e6e6e6; margin:.1rem .25rem .1rem 0; font-size:.78rem; color:#555; background:#fafafa; }
</style>
""", unsafe_allow_html=True)

st.title("💬 Conversational RAG Q&A")
st.caption("Multiple chats, each with its own memory and PDFs. Upload files per chat and ask away.")

# -----------------------------
# API KEY
# -----------------------------
def api_key_ready() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))

def maybe_collect_api_key_ui():
    # Only show if missing, and not at the top of the sidebar
    with st.sidebar.expander("Settings", expanded=False):
        if not api_key_ready():
            st.warning("No GROQ_API_KEY found in .env. Paste it here (kept only in memory).")
            pasted = st.text_input("Groq API key", type="password")
            if pasted:
                os.environ["GROQ_API_KEY"] = pasted
                st.success("Key set for this session. You can close this section.")

maybe_collect_api_key_ui()

# -----------------------------
# SESSION STATE INIT
# -----------------------------
def _init_state():
    ss = st.session_state
    if "chat_list" not in ss:
        ss.chat_list = [{"id": f"chat-{uuid.uuid4().hex[:6]}", "title": "New chat"}]
    if "active_chat_id" not in ss:
        ss.active_chat_id = ss.chat_list[0]["id"]
    if "engines" not in ss:
        ss.engines = {}
    if "stores" not in ss:
        ss.stores = {}
    if "ui_histories" not in ss:
        ss.ui_histories = {}
    if "docs_loaded" not in ss:
        ss.docs_loaded = {}
    if "titled" not in ss:
        ss.titled = {}
    if "uploader_version" not in ss:
        ss.uploader_version = {}   # <-- added here

_init_state()

# -----------------------------
# CHAT HELPERS
# -----------------------------
def chat_chroma_dir(chat_id: str) -> str:
    d = os.path.join(BASE_CHROMA_DIR, chat_id)
    os.makedirs(d, exist_ok=True)
    return d

def ensure_engine_for_chat(chat_id: str) -> ChatEngine:
    ss = st.session_state
    if chat_id in ss.engines:
        return ss.engines[chat_id]
    persist_dir = chat_chroma_dir(chat_id)
    chain, store = build_chain(
        model_name="llama3-8b-8192",
        persist_directory=persist_dir,
        collection_name=DEFAULT_COLLECTION,
    )
    engine: ChatEngine = RagEngine(chain, store)
    ss.engines[chat_id] = engine
    ss.stores[chat_id] = store
    ss.ui_histories.setdefault(chat_id, [])
    ss.docs_loaded[chat_id] = any(os.scandir(persist_dir))
    ss.titled.setdefault(chat_id, False)
    return engine

def reset_chat(chat_id: str):
    ss = st.session_state
    store = ss.stores.get(chat_id)
    if store is not None:
        store.clear()
    ss.ui_histories[chat_id] = []
    ss.titled[chat_id] = False

def clear_pdfs(chat_id: str):
    ss = st.session_state

    # 1) Wipe persisted Chroma collection for this chat
    persist_dir = os.path.join(BASE_CHROMA_DIR, chat_id)
    shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    # 2) Wipe uploaded source files (the ones listed under the expander)
    upload_dir = os.path.join("uploaded", chat_id)
    shutil.rmtree(upload_dir, ignore_errors=True)

    # 3) Clear conversation history (so no PDF-derived memory remains)
    store = ss.stores.get(chat_id)
    if store is not None:
        try:
            store.clear()
        except Exception:
            pass
    ss.ui_histories[chat_id] = []
    ss.titled[chat_id] = False  # optional: allow a new auto-title

    # 4) Drop live engine & store so retriever is recreated against empty DB
    ss.engines.pop(chat_id, None)
    ss.stores.pop(chat_id, None)

    # 5) Flags & notices
    ss.docs_loaded[chat_id] = False
    if "uploaded_notice" in ss:
        ss.uploaded_notice.pop(chat_id, None)

    # 6) Reset the file_uploader widget by bumping its version
    ss.uploader_version[chat_id] = ss.uploader_version.get(chat_id, 0) + 1


def create_new_chat():
    ss = st.session_state
    new_id = f"chat-{uuid.uuid4().hex[:6]}"
    ss.chat_list.append({"id": new_id, "title": "New chat"})
    ss.active_chat_id = new_id

def rename_chat(chat_id: str, new_title: str):
    for c in st.session_state.chat_list:
        if c["id"] == chat_id:
            c["title"] = new_title
            break

def delete_chat(chat_id: str):
    ss = st.session_state
    if len(ss.chat_list) == 1:
        st.warning("At least one chat is required.")
        return
    ss.chat_list = [c for c in ss.chat_list if c["id"] != chat_id]
    ss.ui_histories.pop(chat_id, None)
    ss.engines.pop(chat_id, None)
    ss.stores.pop(chat_id, None)
    shutil.rmtree(chat_chroma_dir(chat_id), ignore_errors=True)
    ss.active_chat_id = ss.chat_list[0]["id"]

def generate_title(first_user: str, first_bot: str) -> str:
    """Use the LLM to auto-title the chat in 3–6 words."""
    try:
        if not api_key_ready():
            # fallback: simple heuristic
            text = (first_user or first_bot or "New chat").strip()
            return (text[:42] + "…") if len(text) > 45 else text
        llm = ChatGroq(model=TITLE_MODEL)
        prompt = (
            "Create a short, 3–6 word title for this conversation topic. "
            "Do not use quotes, punctuation at the end, or emojis. "
            f"\nUser: {first_user}\nAssistant: {first_bot}\nTitle:"
        )
        out = llm.invoke(prompt).content.strip()
        out = out.replace('"', '').replace("'", "")
        return out[:60] or "New chat"
    except Exception:
        text = (first_user or first_bot or "New chat").strip()
        return (text[:42] + "…") if len(text) > 45 else text

# -----------------------------
# SIDEBAR (ChatGPT-like)
# -----------------------------
with st.sidebar:
    st.markdown('<div class="app-brand">Chats</div>', unsafe_allow_html=True)

    # Search
    q = st.text_input("Search chats", value="", label_visibility="collapsed", placeholder="Search chats…")
    filtered = [c for c in st.session_state.chat_list if q.lower() in c["title"].lower()]

    # List
    st.markdown('<div class="chat-list">', unsafe_allow_html=True)
    for c in filtered:
        active = (c["id"] == st.session_state.active_chat_id)
        if st.button(f"{'• ' if active else ''}{c['title']}", key=f"sel-{c['id']}", use_container_width=True):
            st.session_state.active_chat_id = c["id"]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # New / rename / delete
    cols = st.columns([1, 1, 1])
    if cols[0].button("➕ New"):
        create_new_chat(); st.rerun()
    with st.expander("✏️ Rename", expanded=False):
        current = next(c for c in st.session_state.chat_list if c["id"] == st.session_state.active_chat_id)
        new_name = st.text_input("New name", value=current["title"])
        if st.button("Save"):
            rename_chat(st.session_state.active_chat_id, new_name); st.success("Renamed.")
    if cols[2].button("🗑️ Delete"):
        delete_chat(st.session_state.active_chat_id); st.rerun()

    st.markdown("---")
    st.caption("This chat")
    if st.button("Reset chat history"):
        reset_chat(st.session_state.active_chat_id); st.success("History cleared.")
    if st.button("Clear PDFs"):
        clear_pdfs(st.session_state.active_chat_id)
        st.success("PDFs and chat memory cleared for this chat.")
        st.rerun()  # force UI to rebuild (uploader will show blank; tags disappear)



# -----------------------------
# ACTIVE CHAT (engine + state)
# -----------------------------
active_id = st.session_state.active_chat_id
engine = ensure_engine_for_chat(active_id)

# -----------------------------
# 📂 PDF UPLOAD — MINIMAL, PROFESSIONAL (collapsed by default)
# -----------------------------
# 📂 PDF UPLOAD — MINIMAL, PROFESSIONAL (collapsed by default)
with st.expander("➕ Upload PDFs (optional)", expanded=False):
    up_ver = st.session_state.uploader_version.get(active_id, 0)
    uploaded_files = st.file_uploader(
        "Select PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader-{active_id}-{up_ver}",  # versioned key
    )


def _save_uploads(chat_id: str, files):
    paths = []
    if not files: return paths
    upload_dir = os.path.join("uploaded", chat_id)
    os.makedirs(upload_dir, exist_ok=True)
    for f in files:
        p = os.path.join(upload_dir, f.name)
        with open(p, "wb") as out:
            out.write(f.read())
        paths.append(p)
    return paths

if 'uploaded_notice' not in st.session_state:
    st.session_state.uploaded_notice = {}

if uploaded_files:
    saved = _save_uploads(active_id, uploaded_files)
    try:
        persist_dir = chat_chroma_dir(active_id)
        ingest_pdfs(saved, persist_directory=persist_dir, collection_name=DEFAULT_COLLECTION)
        st.session_state.docs_loaded[active_id] = True
        st.session_state.engines.pop(active_id, None)
        st.session_state.stores.pop(active_id, None)
        engine = ensure_engine_for_chat(active_id)
        st.session_state.uploaded_notice[active_id] = f"Ingested {len(saved)} file(s)."
    except Exception as e:
        st.session_state.uploaded_notice[active_id] = f"Ingestion failed: {e}"

# Subtle one-line files list just under the expander (if any)
chat_upload_dir = os.path.join("uploaded", active_id)
if os.path.isdir(chat_upload_dir):
    files = [f.name for f in os.scandir(chat_upload_dir) if f.is_file()]
    if files:
        st.caption("Files: " + " ".join(f"<span class='tag'>{name}</span>" for name in files), unsafe_allow_html=True)

# Optional small notice after upload
note = st.session_state.uploaded_notice.get(active_id)
if note:
    st.caption(note)

st.markdown("---")

# -----------------------------
# Chat bubbles
# -----------------------------
def chat_bubble(role, text, color="#E6F7E6"):
    if role == "user":
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:10px; border-radius:10px; margin:6px 0; text-align:right">
                <b>You:</b> {text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#F0F0F0; padding:10px; border-radius:10px; margin:6px 0; text-align:left">
                <b>Bot:</b> {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------
# MAIN CHAT
# -----------------------------
st.subheader("💬 Chat")
input_hint = "Ask anything… (uses this chat’s PDFs if available)" if st.session_state.docs_loaded.get(active_id, False) \
    else "Ask anything… (no PDFs loaded for this chat yet)"
user_q = st.chat_input(input_hint, key=f"chatinput-{active_id}")

if user_q:
    if not api_key_ready():
        st.error("No API key found. Add GROQ_API_KEY to your .env or set it in Settings.")
    else:
        with st.spinner("Thinking…"):
            # use chat_id as the session id for LC memory
            ans = engine.answer(active_id, user_q)

        # save UI transcript
        st.session_state.ui_histories.setdefault(active_id, []).append(("user", user_q))
        st.session_state.ui_histories.setdefault(active_id, []).append(("bot", ans))

        # auto-title on first exchange if still "New chat"
        chat_meta = next(c for c in st.session_state.chat_list if c["id"] == active_id)
        if not st.session_state.titled.get(active_id, False):
            first_user = user_q
            first_bot = ans
            new_title = generate_title(first_user, first_bot).strip() or "New chat"
            chat_meta["title"] = new_title
            st.session_state.titled[active_id] = True

# Render conversation
for role, msg in st.session_state.ui_histories.get(active_id, []):
    chat_bubble(role, msg, color=bubble_color if role == "user" else "#F0F0F0")
