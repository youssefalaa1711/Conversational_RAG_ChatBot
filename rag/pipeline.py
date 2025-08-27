# rag/pipeline.py
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_groq import ChatGroq

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _join_docs(docs) -> str:
    if not docs:
        return ""  # <- empty context => behave like a normal chatbot
    return "\n\n".join([d.page_content for d in docs])

def build_chain(
    model_name: str = "llama3-8b-8192",
    persist_directory: str = ".chroma/student-rag",
    collection_name: str = "pdf-chat",
):
    # --- Vector store / retriever
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}  # robust on Windows/CPU
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Helper to know if we actually have any vectors
    def has_docs() -> bool:
        try:
            # LangChain's Chroma wrapper exposes the underlying collection
            return (vectorstore._collection.count() or 0) > 0  # type: ignore[attr-defined]
        except Exception:
            # Fallback: attempt a dummy search and see if anything returns
            try:
                _ = retriever.get_relevant_documents("ping")
                # If no exception, rely on empty results downstream
                return True
            except Exception:
                return False

    # --- LLM
    llm = ChatGroq(model=model_name)

    # --- Prompt (expects: question, context, history)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer when it is non-empty. If context is empty, answer normally without fabricating document citations."),
        MessagesPlaceholder("history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # --- Retrieval step: add context (only if docs exist) and PRESERVE history
    def retrieve_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"] if isinstance(inputs, dict) else str(inputs)
        history = inputs.get("history", []) if isinstance(inputs, dict) else []
        if has_docs():
            docs = retriever.get_relevant_documents(question)
            context = _join_docs(docs)
        else:
            context = ""  # => normal chatbot mode (no retrieval)
        return {
            "question": question,
            "context": context,
            "history": history,  # keep history for the prompt
        }

    rag_chain = (
        RunnableLambda(retrieve_step)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # --- History store
    store: Dict[str, ChatMessageHistory] = {}

    def get_session_history(config: Dict[str, Any]) -> ChatMessageHistory:
        sid = config.get("session_id") if isinstance(config, dict) else str(config)
        if sid not in store:
            store[sid] = ChatMessageHistory()
        return store[sid]

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",   # user’s message key
        history_messages_key="history",  # messages placeholder name in the prompt
    )

    return chain_with_history, store
