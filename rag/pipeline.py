# rag/pipeline.py
from typing import Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings  # <- no Torch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_groq import ChatGroq

def _join_docs(docs) -> str:
    """Join retrieved docs into a single context string.
    Empty string => behave like a normal chatbot (no retrieval)."""
    if not docs:
        return ""
    return "\n\n".join([d.page_content for d in docs])

def build_chain(
    model_name: str = "llama-3.1-8b-instant",
    persist_directory: str = ".chroma/student-rag",
    collection_name: str = "pdf-chat",
):
    # --- Vector store / retriever (Torch-free embeddings)
    embeddings = FastEmbedEmbeddings()  # fast, CPU-only, no meta-tensor issues
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Helper: do we actually have any vectors?
    def has_docs() -> bool:
        try:
            count = vectorstore._collection.count()  # private attr but reliable
            return bool(count and count > 0)
        except Exception:
            # If we can't read the count, default to NO docs to avoid accidental retrieval
            return False

    # --- LLM
    llm = ChatGroq(model=model_name)

    # --- Prompt (expects: question, context, history)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer when it is non-empty. "
                   "If context is empty, answer normally without fabricating document citations."),
        MessagesPlaceholder("history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # --- Retrieval step: add context (only if docs exist) and PRESERVE history
    def retrieve_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # RunnableWithMessageHistory supplies {"question": ..., "history": [...]}
        question = inputs["question"] if isinstance(inputs, dict) else str(inputs)
        history = inputs.get("history", []) if isinstance(inputs, dict) else []
        if has_docs():
            docs = retriever.get_relevant_documents(question)
            context = _join_docs(docs)
        else:
            context = ""  # => normal chatbot mode (no retrieval)
        return {"question": question, "context": context, "history": history}

    rag_chain = RunnableLambda(retrieve_step) | qa_prompt | llm | StrOutputParser()

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
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, store
