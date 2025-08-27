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
        # Allow chatting even without PDFs
        return "No documents available."
    return "\n\n".join([d.page_content for d in docs])

def build_chain(
    model_name: str = "llama3-8b-8192",
    persist_directory: str = ".chroma/student-rag",
    collection_name: str = "pdf-chat",
):
    # --- Vector store / retriever
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"}  # <-- force CPU to avoid meta tensor error
        ),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # --- LLM
    llm = ChatGroq(model=model_name)

    # --- Prompt (expects: question, context, history)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer."),
        MessagesPlaceholder("history"),
        ("system", "Context:\n{context}"),
        ("human", "{question}"),
    ])

    # --- Retrieval step: add context AND PRESERVE history
    def retrieve_step(inputs: Dict[str, Any]) -> Dict[str, Any]:
        # inputs will already include {"question": ..., "history": [...]} thanks to RunnableWithMessageHistory
        question = inputs["question"] if isinstance(inputs, dict) else str(inputs)
        history = inputs.get("history", []) if isinstance(inputs, dict) else []
        docs = retriever.get_relevant_documents(question)
        return {
            "question": question,
            "context": _join_docs(docs),
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
