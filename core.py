from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings, get_logger
from retriever import load_retriever
from prompts import build_prompt
from tools import create_support_ticket, create_support_ticket_tool

log = get_logger("faq_bot")

# In-process Memory store (LangChain)
_HISTORY_STORE: dict[str, InMemoryChatMessageHistory] = {}

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _HISTORY_STORE:
        _HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
    return _HISTORY_STORE[session_id]

def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else (s[:max_chars] + "\n...[truncated]")

#embending вопроса
def retrieve_with_scores(question: str):
    retriever = load_retriever()
    emb = OpenAIEmbeddings(api_key=settings.openai_api_key, model=settings.embed_model)

    qvec = emb.embed_query(question)
    pairs = retriever.similarity_search(query_vec=qvec, k=settings.top_k)

    scored = []
    ctx_blocks = []
    for doc, score in pairs:
        src = doc.metadata.get("source") or "kb"
        preview = doc.page_content.strip().replace("\n", " ")
        preview = preview[:200] + ("..." if len(preview) > 200 else "")
        scored.append((f"{src}: {preview}", score))
        ctx_blocks.append(f"[score={score:.3f}]\n{doc.page_content.strip()}")

    #top-k чанков
    top_score = scored[0][1] if scored else 0.0
    context = "\n\n---\n\n".join(ctx_blocks)
    context = _truncate(context, settings.max_context_chars)
    return scored, top_score, context

def build_llm_chain():
    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=0.2,
        max_tokens=settings.max_output_tokens,
    )
    prompt = build_prompt()
    return prompt | llm

def answer_question(question: str, session_id: str = "default") -> Dict[str, Any]:
    if not question or not question.strip():
        return {"answer": "Похоже, сообщение пустое. Напишите вопрос по FAQ.", "action": "OK", "confidence": 0.0, "sources": []}

    if not settings.openai_api_key:
        return {"answer": "Не найден OPENAI_API_KEY. Заполните .env.", "action": "OK", "confidence": 0.0, "sources": []}

    scored, top_score, context = retrieve_with_scores(question)
    log.info(f"Retrieval top_score={top_score:.3f} top_k={settings.top_k}")

    forced_action = "OK" if top_score >= settings.min_sim else "TICKET"

    base = build_llm_chain()
    with_memory = RunnableWithMessageHistory(
        base,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    try:
        ai_msg = with_memory.invoke(
            {"question": question, "context": context or "(пусто)"},
            config={"configurable": {"session_id": session_id}},
        )
        raw = getattr(ai_msg, "content", "")
    except Exception as e:
        log.exception("LLM call failed")
        return {"answer": f"Ошибка при обращении к модели: {e}", "action": "OK", "confidence": 0.0, "sources": []}

    parser = JsonOutputParser()
    try:
        out = parser.parse(raw)
    except Exception:
        out = {"answer": raw.strip() or "Не удалось распарсить ответ модели.", "action": forced_action, "confidence": 0.0, "sources": []}

    answer = str(out.get("answer", "")).strip()
    action = str(out.get("action", forced_action)).strip().upper()
    confidence = float(out.get("confidence", 0.0) or 0.0)
    sources = out.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    if forced_action == "TICKET":
        action = "TICKET"
        if "тикет" not in answer.lower():
            answer = (answer + "\n\n"
                      "Я не нашёл точного ответа в базе знаний. Могу создать тикет в поддержку (кнопка ниже).").strip()

    if not sources:
        sources = [s for s, _ in scored[:settings.top_k]]

    return {
        "answer": answer,
        "action": action,
        "confidence": max(0.0, min(1.0, confidence)),
        "sources": sources,
        "top_score": top_score,
    }

def create_ticket(question: str, contact: str | None = None) -> str:
    _ = create_support_ticket_tool.run({"user_question": question, "contact": contact or ""})
    return create_support_ticket(user_question=question, contact=contact)
