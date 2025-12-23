import uuid
import streamlit as st

from core import answer_question, create_ticket
from config import settings

st.set_page_config(page_title="FAQ Support Bot", page_icon="🛠", layout="centered")

st.title("🛠 FAQ Support Bot")
st.caption("RAG по локальному FAQ + память диалога + тикеты. (LangChain + OpenAI)")

# -----------------------------
# Session management (UI-level)
# -----------------------------
def new_session_id() -> str:
    return "S-" + uuid.uuid4().hex[:8]

if "sessions" not in st.session_state:
    # sessions: dict[session_id] = list[("user"|"assistant", "text")]
    st.session_state.sessions = {}

if "current_session" not in st.session_state:
    sid = new_session_id()
    st.session_state.sessions[sid] = []
    st.session_state.current_session = sid

# Sidebar: select session, create/delete
with st.sidebar:
    st.header("Сессии")

    session_ids = list(st.session_state.sessions.keys())
    if st.session_state.current_session not in st.session_state.sessions:
        # safety fallback
        sid = new_session_id()
        st.session_state.sessions[sid] = []
        st.session_state.current_session = sid

    st.session_state.current_session = st.selectbox(
        "Активная сессия",
        options=session_ids,
        index=session_ids.index(st.session_state.current_session) if session_ids else 0,
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Новая", use_container_width=True):
            sid = new_session_id()
            st.session_state.sessions[sid] = []
            st.session_state.current_session = sid
            st.rerun()

    with colB:
        if st.button("🗑 Удалить", use_container_width=True, disabled=len(session_ids) <= 1):
            sid = st.session_state.current_session
            st.session_state.sessions.pop(sid, None)
            # switch to any remaining
            st.session_state.current_session = list(st.session_state.sessions.keys())[0]
            st.rerun()

    st.divider()
    sid = st.session_state.current_session
    msg_count = len(st.session_state.sessions.get(sid, []))
    st.write(f"**Текущая сессия:** `{sid}`")
    st.write(f"**Сообщений в ней:** {msg_count}")

    st.info(
        "Память хранится **внутри выбранной сессии**. "
        "Она живёт пока запущен Streamlit. "
    )

# -----------------------------
# Main chat
# -----------------------------
sid = st.session_state.current_session
history = st.session_state.sessions[sid]

# Render existing history
for role, content in history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)

# Input
q = st.chat_input("Напишите вопрос по FAQ…")

if q is not None:
    q = q.strip()
    if not q:
        # empty input case (requirement 4.3)
        with st.chat_message("assistant"):
            st.warning("Похоже, сообщение пустое. Напишите вопрос по FAQ.")
    else:
        # 1) Show user message immediately
        history.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        # 2) Assistant bubble + visible thinking indicator
        with st.chat_message("assistant"):
            with st.spinner("Думаю… ищу ответ в базе знаний и формирую ответ"):
                res = answer_question(q, session_id=sid)

            ans = res.get("answer", "").strip() or "Не удалось сформировать ответ."
            st.markdown(ans)

            # Show retrieval debug (super useful for demo)
            with st.expander("Источники (Retrieval)"):
                st.write(f"top_score: {res.get('top_score', 0.0):.3f} | threshold: {settings.min_sim}")
                for s in res.get("sources", []):
                    st.write("- " + str(s))


            # Ticket UI (tool/function demo)
            if str(res.get("action", "")).upper() == "TICKET":
                st.warning("Похоже, в FAQ нет точного ответа. Можно создать тикет в поддержку.")
                contact = st.text_input("Контакт для связи (опционально):", key=f"contact_{sid}")
                if st.button("Создать тикет", type="primary", key=f"ticket_{sid}"):
                    tid = create_ticket(q, contact or None)
                    st.success(f"Тикет создан: {tid} (сохранён в data/tickets.jsonl)")

        # 3) Save assistant message to history
        history.append(("assistant", ans))

        # persist updated history back (explicitly)
        st.session_state.sessions[sid] = history
