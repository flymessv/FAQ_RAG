from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM = """Ты — чат-бот технической поддержки компании.
Отвечай ТОЛЬКО на основе контекста из базы знаний (FAQ).
Если ответа нет в контексте — скажи честно и предложи создать тикет.
Не выдумывай факты.
Отвечай кратко и по шагам.
Формат вывода: JSON.
"""

def build_prompt() -> ChatPromptTemplate:
    # history будет подставляться LangChain Memory (RunnableWithMessageHistory)
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("history"),
            (
                "human",
                "Контекст (FAQ):\n{context}\n\n"
                "Вопрос пользователя: {question}\n\n"
                "Верни строго JSON с ключами: "
                "answer (string), action (\"OK\"|\"TICKET\"), confidence (0..1), sources (array of strings).",
            ),
        ]
    )
