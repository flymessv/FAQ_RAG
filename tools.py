import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

TICKETS_PATH = Path("data/tickets.jsonl")
TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_support_ticket(user_question: str, contact: Optional[str] = None) -> str:
    ticket = {
        "id": f"T-{int(datetime.utcnow().timestamp())}",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "question": user_question,
        "contact": contact,
        "status": "open",
    }
    with TICKETS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return ticket["id"]


@tool("create_support_ticket", description="Создаёт тикет в поддержку и возвращает ID тикета.")
def create_support_ticket_tool(user_question: str, contact: str = "") -> str:
    return create_support_ticket(user_question=user_question, contact=contact or None)

