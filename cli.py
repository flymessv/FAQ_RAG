from core import answer_question

def main():
    history = []
    print("FAQ Support Bot (CLI). Пустая строка — выход.")
    while True:
        q = input("> ").strip()
        if not q:
            break
        history.append(("user", q))
        res = answer_question(q, history)
        print(res["answer"])
        history.append(("assistant", res["answer"]))

if __name__ == "__main__":
    main()
