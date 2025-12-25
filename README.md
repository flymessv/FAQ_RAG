# Чат-бот техподдержки по FAQ на LangChain (RAG)

## 1) Описание

Проект — чат-бот техподдержки, который отвечает на вопросы **по локальной базе знаний (FAQ)**.
Подход: **RAG (Retrieval-Augmented Generation)**.

Формат интерфейса:
- **Web**: Streamlit (`app.py`)

## 2) Что использовано из LangChain (закрывает требования)

В проекте явно применены ключевые компоненты LangChain:

- **LLM через LangChain**: `ChatOpenAI` (OpenAI API).
- **PromptTemplate**: `ChatPromptTemplate` + `MessagesPlaceholder` (см. `prompts.py`).
- **Chain**: пайплайн `prompt | llm` (RunnableSequence) + разбор ответа (см. `core.py`).
- **Memory (память диалога)**: `RunnableWithMessageHistory` + `InMemoryChatMessageHistory` (см. `core.py`).
- **Tools / Function calling**: `@tool create_support_ticket_tool` + функция `create_support_ticket` (см. `tools.py`).
- **Retrieval (векторный поиск)**: эмбеддинги FAQ и косинусная близость (см. `retriever.py` + `core.py`).
- **Document Loaders**: `TextLoader` (загрузка `.md/.txt`) (см. `ingest.py`).
- **Output parsers**: `JsonOutputParser` (см. `core.py`).

> Важно:
> Retrieval реализован без нативной компиляции: numpy + embeddings.

## 3) Полезная функциональность

- Отвечает на вопросы по локальному FAQ
- Показывает источники (какие куски FAQ использовались)
- Если ответ не найден (низкая релевантность) — предлагает создать тикет
- Пишет тикеты в `data/tickets.jsonl`

## 4) Установка и запуск (Windows, без venv)

### 4.1 Установка зависимостей
Открой PowerShell в папке проекта (там, где `requirements.txt`):

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.2 Конфигурация (.env)
Скопируй файл-пример и вставь ключ:

```powershell
copy .env.example .env
notepad .env
```

В `.env` укажи:
- `OPENAI_API_KEY=sk-...`

### 4.3 Построение индекса (обязательно перед первым запуском)
```powershell
python ingest.py
```

### 4.4 Запуск Web UI
```powershell
python -m streamlit run app.py
```

## 5) Где лежит база знаний (FAQ)

Папка: `kb/`

Поддерживаются `.md` и `.txt`.
После любых изменений в `kb/` заново запускай:

```powershell
python ingest.py
```

## 6) Архитектура проекта

Файлы:
- `app.py` — Streamlit интерфейс
- `ingest.py` — индексация базы знаний (чанки + embeddings → `data/index/`)
- `retriever.py` — загрузка индекса и поиск top-k по косинусной близости
- `core.py` — основная логика: retrieval → LLM chain → JSON parser → решение OK/TICKET
- `prompts.py` — шаблон промпта (PromptTemplate)
- `tools.py` — Tool/Function: создание тикета
- `config.py` — конфигурация и логирование

## 7) Схема обработки запроса (Flow)

1) Пользователь вводит вопрос в UI.
2) Retrieval:
   - считаем embedding вопроса
   - ищем top_k похожих чанков FAQ
3) Сбор контекста + ограничение длины контекста.
4) LLM (Chain):
   - `ChatPromptTemplate(history + context + question)` → `ChatOpenAI`
   - Memory: история диалога подставляется через `RunnableWithMessageHistory`
5) Output parsing:
   - `JsonOutputParser` пытается распарсить JSON
6) Если `top_score < MIN_SIM` → `action=TICKET`, показываем предложение создать тикет
7) Tool:
   - по кнопке создаём тикет в `data/tickets.jsonl`

## 8) Примеры использования

1) **Сброс пароля**
   - Вопрос: “Как сбросить пароль?”
   - Ожидаемо: `action=OK`, ответ из FAQ.

2) **Не приходит письмо**
   - Вопрос: “Почему не приходит письмо для подтверждения?”
   - Ожидаемо: `action=OK`, совет проверить спам/домен/задержки.

3) **Возврат средств**
   - Вопрос: “Как оформить возврат?”
   - Ожидаемо: `action=OK`, шаги и условия.

4) **Вопрос вне FAQ**
   - Вопрос: “Сделайте интеграцию с моей CRM X”
   - Ожидаемо: `action=TICKET` + кнопка создания тикета.

5) **Уточнение с памятью**
   - Вопрос 1: “Как сбросить пароль?”
   - Вопрос 2: “А если письмо не приходит?”
   - Ожидаемо: второй ответ учитывает контекст диалога (Memory).

6) **Некорректный запрос**
   - Вопрос: пустая строка
   - Ожидаемо: сообщение “похоже, сообщение пустое…”

## 9) Логи и обработка ошибок

- Логи идут в консоль (`logging`), основные точки:
  - загрузка KB
  - построение индекса
  - top_score retrieval
  - ошибки LLM/API

Обрабатываются ситуации:
- пустой запрос
- отсутствует индекс (не запускали `ingest.py`)
- ошибка OpenAI API (показываем текст ошибки)
- невалидный JSON от модели (возвращаем текст как есть)

## 10) Настройки “дёшево и сердито” (бюджет $5)

По умолчанию:
- `MODEL_NAME=gpt-4o-mini`
- `EMBED_MODEL=text-embedding-3-small`
- `TOP_K=3`
- `MAX_OUTPUT_TOKENS=300`
- `MAX_CONTEXT_CHARS=6000`

Если хочешь ещё дешевле — уменьши:
- `TOP_K=2`
- `MAX_OUTPUT_TOKENS=200`
- `MAX_CONTEXT_CHARS=4000`
