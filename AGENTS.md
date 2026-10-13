# AGENTS.md — правила работы с проектом

Telegram-бот «AI-ассистент студента» (ЛР № 1–2): Python 3.12, aiogram, PostgreSQL,
OpenAI-совместимый LLM API и собственный MCP-сервер.

## Каталоги

| Путь | Назначение |
| --- | --- |
| `app/` | Telegram-бот (MCP-хост): конфигурация, обработчики, сервисы, LLM- и MCP-клиенты |
| `app/handlers/` | Только разбор Telegram-команд и вызов `AssistantService`; без SQL, HTTP и промптов |
| `mcp_server/` | MCP-сервер: инструменты `get_weather`, `get_schedule`, `add_reminder`, ресурс `schedule://current-week` |
| `data/schedule.json` | Обезличенное расписание (без ФИО и персональных данных) |
| `scripts/` | Запуск, деплой, эксперименты (`experiment_lab1.py`, `experiment_lab2.py`), `mcp_check.py` |
| `tests/` | Автотесты; внешние границы подменяются фикстурами из `tests/conftest.py` |
| `docs/` | Методические указания, базовый набор оценки, результаты экспериментов, отчёты |

## Команды

```bash
bash scripts/start.sh --setup-only          # .venv + локальный PostgreSQL (Windows: start.ps1)
.venv/bin/python -m app                      # запуск бота; MCP-сервер стартует дочерним процессом
.venv/bin/python -m pytest -q                # тесты без сети, Telegram и LLM
RUN_INTEGRATION=1 .venv/bin/python -m pytest -m integration -v   # тесты на PostgreSQL
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m scripts.mcp_check        # проверка сервера независимым MCP-клиентом
```

## Миграции и переменные окружения

- Схема БД меняется только добавлением новой миграции в конец `MIGRATIONS` в
  `app/storage.py`. Применённые миграции не редактируются, данные для «исправления»
  схемы не удаляются. Бот применяет миграции при запуске под advisory lock.
- Новая переменная окружения: вымышленный пример в `.env.example`, чтение и проверка в
  `app/config.py` (поле `Settings` называется так же, как переменная, — так его
  передаёт деплой), понятная ошибка без значения секрета, строка в `compose.yaml` для облака.
- Настройки MCP-сервера читает `mcp_server/config.py`; бот передаёт их дочернему процессу явно.

## Архитектурные границы

- `handlers` → `AssistantService` (`app/assistant.py`) → `Agent` (`app/agent.py`),
  `ReminderConfirmations` (`app/confirmations.py`), `Repository` (`app/storage.py`).
- Весь SQL бота — в `app/storage.py`, SQL сервера — в `mcp_server/reminders.py`.
  Только параметры `$1, $2`; данные всегда фильтруются по `chat_id`.
- HTTP к LLM — только `app/llm.py`; к MCP — только `app/mcp_gateway.py`; к погоде —
  только `mcp_server/weather.py` с фиксированными адресами из конфигурации.
- Схемы инструментов объявлены один раз в `mcp_server/schemas.py` (Pydantic,
  `extra="forbid"`). Бот получает их через MCP-обнаружение и не дублирует в коде.
- Модель выбирает инструмент только из обнаруженного списка и `ask_clarification`;
  не более 2 MCP-вызовов на сообщение. Идентификаторы пользователя, `update_id`,
  часовой пояс и подпись владельца берутся из Telegram update и БД, а не от модели.
- `add_reminder` вызывается только из `ReminderConfirmations` после подтверждения и с
  HMAC-подписью владельца в `_meta`; сервер без секрета отклоняет любой вызов.
- Результаты инструментов передаются модели как `role=tool` с пометкой `untrusted_data`.

## Секреты и персональные данные

- Не коммитить `.env`, `.env.cloud`, `.venv`, `.deploy`, дампы БД и журналы.
- Не писать в код, тесты, логи, отчёты и снимки экрана токены, ключи API, пароли,
  адреса прокси с учётными данными, системные промпты в логах, тексты диалогов,
  Telegram ID и содержимое напоминаний. Для корреляции используется случайный
  `request_id`.
- Расписание в репозитории — только обезличенное. Примеры ключей — заведомо вымышленные.
- Секрет, попавший в историю Git, считается раскрытым и перевыпускается.
