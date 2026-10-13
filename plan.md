# План ЛР № 2

Отметки: [x] — выполнено и покрыто тестами, [ ] — выполняется студентом вручную.

## Этап 0. Подготовка

- [x] Ветка `lab2` от завершённой `lab1`; ЛР1 не ломается (тесты ЛР1 проходят без изменений логики).
- [x] Зависимости: `mcp==2.3.0` (официальный Python SDK), `jsonschema`, `tzdata` (зоны
  IANA на Windows); версии закреплены в `constraints.txt`, wheel-пакеты проверены для
  Windows, macOS и Linux x86-64.
- [x] Обезличенное расписание `data/schedule.json` на 4 недели (05.10–01.11.2026).
- [x] `AGENTS.md`, `spec.md`, этот план.

## Этап 1. MCP-сервер (`mcp_server/`)

| Требование | Реализация | Тесты |
| --- | --- | --- |
| [x] Три инструмента с `inputSchema`/`outputSchema`, `structuredContent` | `schemas.py`, `server.py` (низкоуровневый `Server`) | `test_discovery_lists_three_tools_with_strict_schemas_and_resource` |
| [x] `get_weather`: геокодинг, неоднозначность, таймаут, ошибки | `weather.py` (Open-Meteo) | `test_weather.py` (сценарии 1–3) |
| [x] `get_schedule`: ISO-дата, пустой список, источник из конфигурации | `schedule.py` | `test_schedule_*` (сценарии 4–6) |
| [x] `add_reminder`: доверенный владелец, будущее время, идемпотентность | `trust.py`, `reminders.py` | `test_reminder_*`, `test_same_idempotency_key_*` (7–9) |
| [x] Ресурс `schedule://current-week` | `schedule.py` | `test_week_resource_on_month_boundary`, `…_year_boundary…` (12) |
| [x] Отклонение неизвестного инструмента и лишних полей | `server.py` | `test_unknown_tool_is_rejected`, `test_extra_arguments_are_rejected` (14) |

## Этап 2. Схема БД

- [x] Миграция 2: `user_settings.timezone`, `reminders`, `pending_actions`
  (`UNIQUE (chat_id, source_update_id)`), `agent_events`. Тест: `test_migrations_are_idempotent`.

## Этап 3. MCP-клиент и агент (`app/`)

| Требование | Реализация | Тесты |
| --- | --- | --- |
| [x] Подключение при запуске, обнаружение, работа без сервера | `mcp_gateway.py` (супервизор, переподключение) | `test_bot_works_without_mcp_server` |
| [x] Нативный tool calling, схемы только из MCP | `llm.py`, `agent.py` | `test_model_receives_schemas_discovered_via_mcp` |
| [x] Три решения: ответ, уточнение, вызов | `agent.py` (`ask_clarification`) | `test_plain_answer_*`, `test_clarification_*`, `test_weather_question_*` |
| [x] Не более двух MCP-вызовов | `MAX_MCP_CALLS` | `test_agent_stops_after_two_mcp_calls` |
| [x] Проверка аргументов до вызова | `McpGateway.validate` (jsonschema по обнаруженной схеме) | `test_extra_arguments_are_rejected_before_mcp` |
| [x] Результат — данные, а не инструкции | `role=tool`, `untrusted_data` | `test_instruction_like_tool_data_stays_data` |

## Этап 4. Напоминания, часовой пояс, подтверждение

- [x] `/timezone` с проверкой IANA — `test_timezone_is_saved_*`, `test_unknown_timezone_is_rejected` (13).
- [x] Локальное время только в зоне пользователя; DST — `test_timeutil.py`,
  `test_reminder_arguments_are_validated_before_preparation`.
- [x] Подтверждение 5 минут, отмена, чужой пользователь — `test_reminder_is_created_only_after_confirmation`,
  `test_cancel_then_confirm`, `test_confirmation_expires_*`, `test_other_user_*` (11).
- [x] Повтор update и конкурентное подтверждение на PostgreSQL — `test_lab2_integration.py` (10).

## Этап 5. Команды и аудит

- [x] `/tools`, `/why`, `/timezone`; `/start` перечисляет их — `test_tools_command_*`, `test_why_*`.
- [x] Аудит каждого запроса в `agent_events`; в текстовом логе нет Telegram ID и текстов.

## Этап 6. Проверка и эксперимент

- [x] Независимый клиент: `scripts/mcp_check.py` (без секрета `add_reminder` отклоняется).
- [ ] MCP Inspector на тестовой БД — выполняет студент (команда в README).
- [ ] Прогон базового набора `scripts/experiment_lab2.py --label v1` на реальной модели.
- [ ] Изменить один элемент (описание инструмента или правило валидации), повторить ошибочные
  запросы `--label v2`, сохранить обе версии.
- [ ] Тег `lab2`.

## Отклонения от первоначального плана

1. **Низкоуровневый `Server` вместо `MCPServer`.** Высокоуровневый `MCPServer` из SDK 2.3
   молча отбрасывает лишние аргументы, а по ТЗ они должны отклоняться. Поэтому схемы
   задаются явно из Pydantic-моделей с `extra="forbid"`.
2. **Уточнение как локальный инструмент `ask_clarification`.** Вариант «модель пишет
   вопрос текстом» отклонён: тогда уточнение нельзя отличить от ответа без разбора текста.
3. **Неоднозначность города.** Первый вариант «спрашивать при любых одноимённых
   результатах» превращал «Москву» в неоднозначный запрос из-за деревень с тем же
   названием. Добавлено правило доминирования по населению (≥ 100 тыс. и ≥ 10×);
   сопоставимые города по-прежнему требуют выбора.
4. **Секрет доверия генерируется при каждом запуске**, а не хранится в `.env`: его не
   нужно настраивать, и он не может утечь через конфигурацию.
