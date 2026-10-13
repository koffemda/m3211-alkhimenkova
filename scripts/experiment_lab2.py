"""Эксперимент ЛР2: качество маршрутизации на базовом наборе docs/evaluation/lab2-routing.json.

    .venv/bin/python -m scripts.experiment_lab2 --label v1            # macOS / Linux
    .\\.venv\\Scripts\\python.exe -m scripts.experiment_lab2 --label v1  # Windows

Скрипт запускает настоящий MCP-сервер (обнаружение схем), фиксирует время и часовой
пояс из набора и для каждого запроса получает первое решение модели — ровно то, что
бот получил бы от agent.decide. Инструменты не выполняются: для напоминаний
оценивается подготовленный вызов и его аргументы, запись не создаётся.
После изменения одного элемента решения запустите повторно с --label v2.
"""

import argparse
import asyncio
import json
import re
import secrets
from datetime import datetime
from pathlib import Path

from app.__main__ import mcp_server_parameters
from app.agent import CLARIFY_TOOL, Agent
from app.config import ConfigError, Settings
from app.llm import LLMError, OpenAICompatibleClient
from app.mcp_gateway import McpGateway
from app.prompts import MODES

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "docs" / "evaluation" / "lab2-routing.json"
TOOLS = {"get_weather", "get_schedule", "add_reminder"}
TEMPERATURE = 0.0


def normalize(text: str) -> str:
    text = text.casefold().replace("ё", "е").strip().rstrip(".!")
    return re.sub(r"\s*,\s*", ", ", re.sub(r"\s+", " ", text))


def arguments_match(action: str, expected: dict, actual: dict) -> bool:
    """Обязательные аргументы совпадают с ожидаемыми (с нормализацией регистра и формата)."""
    try:
        if action == "get_weather":
            return normalize(actual["city"]) == normalize(expected["city"])
        if action == "get_schedule":
            return actual["date"] == expected["date"]
        if action == "add_reminder":
            same_time = datetime.fromisoformat(actual["remind_at"]) == datetime.fromisoformat(
                expected["remind_at"]
            )
            return same_time and normalize(actual["text"]) == normalize(expected["text"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


async def evaluate(settings: Settings, dataset: dict) -> list[dict]:
    context = dataset["evaluation_context"]
    now = datetime.fromisoformat(context["now"])
    gateway = McpGateway(mcp_server_parameters(settings, secrets.token_hex(16)))
    llm = OpenAICompatibleClient(settings)
    await gateway.start()
    if not gateway.available:
        raise SystemExit("MCP-сервер не запустился — проверьте логи выше.")
    agent = Agent(
        llm,
        gateway,
        repository=None,  # decide() не обращается к БД
        max_messages=settings.history_max_messages,
        max_chars=settings.history_max_chars,
        schedule_timezone=settings.schedule_timezone,
        clock=lambda: now,
    )
    rows = []
    try:
        for case in dataset["cases"]:
            print(f"{case['id']}…", flush=True)
            row = {"id": case["id"], "query": case["query"], "expected": case["expected_action"]}
            try:
                _, result = await agent.decide(
                    MODES["study"], [], case["query"], temperature=TEMPERATURE,
                    timezone=context["timezone"],
                )  # fmt: skip
            except LLMError as exc:
                rows.append({**row, "actual": f"ошибка {type(exc).__name__}", "arguments": {}})
                continue
            if result.tool_calls:
                call = result.tool_calls[0]
                action = "clarify" if call.name == CLARIFY_TOOL else call.name
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": call.arguments}
            else:
                action, arguments = "respond", {}
            row.update(actual=action, arguments=arguments, text=result.text[:300])
            if case["expected_action"] in TOOLS:
                row["arguments_ok"] = action == case["expected_action"] and arguments_match(
                    action, case["expected_arguments"], arguments
                )
            rows.append(row)
    finally:
        await gateway.stop()
        await llm.close()
    return rows


def report(rows: list[dict], meta: dict) -> str:
    correct = sum(row["actual"] == row["expected"] for row in rows)
    tool_rows = [row for row in rows if row["expected"] in TOOLS]
    args_ok = sum(bool(row.get("arguments_ok")) for row in tool_rows)
    routing, argument = correct / len(rows), args_ok / len(tool_rows)
    success = correct >= 10 and args_ok >= 8
    lines = [
        f"# Эксперимент ЛР2: маршрутизация ({meta['label']})",
        "",
        f"- Дата запуска: {meta['date']}",
        f"- Модель: `{meta['model']}`, temperature={TEMPERATURE}, режим /study",
        f"- Время и часовой пояс набора: {meta['now']} ({meta['timezone']})",
        f"- Набор: `docs/evaluation/lab2-routing.json` (версия {meta['version']}), без изменений",
        "",
        "| № | Запрос | Ожидаемое действие | Фактическое действие "
        "| Аргументы корректны | Результат |",
        "|---:|---|---|---|---|---|",
    ]
    for number, row in enumerate(rows, 1):
        ok = row["actual"] == row["expected"]
        if row["expected"] in TOOLS:
            args = "да" if row.get("arguments_ok") else "нет"
        else:
            args = "—"
        shown = json.dumps(row.get("arguments", {}), ensure_ascii=False)
        actual = row["actual"] + (f" `{shown}`" if row.get("arguments") else "")
        lines.append(
            f"| {number} | {row['query']} | {row['expected']} | {actual} | {args} | "
            f"{'верно' if ok and args != 'нет' else 'ошибка'} |"
        )
    lines += [
        "",
        f"routing_accuracy = {correct} / {len(rows)} = {routing:.2f}",
        f"argument_accuracy = {args_ok} / {len(tool_rows)} = {argument:.2f}",
        "",
        f"Критерий успеха (≥ 10/12 и ≥ 8/9): **{'выполнен' if success else 'не выполнен'}**.",
        "",
        "## Разбор ошибок",
        "",
        "_Для каждой ошибочной строки укажите причину: неясное описание инструмента, "
        "неоднозначный запрос, ошибка извлечения аргумента или недостаточная валидация._",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент ЛР2: маршрутизация")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--label", default="v1", help="версия решения, например v1 или v2")
    parser.add_argument("--out", default="docs/experiments")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(exc)
        return 1
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    started = datetime.now().astimezone()
    rows = asyncio.run(evaluate(settings, dataset))
    meta = {
        "label": args.label,
        "date": started.isoformat(timespec="seconds"),
        "model": settings.openrouter_model,
        "now": dataset["evaluation_context"]["now"],
        "timezone": dataset["evaluation_context"]["timezone"],
        "version": dataset["version"],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"lab2-routing-{args.label}-{started:%Y-%m-%d-%H%M}"
    stem.with_suffix(".json").write_text(
        json.dumps({**meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stem.with_suffix(".md").write_text(report(rows, meta), encoding="utf-8")
    print(f"Готово: {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
