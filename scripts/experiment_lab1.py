"""Эксперимент ЛР1: влияние temperature и сравнение исходной/улучшенной инструкции.

Запуск из корня проекта (нужен заполненный .env с ключом LLM):
    .venv/bin/python -m scripts.experiment_lab1            # macOS / Linux
    .\\.venv\\Scripts\\python.exe -m scripts.experiment_lab1  # Windows

Каждый запрос отправляется с пустой историей — так же, как бот после /reset:
системная инструкция режима /study + текст запроса. Меняется только temperature.
Результат: docs/experiments/lab1-<дата>.md (таблицы + полные ответы) и .json.
"""

import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from app.assistant import ALLOWED_TEMPERATURES
from app.config import ConfigError, Settings
from app.context import build_messages
from app.llm import ChatMessage, LLMError, OpenAICompatibleClient
from app.prompts import MODES

QUERY = "Чем передача аргумента по значению отличается от передачи по ссылке в C++?"
RUNS = 3
# Намеренно неполная инструкция: нет аудитории, ограничений и формата.
WEAK_PROMPT = "Ты помощник. Ответь на вопрос."
LABELS = ("Кратко:", "Объяснение:", "Пример:", "Проверь себя:")
MAX_WORDS = 250


def check_format(text: str) -> dict[str, bool]:
    """Проверяемые признаки формата режима /study, названные до эксперимента."""
    positions = [
        match.start() if (match := re.search(rf"^\s*\**{re.escape(label)}", text, re.M)) else -1
        for label in LABELS
    ]
    without_code = re.sub(r"```.*?```", "", text, flags=re.S)
    return {
        "blocks": all(p >= 0 for p in positions) and positions == sorted(positions),
        "code": bool(re.search(r"```[a-zA-Z+#]*\n.*?```", text, re.S)),
        "length": len(without_code.split()) <= MAX_WORDS,
        "no_headers": not re.search(r"^#", text, re.M),
    }


async def run_once(client, messages, temperature) -> dict:
    try:
        result = await client.complete(messages, temperature=temperature)
    except LLMError as exc:
        return {"temperature": temperature, "error": type(exc).__name__}
    checks = check_format(result.text)
    return {
        "temperature": temperature,
        "text": result.text,
        "length": len(result.text),
        "words": len(result.text.split()),
        "format_ok": all(checks.values()),
        "checks": checks,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
    }


def table(rows: list[dict], start: int = 1) -> str:
    lines = [
        "| № | temperature | Длина, симв. | Слов | Формат | Токены (prompt/completion/total) "
        "| finish | Наблюдения |",
        "|---:|---:|---:|---:|---|---|---|---|",
    ]
    for number, row in enumerate(rows, start):
        if "error" in row:
            lines.append(f"| {number} | {row['temperature']:.1f} | — | — | — | — | "
                         f"ошибка {row['error']} | |")  # fmt: skip
            continue
        failed = [name for name, ok in row["checks"].items() if not ok]
        verdict = "да" if row["format_ok"] else "нет (" + ", ".join(failed) + ")"
        tokens = f"{row['prompt_tokens']}/{row['completion_tokens']}/{row['total_tokens']}"
        lines.append(
            f"| {number} | {row['temperature']:.1f} | {row['length']} | {row['words']} | "
            f"{verdict} | {tokens} | {row['finish_reason']} | _заполнить_ |"
        )
    return "\n".join(lines)


def answers(rows: list[dict], title: str) -> str:
    parts = [f"## {title}\n"]
    for number, row in enumerate(rows, 1):
        body = row.get("text") or f"(ошибка {row.get('error')})"
        parts.append(f"### Запуск {number}, temperature {row['temperature']:.1f}\n\n{body}\n")
    return "\n".join(parts)


async def collect(settings: Settings, query: str) -> tuple[list[dict], list[dict]]:
    client = OpenAICompatibleClient(settings)
    try:
        messages = build_messages(MODES["study"], [], query, max_messages=0, max_chars=10**6)
        temperature_rows = []
        for temperature in ALLOWED_TEMPERATURES:
            for run in range(1, RUNS + 1):
                print(f"temperature={temperature:.1f}, запуск {run}/{RUNS}…", flush=True)
                temperature_rows.append(await run_once(client, messages, temperature))
        weak = [ChatMessage("system", WEAK_PROMPT), ChatMessage("user", query)]
        prompt_rows = []
        for prompt_messages in (weak, messages):
            for run in range(1, RUNS + 1):
                print(f"сравнение промптов, запуск {run}/{RUNS}…", flush=True)
                prompt_rows.append(await run_once(client, prompt_messages, 0.3))
    finally:
        await client.close()
    return temperature_rows, prompt_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент ЛР1 с temperature")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--query", default=QUERY)
    parser.add_argument("--out", default="docs/experiments")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(exc)
        return 1
    mode = MODES["study"]
    started = datetime.now().astimezone()
    temperature_rows, prompt_rows = asyncio.run(collect(settings, args.query))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"lab1-{started:%Y-%m-%d-%H%M}"
    meta = {
        "date": started.isoformat(timespec="seconds"),
        "model": settings.openrouter_model,
        "api": settings.openrouter_base_url + "/chat/completions",
        "max_tokens": settings.llm_max_tokens,
        "query": args.query,
        "system_prompt": mode.system_prompt,
        "weak_prompt": WEAK_PROMPT,
    }
    stem.with_suffix(".json").write_text(
        json.dumps({**meta, "temperature_runs": temperature_rows, "prompt_runs": prompt_rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )  # fmt: skip
    ok = {
        t: sum(r.get("format_ok", False) for r in temperature_rows if r["temperature"] == t)
        for t in ALLOWED_TEMPERATURES
    }
    report = f"""# Эксперимент ЛР1: temperature

- Дата: {meta["date"]}
- Модель: `{meta["model"]}`, API: `{meta["api"]}` (Chat Completions)
- Запрос: «{args.query}»
- Режим: /study, история пуста (эквивалент /reset перед каждым запросом)
- Постоянные настройки: max_tokens={meta["max_tokens"]}; top_p, seed и прочие параметры
  не передаются (значения провайдера по умолчанию)
- Признаки хорошего ответа (заданы до запуска): четыре блока в порядке
  {", ".join(LABELS)} (`blocks`); блок кода с указанием языка (`code`);
  не больше {MAX_WORDS} слов вне кода (`length`); без Markdown-заголовков (`no_headers`).

## Активный системный промпт

```text
{mode.system_prompt}
```

## Таблица 12 запусков

{table(temperature_rows)}

Формат соблюдён: {"; ".join(f"T={t:.1f} — {n}/{RUNS}" for t, n in ok.items())}.

## Сравнение инструкций (temperature 0.3)

Исходная инструкция: «{WEAK_PROMPT}». Улучшенная — системный промпт /study выше
(R.C.T.F. + ограничения). Запуски 1–3 — исходная, 4–6 — улучшенная.

{table(prompt_rows)}

{answers(temperature_rows, "Ответы: temperature")}
{answers(prompt_rows, "Ответы: сравнение инструкций")}"""
    stem.with_suffix(".md").write_text(report, encoding="utf-8")
    print(f"Готово: {stem.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
