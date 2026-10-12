"""Автоматическая проверка формата ответа /study, используемая в эксперименте."""

from scripts.experiment_lab1 import check_format

GOOD = (
    "Кратко: по значению передаётся копия.\n"
    "Объяснение: функция меняет копию, а не оригинал.\n"
    "Пример:\n```cpp\nvoid f(int& x) { ++x; }\n```\n"
    "Проверь себя: что изменится, если убрать &?"
)


def test_good_answer_passes_all_checks():
    assert check_format(GOOD) == {"blocks": True, "code": True, "length": True, "no_headers": True}


def test_wrong_order_missing_code_and_headers_are_detected():
    # Arrange
    text = "# Ответ\nПример: см. ниже\nКратко: копия\nОбъяснение: ...\nПроверь себя: ?"
    # Act
    checks = check_format(text)
    # Assert
    assert checks == {"blocks": False, "code": False, "length": True, "no_headers": False}


def test_too_long_answer_fails_length_but_code_is_not_counted():
    # Arrange
    code = "```python\n" + "x = 1\n" * 400 + "```"
    long_text = GOOD.replace("Объяснение:", "Объяснение: " + "слово " * 260)
    # Act / Assert
    assert check_format(GOOD.replace("```cpp", code + "\n```cpp"))["length"] is True
    assert check_format(long_text)["length"] is False
