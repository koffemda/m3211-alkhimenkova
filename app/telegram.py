from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession

from app.config import Settings


def create_bot(settings: Settings) -> Bot:
    # Одна сессия для getMe, polling и ответов; прямого fallback нет.
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None, timeout=40)
    return Bot(token=settings.bot_token, session=session)


TELEGRAM_TEXT_LIMIT = 4096


def utf16_length(text: str) -> int:
    # Telegram считает длину в единицах UTF-16: эмодзи вне BMP занимают две.
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Делит текст на части не длиннее limit без потери символов и с сохранением порядка.

    Разрез ставится после последнего перевода строки или пробела во второй половине
    части; если их нет — по жёсткой границе. Части только из пробелов не отправляются.
    """
    parts = []
    rest = text
    while utf16_length(rest) > limit:
        units = cut = 0
        for index, char in enumerate(rest):
            units += 2 if ord(char) > 0xFFFF else 1
            if units > limit:
                break
            cut = index + 1
        for separator in ("\n", " "):
            position = rest.rfind(separator, 0, cut)
            if position >= cut // 2:
                cut = position + 1
                break
        parts.append(rest[:cut])
        rest = rest[cut:]
    parts.append(rest)
    return [part for part in parts if part.strip()]
