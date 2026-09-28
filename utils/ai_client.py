# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Единая точка обращения к ИИ — при смене провайдера/добавлении ретраев
меняется только этот файл.
"""
from __future__ import annotations

import streamlit as st
from openai import OpenAI, APIError, APITimeoutError

from config import REQUEST_TIMEOUT, MAX_OUTPUT_TOKENS


class AIRequestError(Exception):
    """Ошибка обращения к API GPTunneL."""


@st.cache_resource(show_spinner=False)
def _get_client() -> OpenAI:
    """
    Клиент кэшируется как ресурс (не пересоздаётся на каждый rerun).
    Ключ читается ТОЛЬКО из st.secrets — никогда не хардкодьте его в коде.
    """
    api_key = st.secrets["GPTUNNEL_API_KEY"]
    base_url = st.secrets.get("GPTUNNEL_BASE_URL", "https://gptunnel.ru/v1")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)


def analyze_tz_with_ai(tz_text: str, system_prompt: str, model_name: str) -> str:
    """
    Отправляет текст ТЗ выбранной модели и возвращает сырой текстовый ответ
    (ожидается JSON-строка — парсится дальше в excel_generator.py).

    Args:
        tz_text: извлечённый текст технического задания.
        system_prompt: системный промпт (может быть кастомизирован пользователем).
        model_name: id модели в API GPTunneL, например "gpt-4o".

    Returns:
        Сырой текст ответа модели.

    Raises:
        AIRequestError: при сетевой ошибке, таймауте или ошибке API.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tz_text},
            ],
        )
    except APITimeoutError as exc:
        raise AIRequestError("Превышено время ожидания ответа от модели.") from exc
    except APIError as exc:
        raise AIRequestError(f"Ошибка API GPTunneL: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AIRequestError(f"Неизвестная ошибка при обращении к ИИ: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise AIRequestError("Модель вернула пустой ответ.")
    return content