# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Единая точка обращения к ИИ. Реализует безопасный агентский цикл с самоисправлением (Self-Correction).
"""
from __future__ import annotations

import json
import re
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


def _validate_json_structure(raw_content: str) -> tuple[bool, str]:
    """
    Внутренний валидатор структуры ответа модели, повторяющий конвейер excel_generator.
    Проверяет наличие JSON, валидность синтаксиса и корневой ключ 'items'.

    Returns:
        (is_valid, error_message_or_extracted_json)
    """
    # 1. Попытка вырезать JSON блок с помощью регулярного выражения
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return False, "Критическая ошибка: В твоем ответе полностью отсутствует JSON-объект, ограниченный фигурной скобкой '{' и '}'."

    json_str = match.group(0)

    # 2. Проверка валидности синтаксиса JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"Ошибка синтаксиса JSON (JSONDecodeError): {str(exc)}. Пожалуйста, исправь структуру, проверь запятые и закрывающие кавычки/скобки."

    # 3. Проверка соответствия схеме (наличие ключа items)
    if not isinstance(data, dict) or "items" not in data:
        return False, "Ошибка схемы данных: Корневой объект должен содержать ключ 'items', внутри которого находится массив позиций ТЗ."

    if not isinstance(data["items"], list):
        return False, "Ошибка схемы данных: Ключ 'items' должен строго являться массивом объектов (списком позиций)."

    return True, raw_content


def analyze_tz_with_ai(tz_text: str, system_prompt: str, model_name: str, max_retries: int = 3) -> str:
    """
    Отправляет текст ТЗ выбранной модели с ограничением итераций.
    В случае генерации некорректного JSON или нарушения схемы, агент автоматически
    исправляет свой ответ в рамках контекста (Self-Correction).
    Защищает лимиты токенов, очищая историю от громоздких ошибочных ответов.

    Args:
        tz_text: извлечённый текст технического задания.
        system_prompt: системный промпт (может быть кастомизирован пользователем).
        model_name: id модели в API GPTunneL, например "gpt-4o".
        max_retries: максимальное число попыток самоисправления агента (дефолт: 2).

    Returns:
        Сырой текст ответа модели, гарантированно содержащий валидный JSON.

    Raises:
        AIRequestError: при сетевых ошибках или если агент исчерпал попытки исправления.
    """
    client = _get_client()

    # Первичный контекст: система и само ТЗ
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": tz_text},
    ]

    # Контейнер для хранения сообщений текущей итерации исправления
    correction_messages = []

    for attempt in range(1, max_retries + 1):
        # Собираем пул сообщений для отправки: база + только актуальная жалоба на ошибку
        messages_to_send = base_messages + correction_messages

        try:
            with st.spinner(f"Анализ ТЗ моделью {model_name} (Попытка {attempt}/{max_retries})..."):
                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.1,  # Минимальная температура для максимальной строгости формата
                    messages=messages_to_send,
                )
        except APITimeoutError as exc:
            raise AIRequestError("Превышено время ожидания ответа от модели.") from exc
        except APIError as exc:
            raise AIRequestError(f"Ошибка API GPTunneL: {exc}") from exc
        except Exception as exc:
            raise AIRequestError(f"Неизвестная ошибка при обращении к ИИ: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            if attempt == max_retries:
                raise AIRequestError("Модель вернула пустой ответ после всех попыток.")
            correction_messages = [{"role": "user", "content": "Ты вернул пустой ответ. Выведи строго валидный JSON."}]
            continue

        # Внутренняя агентская валидация полученного ответа
        is_valid, validation_result = _validate_json_structure(content)

        if is_valid:
            if attempt > 1:
                st.success(f"Агент успешно исправил формат данных на шаге {attempt}!")
            return content

        # Если JSON невалиден и попытки еще не исчерпаны
        if attempt < max_retries:
            st.warning(
                f"Попытка {attempt}: Обнаружена ошибка формата. Агент оптимизирует контекст и отправляет лог на исправление...")

            # Формируем жесткую обратную связь без накопления мусора в истории сообщений.
            # Обрезаем контент старого ответа ИИ, чтобы модель понимала, где ошиблась, но мы не переплачивали за токены.
            feedback_message = (
                f"Твой предыдущий ответ содержал ошибку:\n{validation_result}\n\n"
                "Исправь её. Запрещено писать вводные слова, рассуждения или markdown-теги. "
                "Выведи ТОЛЬКО чистый, исправленный JSON-объект, строго соответствующий схеме."
            )
            correction_messages = [
                {"role": "assistant", "content": content[:1000] + "... [обрезано для экономии токенов]"},
                {"role": "user", "content": feedback_message}
            ]
        else:
            # Все лимиты попыток исчерпаны, падаем с ошибкой для интерфейса Streamlit
            raise AIRequestError(
                f"ИИ-агент исчерпал лимит попыток ({max_retries}) и не смог выдать верный формат данных. "
                f"Последняя ошибка валидации: {validation_result}"
            )

    return ""
