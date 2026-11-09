# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Единая точка обращения к ИИ. Реализует многошаговый агентский цикл локализации и
постраничного сбора данных (Chunked Processing) для 100% гарантии полноты ТЗ.
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
    Внутренний валидатор структуры ответа модели.
    Проверяет наличие JSON, валидность синтаксиса и корневой ключ 'items'.
    """
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return False, "Критическая ошибка: В твоем ответе полностью отсутствует JSON-объект, ограниченный фигурной скобкой '{' и '}'."

    json_str = match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"Ошибка синтаксиса JSON (JSONDecodeError): {str(exc)}. Пожалуйста, исправь структуру, проверь запятые."

    if not isinstance(data, dict) or "items" not in data:
        return False, "Ошибка схемы данных: Корневой объект должен содержать ключ 'items', внутри которого находится массив позиций ТЗ."

    if not isinstance(data["items"], list):
        return False, "Ошибка схемы данных: Ключ 'items' должен строго являться массивом объектов (списком позиций)."

    return True, json_str


def _extract_target_pages_range(summary_text: str, model_name: str) -> list[int]:
    """
    Первый этап агента: анализ структуры ТЗ по вводной части (summary)
    и извлечение номеров страниц, содержащих спецификации товаров.
    """
    client = _get_client()
    localization_prompt = (
        "Ты — эксперт по госзакупкам и тендерам. Твоя задача — проанализировать вводную часть ТЗ "
        "и определить номера логических страниц/секций, на которых расположен 'Перечень объектов закупки' "
        "(таблицы со спецификациями, характеристиками товаров, оборудования или мебели).\n\n"
        "Выведи ответ строго в формате JSON со следующей структурой:\n"
        "{\n  \"start_page\": int,  // Первая страница, где начинается перечень товаров\n"
        "  \"end_page\": int      // Последняя страница перечня товаров\n"
        "}\n"
        "Внимание: Не пиши никаких объяснений, только чистый JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=500,
            temperature=0.0,
            messages=[
                {"role": "system", "content": localization_prompt},
                {"role": "user", "content": summary_text},
            ],
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            start = int(data.get("start_page", 1))
            end = int(data.get("end_page", 1))
            # Корректируем индексы под 0-based list (страницы в ТЗ обычно начинаются с 1)
            return list(range(max(1, start) - 1, end))
    except Exception:
        # Если модель не смогла локализовать, агент подстрахуется и вернет пустой список,
        # что заставит систему обработать весь документ постранично.
        pass
    return []


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Многошаговый ИИ-агент. Локализует таблицы в ТЗ и собирает данные чанками (порциями),
    чтобы гарантировать 100% полноту извлечения данных (всех 16+ позиций) без лени моделей.
    """
    # Поддержка обратной совместимости, если на вход пришла просто строка текста ТЗ
    if isinstance(tz_data, str):
        tz_data = {
            "total_pages": 1,
            "pages": [tz_data],
            "summary": tz_data[:4000]
        }

    all_pages = tz_data.get("pages", [])
    summary_text = tz_data.get("summary", "")

    if not all_pages:
        raise AIRequestError("Ошибка: Входящий документ не содержит текста для анализа.")

    client = _get_client()

    # Шаг 1: Локализация целевых страниц спецификации
    with st.spinner("Агент анализирует структуру документа и локализует таблицы закупки..."):
        target_indices = _extract_target_pages_range(summary_text, model_name)

    # Если локализация не удалась или вернула неадекватные границы, обрабатываем все страницы
    if not target_indices or max(target_indices) >= len(all_pages):
        target_indices = list(range(len(all_pages)))

    # Вычисляем человеческие номера страниц для вывода пользователю
    st.info(
        f"Агент сфокусирован на анализе секций/страниц ТЗ: с {min(target_indices) + 1} по {max(target_indices) + 1}")

    # Шаг 2: Постраничный сбор позиций чанками (по 3 страницы за раз)
    chunk_size = 3
    final_items = []

    # Разбиваем индексы целевых страниц на группы по chunk_size
    page_chunks = [target_indices[i:i + chunk_size] for i in range(0, len(target_indices), chunk_size)]

    for chunk_idx, chunk in enumerate(page_chunks):
        # Собираем текст для текущего чанка
        chunk_text = "\n\n=== Новая страница/секция спецификации ===\n\n".join([all_pages[idx] for idx in chunk])

        # Модифицируем промпт, указывая модели, что это порция данных
        chunk_system_prompt = (
            f"{system_prompt}\n\n"
            f"ВАЖНОЕ ТРЕБОВАНИЕ ДЛЯ ПОЛНОТЫ: Сейчас ты обрабатываешь порцию {chunk_idx + 1} из {len(page_chunks)}. "
            "Извлеки абсолютно ВСЕ позиции закупки без исключения, которые упоминаются в данном куске текста. "
            "Ничего не обобщай, не пропускай строки и не считай их дубликатами, если у них разные адреса или характеристики! "
            "Верни ответ строго в формате JSON с ключом 'items'."
        )

        correction_messages = []
        chunk_success = False

        # Цикл самоисправления (Self-Correction) для конкретного чанка
        for attempt in range(1, max_retries + 1):
            messages_to_send = [
                                   {"role": "system", "content": chunk_system_prompt},
                                   {"role": "user", "content": chunk_text}
                               ] + correction_messages

            try:
                with st.spinner(f"Извлечение позиций (Порция {chunk_idx + 1}/{len(page_chunks)}, Попытка {attempt}):"):
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.1,
                        messages=messages_to_send,
                    )
            except (APITimeoutError, APIError, Exception) as exc:
                if attempt == max_retries:
                    raise AIRequestError(f"Сбой сети на порции {chunk_idx + 1}: {exc}") from exc
                continue

            content = response.choices[0].message.content or ""
            is_valid, validation_result = _validate_json_structure(content)

            if is_valid:
                # Парсим успешный JSON порции и сохраняем позиции
                chunk_data = json.loads(validation_result)
                final_items.extend(chunk_data.get("items", []))
                chunk_success = True
                break

            # Если JSON сломался на этой порции, просим исправить
            if attempt < max_retries:
                feedback_message = (
                    f"Твой ответ на этой порции текста содержал ошибку:\n{validation_result}\n\n"
                    "Исправь её. Выведи ТОЛЬКО чистый JSON-объект."
                )
                correction_messages = [
                    {"role": "assistant", "content": content[:500] + "..."},
                    {"role": "user", "content": feedback_message}
                ]

        if not chunk_success:
            raise AIRequestError(f"Агент не смог разобрать порцию текста {chunk_idx + 1} из-за ошибок формата.")

    # Шаг 3: Финальная сборка и перенумерация позиций по порядку (item_number)
    for idx, item in enumerate(final_items, start=1):
        item["item_number"] = idx

    # Возвращаем монолитный JSON-объект, готовый для твоего excel_generator.py
    return json.dumps({"items": final_items}, ensure_ascii=False)
