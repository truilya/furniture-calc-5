# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Автономный ИИ-агент с двухэтапной фильтрацией (Python-радар + ИИ-верификатор заголовков),
защищающий систему от парсинга посторонних таблиц (адресов, графиков и т.д.).
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
    api_key = st.secrets["GPTUNNEL_API_KEY"]
    base_url = st.secrets.get("GPTUNNEL_BASE_URL", "https://gptunnel.ru/v1")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)


def _validate_json_structure(raw_content: str) -> tuple[bool, str]:
    """Внутренний валидатор структуры ответа модели."""
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return False, "Критическая ошибка: В ответе отсутствует JSON-объект."

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"Ошибка синтаксиса JSON: {str(exc)}."

    if not isinstance(data, dict) or "items" not in data:
        return False, "Ошибка схемы: Отсутствует корневой ключ 'items'."

    if not isinstance(data["items"], list):
        return False, "Ошибка схемы: Ключ 'items' должен быть массивом."

    return True, json_str


def _verify_pages_with_ai(candidate_pages_meta: str, model_name: str) -> list[int]:
    """
    Шаг 1.2: ИИ-верификатор. Анализирует заголовки и превью страниц-кандидатов
    и отсекает лишние таблицы (адреса, графики поставки, ГОСТы).
    """
    client = _get_client()
    verification_prompt = (
        "Ты — аналитик тендерной документации. Перед тобой список страниц документа, на которых обнаружены таблицы.\n"
        "Твоя задача — определить, на каких именно страницах находится 'Перечень объектов закупки' / 'Спецификация товаров' "
        "(таблица, где перечислены наименования закупаемых товаров, мебели или оборудования, их технические характеристики и количество).\n\n"
        "ВНИМАНИЕ: Строго исключи страницы, которые содержат посторонние таблицы, такие как:\n"
        "- Адресные перечни / Адреса доставки / Распределение по объектам\n"
        "- Графики и сроки поставки\n"
        "- Перечни нормативных актов (ГОСТы, СНиПы)\n\n"
        "Выведи ответ строго в формате JSON (массив номеров страниц, которые нужно обработать):\n"
        "{\n  \"target_pages\": [8, 9, 10, 11]\n}\n"
        "Не пиши никаких объяснений до или после JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=400,
            temperature=0.0,
            messages=[
                {"role": "system", "content": verification_prompt},
                {"role": "user", "content": candidate_pages_meta},
            ],
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return [int(p) for p in data.get("target_pages", [])]
    except Exception:
        pass
    return []


def _detect_specification_pages_pipeline(pages: list[str], model_name: str) -> list[int]:
    """
    Двухэтапный конвейер локализации:
    1. Быстрый Python-фильтр по маркерам табличной структуры.
    2. Точечный ИИ-анализ заголовков для отсечения адресных и юридических таблиц.
    """
    # Этап 1: Python находит все страницы с табличной структурой
    candidate_indices = []
    for idx, page_text in enumerate(pages):
        # Если на странице есть разделители строк таблицы из parsers.py
        if page_text.count("|") > 2 or "характеристики" in page_text.lower():
            candidate_indices.append(idx)

    if not candidate_indices:
        return []

    # Этап 2: Строим легкое текстовое превью (заголовки) найденных страниц для ИИ
    meta_lines = []
    for idx in candidate_indices:
        # Берем первые 200 символов страницы, чтобы ИИ увидел контекст и заголовок таблицы
        preview = pages[idx][:200].replace('\n', ' ').strip()
        meta_lines.append(f"Страница {idx + 1}: {preview}...")

    candidate_pages_meta = "\n".join(meta_lines)

    # Отправляем превью на верификацию модели
    with st.spinner("Агент отсекает посторонние таблицы (адреса, графики)..."):
        target_pages = _verify_pages_with_ai(candidate_pages_meta, model_name)

    # Переводим человеческие номера страниц обратно в 0-based индексы Python
    valid_indices = []
    for p in target_pages:
        idx = p - 1
        if 0 <= idx < len(pages) and idx in candidate_indices:
            valid_indices.append(idx)

    return sorted(list(set(valid_indices)))


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Полностью автономный многошаговый ИИ-агент.
    Умный поиск спецификаций с защитой от ложных таблиц + постраничный сбор порциями.
    """
    if isinstance(tz_data, str):
        tz_data = {
            "total_pages": 1,
            "pages": [tz_data],
            "summary": tz_data[:4000]
        }

    all_pages = tz_data.get("pages", [])
    if not all_pages:
        raise AIRequestError("Ошибка: Входящий документ не содержит текста для анализа.")

    # Шаг 1: Двухэтапная локализация страниц спецификации
    target_indices = _detect_specification_pages_pipeline(all_pages, model_name)

    # Подстраховка
    if not target_indices:
        st.warning("Маркеры спецификаций не распознаны. Обрабатываем документ целиком.")
        target_indices = list(range(len(all_pages)))
    else:
        human_pages = [idx + 1 for idx in target_indices]
        st.info(f"🎯 Агент отфильтровал документ и сфокусировался на спецификациях (страницы: {human_pages})")

    # Шаг 2: Постраничный сбор позиций чанками (по 3 страницы за раз)
    chunk_size = 3
    final_items = []
    page_chunks = [target_indices[i:i + chunk_size] for i in range(0, len(target_indices), chunk_size)]

    client = _get_client()

    for chunk_idx, chunk in enumerate(page_chunks):
        chunk_text = "\n\n=== Страница спецификации ===\n\n".join([all_pages[idx] for idx in chunk])

        chunk_system_prompt = (
            f"{system_prompt}\n\n"
            f"ТРЕБОВАНИЕ ПОЛНОТЫ: Ты обрабатываешь порцию таблиц {chunk_idx + 1} из {len(page_chunks)}. "
            "Извлеки строго ВСЕ позиции закупки без исключения из этого куска текста. "
            "Не пропускай строки, даже если наименования кажутся одинаковыми — у них могут отличаться цвета или объемы! "
            "Верни строго JSON с ключом 'items'."
        )

        correction_messages = []
        chunk_success = False

        for attempt in range(1, max_retries + 1):
            messages_to_send = [
                                   {"role": "system", "content": chunk_system_prompt},
                                   {"role": "user", "content": chunk_text}
                               ] + correction_messages

            try:
                with st.spinner(f"Извлечение данных (Порция {chunk_idx + 1}/{len(page_chunks)}, Попытка {attempt}):"):
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.1,
                        messages=messages_to_send,
                    )
            except (APITimeoutError, APIError, Exception) as exc:
                if attempt == max_retries:
                    raise AIRequestError(f"Сбой сети на порции {chunk_idx + 1}: {exc}")
                continue

            content = response.choices[0].message.content or ""
            is_valid, validation_result = _validate_json_structure(content)

            if is_valid:
                chunk_data = json.loads(validation_result)
                final_items.extend(chunk_data.get("items", []))
                chunk_success = True
                break

            if attempt < max_retries:
                feedback_message = (
                    f"Ошибка формата JSON на текущей порции:\n{validation_result}\n\n"
                    "Исправь ошибку структуры и выведи строго чистый JSON-объект."
                )
                correction_messages = [
                    {"role": "assistant", "content": content[:500] + "..."},
                    {"role": "user", "content": feedback_message}
                ]

        if not chunk_success:
            raise AIRequestError(f"Агент не смог разобрать порцию текста {chunk_idx + 1} из-за ошибок формата.")

    # Шаг 3: Финальная склейка и сквозная перенумерация
    for idx, item in enumerate(final_items, start=1):
        item["item_number"] = idx

    return json.dumps({"items": final_items}, ensure_ascii=False)

