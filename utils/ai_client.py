# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Автономный ИИ-агент с двухэтапной фильтрацией страниц, склейкой разорванных позиций
и обработкой окном с перекрытием (Sliding Window with Overlap).
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
    ИИ-верификатор. Анализирует заголовки и превью страниц-кандидатов
    и отсекает лишние таблицы (адреса, графики поставки, ГОСТы).
    """
    client = _get_client()
    verification_prompt = (
        "Ты — аналитик тендерной документации. Перед тобой список страниц документа, на которых обнаружены таблицы.\n"
        "Твоя задача — определить, на каких именно страницах находится 'Перечень объектов закупки' / 'Спецификация товаров' "
        "(таблица, где перечислены наименования закупаемых товаров, мебели или оборудования, их технические характеристики и общее количество).\n\n"
        "ВНИМАНИЕ: Строго исключи страницы, которые содержат посторонние таблицы, такие как:\n"
        "- Адресные перечни / Адреса доставки / Распределение по адресам (Приложение 2 и т.п.)\n"
        "- Графики и сроки поставки\n"
        "- Перечни нормативных актов (ГОСТы, СНиПы)\n\n"
        "Выведи ответ строго в формате JSON (массив номеров страниц, которые нужно обработать):\n"
        "{\n  \"target_pages\": [8, 9, 10]\n}\n"
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
    candidate_indices = []
    for idx, page_text in enumerate(pages):
        if page_text.count("|") > 2 or "характеристики" in page_text.lower():
            candidate_indices.append(idx)

    if not candidate_indices:
        return []

    meta_lines = []
    for idx in candidate_indices:
        preview = pages[idx][:250].replace("\n", " ").strip()
        meta_lines.append(f"Страница {idx + 1}: {preview}...")

    candidate_pages_meta = "\n".join(meta_lines)

    with st.spinner("Агент отсекает посторонние таблицы (адреса, графики)..."):
        target_pages = _verify_pages_with_ai(candidate_pages_meta, model_name)

    valid_indices = []
    for p in target_pages:
        idx = p - 1
        if 0 <= idx < len(pages) and idx in candidate_indices:
            valid_indices.append(idx)

    return sorted(list(set(valid_indices)))


def _build_overlapping_chunks(indices: list[int], chunk_size: int = 8, overlap: int = 1) -> list[list[int]]:
    """
    Разбивает список индексов страниц на крупные блоки с перекрытием (overlap),
    чтобы позиции, разорванные на границе чанков, целиком попадали хотя бы в один блок.
    """
    if len(indices) <= chunk_size:
        return [indices]

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(indices), step):
        chunk = indices[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(indices):
            break
    return chunks


def _deduplicate_items(items: list[dict]) -> list[dict]:
    """
    Удаляет дубликаты позиций, которые могли возникнуть из-за перекрытия (overlap) страниц.
    Сравнивает нормализованное наименование + габариты + характеристики.
    """
    seen = set()
    unique_items = []
    for item in items:
        chars_key = tuple(sorted(str(c).strip().lower() for c in item.get("characteristics", [])))
        signature = (
            str(item.get("name", "")).strip().lower(),
            item.get("max_width_mm"),
            item.get("max_depth_mm"),
            item.get("max_height_mm"),
            item.get("quantity"),
            chars_key,
        )
        if signature not in seen:
            seen.add(signature)
            unique_items.append(item)
    return unique_items


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Полностью автономный многошаговый ИИ-агент.
    Находит спецификации, склеивает разорванные между страницами позиции
    и собирает данные крупными окнами с перекрытием.
    """
    if isinstance(tz_data, str):
        tz_data = {
            "total_pages": 1,
            "pages": [tz_data],
            "summary": tz_data[:4000],
        }

    all_pages = tz_data.get("pages", [])
    if not all_pages:
        raise AIRequestError("Ошибка: Входящий документ не содержит текста для анализа.")

    # Шаг 1: Двухэтапная локализация страниц спецификации
    target_indices = _detect_specification_pages_pipeline(all_pages, model_name)

    if not target_indices:
        st.warning("Маркеры спецификаций не распознаны. Обрабатываем документ целиком.")
        target_indices = list(range(len(all_pages)))
    else:
        human_pages = [idx + 1 for idx in target_indices]
        st.info(f"🎯 Агент сфокусировался на спецификациях (страницы: {human_pages})")

    # Шаг 2: Формируем крупные блоки по 8 страниц с перекрытием в 1 страницу
    page_chunks = _build_overlapping_chunks(target_indices, chunk_size=8, overlap=1)
    raw_collected_items = []

    client = _get_client()

    for chunk_idx, chunk in enumerate(page_chunks):
        chunk_text = "\n".join([all_pages[idx] for idx in chunk])

        is_first = (chunk_idx == 0)
        is_last = (chunk_idx == len(page_chunks) - 1)

        boundary_rules = (
            "ПРАВИЛО СКЛЕЙКИ РАЗОРВАННЫХ ТАБЛИЦ:\n"
            "1. В PDF-документах одна позиция таблицы часто разрывается между соседними страницами "
            "(на одной странице заголовок и начало характеристик, на следующей — конец характеристик и остаток количества). "
            "Обязательно объединяй такие разорванные части в одну целую позицию и суммируй её количество!\n"
        )
        if not is_first:
            boundary_rules += (
                "2. Если в самом начале фрагмента идет обрезанный «хвост» позиции без её номера и названия — "
                "ИГНОРИРУЙ этот хвост, он уже учтен в предыдущем блоке.\n"
            )
        if not is_last:
            boundary_rules += (
                "3. Если последняя позиция в самом конце фрагмента оборвана (не хватает характеристик или количества) — "
                "НЕ включай её в ответ, она целиком войдет в следующий блок.\n"
            )

        chunk_system_prompt = (
            f"{system_prompt}\n\n"
            f"Сейчас ты обрабатываешь блок {chunk_idx + 1} из {len(page_chunks)}.\n"
            f"{boundary_rules}\n"
            "Извлеки ВСЕ полные позиции закупки из этого блока без пропусков и обобщений. "
            "Не считай позиции дубликатами, если у них отличается цвет, тип секции (прямая/угловая), размеры или количество. "
            "Верни строго JSON с ключом 'items'."
        )

        correction_messages = []
        chunk_success = False

        for attempt in range(1, max_retries + 1):
            messages_to_send = [
                {"role": "system", "content": chunk_system_prompt},
                {"role": "user", "content": chunk_text},
            ] + correction_messages

            try:
                with st.spinner(f"Извлечение данных (Блок {chunk_idx + 1}/{len(page_chunks)}, Попытка {attempt})..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.1,
                        messages=messages_to_send,
                    )
            except (APITimeoutError, APIError, Exception) as exc:
                if attempt == max_retries:
                    raise AIRequestError(f"Сбой сети на блоке {chunk_idx + 1}: {exc}") from exc
                continue

            content = response.choices[0].message.content or ""
            is_valid, validation_result = _validate_json_structure(content)

            if is_valid:
                chunk_data = json.loads(validation_result)
                raw_collected_items.extend(chunk_data.get("items", []))
                chunk_success = True
                break

            if attempt < max_retries:
                feedback_message = (
                    f"Ошибка формата JSON на текущем блоке:\n{validation_result}\n\n"
                    "Исправь ошибку структуры и выведи строго чистый JSON-объект."
                )
                correction_messages = [
                    {"role": "assistant", "content": content[:500] + "..."},
                    {"role": "user", "content": feedback_message},
                ]

        if not chunk_success:
            raise AIRequestError(f"Агент не смог разобрать блок текста {chunk_idx + 1} из-за ошибок формата.")

    # Шаг 3: Дедупликация на стыках перекрытия и сквозная перенумерация
    final_items = _deduplicate_items(raw_collected_items)
    for idx, item in enumerate(final_items, start=1):
        item["item_number"] = idx

    return json.dumps({"items": final_items}, ensure_ascii=False)
