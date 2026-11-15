# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Автономный ИИ-агент с двухэтапной фильтрацией страниц, склейкой разорванных позиций
окном с перекрытием и умным слиянием по оригинальному номеру позиции из ТЗ.
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


def _build_overlapping_chunks(indices: list[int], chunk_size: int = 5, overlap: int = 2) -> list[list[int]]:
    """
    Разбивает список индексов страниц на компактные блоки по 5 страниц с перекрытием в 2 страницы,
    чтобы модель не упиралась в лимит выходных токенов и не пропускала характеристики.
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


def _item_completeness_score(item: dict) -> tuple[int, int, int]:
    """
    Оценивает полноту извлечённой позиции при слиянии дублей с границ перекрытия:
    приоритет у версии, где заполнено количество, больше характеристик и заполнены габариты.
    """
    has_qty = 1 if item.get("quantity") is not None else 0
    chars_count = len(item.get("characteristics") or [])
    dims_count = sum(
        1 for k in ("max_width_mm", "max_depth_mm", "max_height_mm") if item.get(k) is not None
    )
    return (has_qty, chars_count, dims_count)


def _merge_and_deduplicate_items(items: list[dict]) -> list[dict]:
    """
    Объединяет позиции из перекрывающихся блоков по оригинальному номеру в ТЗ (source_number).
    Если позиция попала на границу чанков дважды (в одном обрезана, в другом полная),
    выбирает самую полную версию.
    """
    by_source_num: dict[int, dict] = {}
    fallback_items: list[dict] = []
    seen_fallback = set()

    for item in items:
        raw_src_num = item.get("source_number")
        src_num = None
        if raw_src_num is not None:
            try:
                src_num = int(re.sub(r"\D", "", str(raw_src_num)))
            except ValueError:
                src_num = None

        if src_num is not None and src_num > 0:
            if src_num not in by_source_num:
                by_source_num[src_num] = item
            else:
                existing = by_source_num[src_num]
                if _item_completeness_score(item) > _item_completeness_score(existing):
                    by_source_num[src_num] = item
        else:
            chars_key = tuple(sorted(str(c).strip().lower() for c in item.get("characteristics", [])))
            sig = (
                str(item.get("name", "")).strip().lower(),
                item.get("max_width_mm"),
                item.get("max_depth_mm"),
                item.get("max_height_mm"),
                item.get("quantity"),
                chars_key,
            )
            if sig not in seen_fallback:
                seen_fallback.add(sig)
                fallback_items.append(item)

    sorted_numbered = [by_source_num[k] for k in sorted(by_source_num.keys())]
    merged = sorted_numbered + fallback_items

    for idx, item in enumerate(merged, start=1):
        item.pop("source_number", None)
        item["item_number"] = idx

    return merged


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Полностью автономный многошаговый ИИ-агент.
    Находит спецификации, склеивает разорванные между страницами позиции
    компактными окнами с перекрытием и объединяет результаты по номеру позиции в ТЗ.
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

    # Шаг 2: Компактные блоки по 5 страниц с перекрытием в 2 страницы
    page_chunks = _build_overlapping_chunks(target_indices, chunk_size=5, overlap=2)
    raw_collected_items = []

    client = _get_client()

    for chunk_idx, chunk in enumerate(page_chunks):
        chunk_text = "\n".join([all_pages[idx] for idx in chunk])

        is_first = (chunk_idx == 0)
        is_last = (chunk_idx == len(page_chunks) - 1)

        boundary_rules = (
            "ПРАВИЛО СКЛЕЙКИ РАЗОРВАННЫХ ПОЗИЦИЙ:\n"
            "1. Новая позиция начинается ТОЛЬКО там, где есть её порядковый номер и наименование товара (в ячейке столбца или в объединяющей строке-заголовке). "
            "Если на новой странице сразу идут характеристики без нового номера и названия позиции — объедини их с предыдущей позицией.\n"
        )
        if not is_first:
            boundary_rules += (
                "2. Если в самом начале фрагмента идет обрезанный «хвост» характеристик без номера и наименования позиции — "
                "ИГНОРИРУЙ этот хвост, он уже учтён в предыдущем блоке.\n"
            )
        if not is_last:
            boundary_rules += (
                "3. Если последняя позиция в конце фрагмента оборвана (не дошла до конца характеристик или количества) — "
                "можешь либо пропустить её, либо извлечь то, что есть, указав её точный source_number из ТЗ (в следующем блоке она будет собрана целиком).\n"
            )

        chunk_system_prompt = (
            f"{system_prompt}\n\n"
            f"Сейчас ты обрабатываешь блок {chunk_idx + 1} из {len(page_chunks)}.\n"
            f"{boundary_rules}\n"
            "ВАЖНО:\n"
            "- Обязательно заполняй поле 'source_number' точным номером позиции из ТЗ (1, 2, 3...).\n"
            "- Для КАЖДОЙ позиции извлекай ВСЕ технические характеристики без сокращений (включая 'Вид товара' и др.), "
            "но НЕ включай в характеристики общие условия доставки, разгрузки, гарантии и нормативные акты из конца документа.\n"
            "- Верни строго JSON с ключом 'items'."
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
                        temperature=0.0,
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

    # Шаг 3: Умное слияние позиций по оригинальному номеру в ТЗ (source_number) и сквозная перенумерация
    final_items = _merge_and_deduplicate_items(raw_collected_items)

    return json.dumps({"items": final_items}, ensure_ascii=False)
