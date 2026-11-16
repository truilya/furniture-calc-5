# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Двухпроходный автономный ИИ-агент:
1. Локализует страницы со спецификациями (отсекая адреса, графики и ГОСТы).
2. Проход 1 (Skeleton): строит эталонный реестр всех позиций спецификации с привязкой к страницам.
3. Проход 2 (Batch Extraction): извлекает характеристики и массив слагаемых количества
   малыми группами по 2 позиции (чтобы избежать расфокусировки на однотипных позициях),
   а итоговое количество вычисляет программно на стороне Python.
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


def _validate_json_structure(raw_content: str, required_key: str = "items") -> tuple[bool, str]:
    """Внутренний валидатор структуры ответа модели."""
    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if not match:
        return False, "Критическая ошибка: В ответе отсутствует JSON-объект."

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return False, f"Ошибка синтаксиса JSON: {str(exc)}."

    if not isinstance(data, dict) or required_key not in data:
        return False, f"Ошибка схемы: Отсутствует корневой ключ '{required_key}'."

    if not isinstance(data[required_key], list):
        return False, f"Ошибка схемы: Ключ '{required_key}' должен быть массивом."

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
        "(таблица, где перечислены наименования закупаемых товаров, мебели или оборудования, их технические характеристики и количество).\n\n"
        "ВНИМАНИЕ: Строго исключи страницы, которые содержат посторонние таблицы, такие как:\n"
        "- Отдельные приложения с адресными перечнями / графиками поставки (Приложение 2 и т.п.)\n"
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
        if page_text.count("|") > 2 or "<table" in page_text.lower() or "характеристики" in page_text.lower():
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


def _extract_specification_skeleton(
    pages: list[str],
    target_indices: list[int],
    model_name: str,
    max_retries: int = 2,
) -> list[dict]:
    """
    Проход 1 (Skeleton): Строит эталонный реестр всех позиций спецификации
    без извлечения характеристик, чтобы исключить дубли, пропуски и лишние позиции на стыках.
    """
    client = _get_client()

    marked_pages = []
    for idx in target_indices:
        marked_pages.append(f"=== СТРАНИЦА {idx + 1} ===\n{pages[idx]}")
    full_spec_text = "\n\n".join(marked_pages)

    skeleton_prompt = (
        "Ты — аналитик тендерной документации. Перед тобой полный текст спецификации с маркерами страниц (=== СТРАНИЦА N ===).\n"
        "Твоя задача — составить ТОЛЬКО точный реестр всех самостоятельных позиций закупки (без характеристик и без количества).\n\n"
        "Правила:\n"
        "1. Новая позиция начинается ТОЛЬКО там, где есть её явный порядковый номер и наименование товара (в отдельном столбце или в объединяющей строке-заголовке).\n"
        "2. НЕ объединяй позиции с одинаковым названием, если они идут под разными порядковыми номерами (например, позиции 1, 2, 3, 4...).\n"
        "3. НЕ включай итоговые строки («Итого», «Всего»), примечания, условия доставки или гарантии.\n"
        "4. Для каждой позиции укажи start_page (номер страницы, где она начинается) и end_page (номер страницы, где она заканчивается, включая её количество).\n\n"
        "Верни строго JSON:\n"
        "{\n"
        "  \"skeleton\": [\n"
        "    {\"source_number\": 1, \"name\": \"Наименование позиции\", \"start_page\": 8, \"end_page\": 9}\n"
        "  ]\n"
        "}"
    )

    correction_messages = []
    for attempt in range(1, max_retries + 1):
        messages = [
            {"role": "system", "content": skeleton_prompt},
            {"role": "user", "content": full_spec_text},
        ] + correction_messages

        try:
            with st.spinner(f"Шаг 1/2: Построение реестра позиций спецификации (попытка {attempt})..."):
                response = client.chat.completions.create(
                    model=model_name,
                    max_tokens=3000,
                    temperature=0.0,
                    messages=messages,
                )
        except (APITimeoutError, APIError, Exception) as exc:
            if attempt == max_retries:
                raise AIRequestError(f"Ошибка при построении реестра позиций: {exc}") from exc
            continue

        content = response.choices[0].message.content or ""
        is_valid, validation_result = _validate_json_structure(content, required_key="skeleton")
        if is_valid:
            data = json.loads(validation_result)
            skeleton = data.get("skeleton", [])
            if skeleton:
                return skeleton

        if attempt < max_retries:
            correction_messages = [
                {"role": "assistant", "content": content[:500] + "..."},
                {"role": "user", "content": f"Ошибка формата JSON: {validation_result}. Верни строго JSON с ключом 'skeleton'."},
            ]

    return []


def _compute_exact_quantity(item: dict) -> int | float | None:
    """
    Программно вычисляет итоговое количество позиции из массива слагаемых `quantities`,
    полностью исключая арифметические ошибки LLM при сложении в уме.
    """
    raw_quantities = item.get("quantities")
    if isinstance(raw_quantities, list) and raw_quantities:
        numeric_parts = []
        for val in raw_quantities:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_parts.append(val)
            elif isinstance(val, str):
                cleaned = val.strip().replace(" ", "").replace(",", ".")
                match = re.search(r"\d+(?:\.\d+)?", cleaned)
                if match:
                    num = float(match.group(0))
                    numeric_parts.append(int(num) if num.is_integer() else num)

        if numeric_parts:
            total = sum(numeric_parts)
            return int(total) if isinstance(total, float) and total.is_integer() else total

    fallback_qty = item.get("quantity")
    if isinstance(fallback_qty, (int, float)) and not isinstance(fallback_qty, bool):
        return int(fallback_qty) if isinstance(fallback_qty, float) and fallback_qty.is_integer() else fallback_qty

    return None


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Полностью автономный многошаговый ИИ-агент:
    1. Локализует страницы спецификации.
    2. Строит эталонный реестр всех позиций (Skeleton).
    3. Собирает характеристики и слагаемые количества малыми батчами по 2 позиции,
       после чего суммирует количество на стороне Python.
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

    # Шаг 1: Локализация страниц спецификации
    target_indices = _detect_specification_pages_pipeline(all_pages, model_name)
    if not target_indices:
        st.warning("Маркеры спецификаций не распознаны. Обрабатываем документ целиком.")
        target_indices = list(range(len(all_pages)))
    else:
        human_pages = [idx + 1 for idx in target_indices]
        st.info(f"🎯 Агент сфокусировался на спецификациях (страницы: {human_pages})")

    # Шаг 2: Построение эталонного реестра позиций (Skeleton)
    skeleton = _extract_specification_skeleton(all_pages, target_indices, model_name, max_retries=max_retries)
    if not skeleton:
        raise AIRequestError("Не удалось определить перечень позиций в спецификации документа.")

    st.info(f"📋 В спецификации обнаружено позиций: {len(skeleton)}. Запускаем детальный сбор...")

    client = _get_client()
    collected_items: list[dict] = []

    # Шаг 3: Батчевый сбор по 2 позиции за раз (чтобы избежать расфокусировки на однотипных позициях)
    batch_size = 2
    total_batches = (len(skeleton) + batch_size - 1) // batch_size

    min_spec_idx = min(target_indices)
    max_spec_idx = max(target_indices)

    for b_idx in range(0, len(skeleton), batch_size):
        batch = skeleton[b_idx:b_idx + batch_size]
        current_batch_num = (b_idx // batch_size) + 1

        # Определяем диапазон страниц для текущего батча с запасом +-1 страница
        start_pages = [int(x.get("start_page", min_spec_idx + 1)) - 1 for x in batch if x.get("start_page")]
        end_pages = [int(x.get("end_page", max_spec_idx + 1)) - 1 for x in batch if x.get("end_page")]

        p_start = max(min_spec_idx, (min(start_pages) if start_pages else min_spec_idx) - 1)
        p_end = min(max_spec_idx, (max(end_pages) if end_pages else max_spec_idx) + 1)

        slice_indices = [idx for idx in target_indices if p_start <= idx <= p_end]
        if not slice_indices:
            slice_indices = target_indices

        batch_text = "\n\n".join([f"=== СТРАНИЦА {idx + 1} ===\n{all_pages[idx]}" for idx in slice_indices])

        target_list_str = "\n".join(
            [f"- Позиция №{item['source_number']}: «{item['name']}» (стр. {item.get('start_page', '?')}–{item.get('end_page', '?')})" for item in batch]
        )

        batch_system_prompt = (
            f"{system_prompt}\n\n"
            f"ЗАДАЧА ТЕКУЩЕГО ШАГА (Батч {current_batch_num} из {total_batches}):\n"
            f"Извлеки полные данные СТРОГО для следующих {len(batch)} позиций (игнорируй все остальные позиции в тексте):\n"
            f"{target_list_str}\n\n"
            "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ДЛЯ ЭТОГО БАТЧА:\n"
            "1. Верни ровно столько объектов в массиве 'items', сколько указано в списке выше, в том же порядке и с теми же 'source_number'.\n"
            "2. Внимательно следи за границами каждой позиции! Позиции могут иметь одинаковое название и похожие характеристики — НЕ бери количество от соседней позиции.\n"
            "3. Таблица позиции может продолжаться на следующей странице (=== СТРАНИЦА N+1 ===) — обязательно дойди до самого конца этой позиции, где указано её количество (или разбивка количества по адресам/строкам).\n"
            "4. НЕ складывай числа в уме! В поле 'qty_note' кратко опиши, из каких цифр складывается количество именно для ЭТОЙ позиции, "
            "а в массив 'quantities' запиши все слагаемые количества числами (например: [8, 9, 8] или [25]). Если в позиции указаны и слагаемые по адресам, и итоговая строка по этой позиции — запиши ТОЛЬКО слагаемые ИЛИ ТОЛЬКО итог, чтобы не удвоить сумму.\n"
            "5. Для КАЖДОЙ позиции извлекай абсолютно ВСЕ технические характеристики без сокращений (включая 'Вид товара' и др.), исключая общие условия доставки/гарантии в конце документа."
        )

        correction_messages = []
        batch_success = False

        for attempt in range(1, max_retries + 1):
            messages_to_send = [
                {"role": "system", "content": batch_system_prompt},
                {"role": "user", "content": batch_text},
            ] + correction_messages

            try:
                with st.spinner(f"Шаг 2/2: Сбор позиций (группа {current_batch_num}/{total_batches}, попытка {attempt})..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.0,
                        messages=messages_to_send,
                    )
            except (APITimeoutError, APIError, Exception) as exc:
                if attempt == max_retries:
                    raise AIRequestError(f"Сбой сети на группе {current_batch_num}: {exc}") from exc
                continue

            content = response.choices[0].message.content or ""
            is_valid, validation_result = _validate_json_structure(content, required_key="items")

            if is_valid:
                batch_data = json.loads(validation_result)
                items_in_batch = batch_data.get("items", [])

                for item in items_in_batch:
                    item["quantity"] = _compute_exact_quantity(item)
                    item.pop("quantities", None)
                    item.pop("qty_note", None)
                    item.pop("source_number", None)

                collected_items.extend(items_in_batch)
                batch_success = True
                break

            if attempt < max_retries:
                feedback_message = (
                    f"Ошибка формата JSON на группе {current_batch_num}:\n{validation_result}\n\n"
                    "Исправь ошибку структуры и выведи строго чистый JSON-объект."
                )
                correction_messages = [
                    {"role": "assistant", "content": content[:500] + "..."},
                    {"role": "user", "content": feedback_message},
                ]

        if not batch_success:
            raise AIRequestError(f"Агент не смог разобрать группу позиций {current_batch_num} из-за ошибок формата.")

    # Сквозная финальная перенумерация
    for idx, item in enumerate(collected_items, start=1):
        item["item_number"] = idx

    return json.dumps({"items": collected_items}, ensure_ascii=False)
