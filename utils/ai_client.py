# utils/ai_client.py
"""
Обёртка над API GPTunneL (эндпоинт совместим с OpenAI Chat Completions).
Универсальный двухпроходный ИИ-агент:
1. Локализует страницы спецификации (отсекая адреса, графики и ГОСТы).
2. Проход 1 (Skeleton): строит эталонный реестр всех позиций закупки (номер, название, страницы).
3. Проход 2 (Batch Extraction): детально извлекает характеристики, габариты и итоговое количество
   группами по 4 конкретные позиции из эталонного списка, исключая разрывы на стыках страниц.
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
        "(таблица, где перечислены наименования закупаемых товаров, мебели или оборудования, их технические характеристики и количество).\n\n"
        "ВНИМАНИЕ: Строго исключи страницы, которые содержат ТОЛЬКО посторонние таблицы, такие как:\n"
        "- Отдельные адресные перечни / Адреса доставки (Приложение 2 и т.п.)\n"
        "- Графики и сроки поставки\n"
        "- Перечни нормативных актов (ГОСТы, СНиПы)\n\n"
        "Выведи ответ строго в формате JSON (массив номеров страниц, которые нужно обработать):\n"
        "{\n  \"target_pages\": [8, 9, 10]\n}\n"
        "Не пиши никаких объяснений до или после JSON."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=500,
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
        preview = pages[idx][:300].replace("\n", " ").strip()
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


def _extract_specification_skeleton(all_pages: list[str], target_indices: list[int], model_name: str) -> list[dict]:
    """
    Проход 1 (Skeleton): просматривает всю спецификацию целиком и строит точный эталонный
    реестр всех позиций закупки (номер, наименование, стартовая и конечная страницы).
    """
    client = _get_client()

    tagged_pages = []
    for idx in target_indices:
        tagged_pages.append(f"=== СТРАНИЦА {idx + 1} ===\n{all_pages[idx]}")
    full_spec_text = "\n\n".join(tagged_pages)

    skeleton_prompt = (
        "Ты — эксперт по анализу тендерных спецификаций. Перед тобой полный текст спецификации с маркерами страниц.\n"
        "Твоя задача — составить ТОЧНЫЙ эталонный реестр всех самостоятельных позиций закупки в порядке их следования в документе.\n\n"
        "Правила:\n"
        "1. Новая позиция начинается там, где есть её порядковый номер и наименование товара (в первом столбце таблицы или в объединяющей строке-заголовке).\n"
        "2. Одна позиция часто занимает несколько страниц (заголовок на одной странице, характеристики и количество — на следующих). НЕ разбивай одну позицию на несколько!\n"
        "3. Не включай в список итоговые строки ('Итого', 'Всего') и разделы с условиями поставки/гарантии.\n"
        "4. Для каждой позиции укажи:\n"
        "   - 'source_number': оригинальный порядковый номер позиции в ТЗ (целое число: 1, 2, 3...)\n"
        "   - 'name': точное наименование объекта закупки из ТЗ\n"
        "   - 'start_page': номер страницы (из маркера === СТРАНИЦА N ===), где начинается эта позиция\n"
        "   - 'end_page': номер страницы, где заканчиваются её характеристики и указано количество\n\n"
        "Верни строго JSON без пояснений:\n"
        "{\n  \"skeleton\": [\n    {\"source_number\": 1, \"name\": \"Наименование\", \"start_page\": 8, \"end_page\": 9}\n  ]\n}"
    )

    try:
        with st.spinner("Шаг 1/2: Агент строит эталонный реестр всех позиций ТЗ..."):
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=2048,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": skeleton_prompt},
                    {"role": "user", "content": full_spec_text},
                ],
            )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            skeleton = data.get("skeleton", [])
            if isinstance(skeleton, list) and skeleton:
                return skeleton
    except Exception:
        pass
    return []


def _slice_pages_for_batch(
    all_pages: list[str],
    target_indices: list[int],
    batch_items: list[dict],
) -> str:
    """
    Выбирает диапазон страниц для конкретного батча позиций с запасом ±1 страница от спецификации,
    чтобы гарантированно захватить начало первой позиции батча и количество последней позиции батча.
    """
    min_page = min(int(it.get("start_page", 1)) for it in batch_items)
    max_page = max(int(it.get("end_page", len(all_pages))) for it in batch_items)

    # Переводим в 0-based индексы
    start_idx = max(0, min_page - 1)
    end_idx = min(len(all_pages) - 1, max_page - 1)

    # Расширяем на 1 страницу спецификации в обе стороны для страховки на стыках
    pos_in_target_start = 0
    pos_in_target_end = len(target_indices) - 1
    for pos, t_idx in enumerate(target_indices):
        if t_idx <= start_idx:
            pos_in_target_start = pos
        if t_idx >= end_idx:
            pos_in_target_end = pos
            break

    safe_start_pos = max(0, pos_in_target_start - 1)
    safe_end_pos = min(len(target_indices) - 1, pos_in_target_end + 1)

    selected_indices = target_indices[safe_start_pos : safe_end_pos + 1]
    return "\n\n".join([f"=== СТРАНИЦА {idx + 1} ===\n{all_pages[idx]}" for idx in selected_indices])


def analyze_tz_with_ai(tz_data: str | dict, system_prompt: str, model_name: str, max_retries: int = 2) -> str:
    """
    Универсальный двухпроходный ИИ-агент:
    1. Находит страницы спецификации.
    2. Строит эталонный реестр всех позиций (Skeleton).
    3. Извлекает полные данные группами по 4 позиции из эталонного списка.
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

    client = _get_client()

    # Шаг 2: Построение эталонного реестра позиций (Проход 1)
    skeleton = _extract_specification_skeleton(all_pages, target_indices, model_name)

    if skeleton:
        st.info(f"📋 В спецификации обнаружено позиций: {len(skeleton)}. Начинаю детальный сбор...")
        batch_size = 4
        batches = [skeleton[i : i + batch_size] for i in range(0, len(skeleton), batch_size)]
    else:
        # Резервный режим, если скелет не построился: делим по 5 страниц
        step = 4
        page_groups = [target_indices[i : i + 5] for i in range(0, len(target_indices), step)]
        batches = [[{"page_indices": grp}] for grp in page_groups]

    collected_items: list[dict] = []

    # Шаг 3: Детальное извлечение батчами (Проход 2)
    for b_idx, batch in enumerate(batches):
        if "page_indices" in batch[0]:
            # Резервный режим без скелета
            grp = batch[0]["page_indices"]
            context_text = "\n\n".join([f"=== СТРАНИЦА {idx + 1} ===\n{all_pages[idx]}" for idx in grp])
            target_instruction = (
                "Извлеки все полные позиции закупки из представленного фрагмента. "
                "Обязательно склеивай позиции, разорванные между соседними страницами."
            )
        else:
            # Основной режим по эталонному списку из 4 позиций
            context_text = _slice_pages_for_batch(all_pages, target_indices, batch)
            targets_list_str = "\n".join(
                [f"- Позиция №{it.get('source_number')}: «{it.get('name')}» (ориентировочно стр. {it.get('start_page')}–{it.get('end_page')})" for it in batch]
            )
            target_instruction = (
                f"Твоя задача — извлечь полные данные СТРОГО для следующих {len(batch)} позиций из спецификации:\n"
                f"{targets_list_str}\n\n"
                "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ДЛЯ ЭТОГО БЛОКА:\n"
                "1. Верни в массиве 'items' РОВНО эти позиции (в том же количестве и порядке). Другие позиции из текста игнорируй.\n"
                "2. Одна позиция часто растянута на 2–3 страницы: заголовок и начало характеристик на одной странице, "
                "продолжение характеристик на второй, а итоговое количество (или разбивка количества по адресам/строкам) — на следующей странице. "
                "Обязательно проследи каждую позицию до самого конца её блока в таблице, чтобы собрать ВСЕ характеристики и ТОЧНОЕ количество!\n"
                "3. Если количество для позиции указано несколькими цифрами (например, по адресам или партиям внутри этой позиции) — обязательно сложи их в одно итоговое число 'quantity'.\n"
                "4. Для КАЖДОЙ позиции извлекай абсолютно ВСЕ технические характеристики без сокращений (включая 'Вид товара' и др.), даже если они повторяются от позиции к позиции.\n"
                "5. НЕ включай в характеристики общие условия доставки, разгрузки, гарантии и нормативные акты."
            )

        batch_system_prompt = (
            f"{system_prompt}\n\n"
            f"Сейчас ты обрабатываешь группу {b_idx + 1} из {len(batches)}.\n"
            f"{target_instruction}\n"
            "Верни строго чистый JSON с ключом 'items'."
        )

        correction_messages = []
        batch_success = False

        for attempt in range(1, max_retries + 1):
            messages_to_send = [
                {"role": "system", "content": batch_system_prompt},
                {"role": "user", "content": context_text},
            ] + correction_messages

            try:
                with st.spinner(f"Извлечение данных (Группа {b_idx + 1}/{len(batches)}, Попытка {attempt})..."):
                    response = client.chat.completions.create(
                        model=model_name,
                        max_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.0,
                        messages=messages_to_send,
                    )
            except (APITimeoutError, APIError, Exception) as exc:
                if attempt == max_retries:
                    raise AIRequestError(f"Сбой сети на группе {b_idx + 1}: {exc}") from exc
                continue

            content = response.choices[0].message.content or ""
            is_valid, validation_result = _validate_json_structure(content)

            if is_valid:
                batch_data = json.loads(validation_result)
                items_out = batch_data.get("items", [])
                collected_items.extend(items_out)
                batch_success = True
                break

            if attempt < max_retries:
                feedback_message = (
                    f"Ошибка формата JSON на текущей группе:\n{validation_result}\n\n"
                    "Исправь ошибку структуры и выведи строго чистый JSON-объект."
                )
                correction_messages = [
                    {"role": "assistant", "content": content[:500] + "..."},
                    {"role": "user", "content": feedback_message},
                ]

        if not batch_success:
            raise AIRequestError(f"Агент не смог разобрать группу {b_idx + 1} из-за ошибок формата.")

    # Сквозная перенумерация итогового списка
    for idx, item in enumerate(collected_items, start=1):
        item.pop("source_number", None)
        item["item_number"] = idx

    return json.dumps({"items": collected_items}, ensure_ascii=False)
