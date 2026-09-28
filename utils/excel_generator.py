# utils/excel_generator.py
"""
Парсинг структурированного (JSON) ответа ИИ в pandas.DataFrame
и генерация xlsx-файла. DataFrame — единый источник правды:
он же используется для st.data_editor в будущем.
"""
from __future__ import annotations

import io
import json
import re

import pandas as pd


class ResponseParsingError(Exception):
    """Ошибка разбора ответа ИИ в табличный вид."""


# Поля соответствуют новой схеме ответа модели (закупочные позиции):
# item_number, name, characteristics, max_width_mm, max_depth_mm, max_height_mm, quantity
EXPECTED_COLUMNS = [
    "item_number",
    "name",
    "characteristics",
    "max_width_mm",
    "max_depth_mm",
    "max_height_mm",
    "quantity",
]

# Колонки, значения которых относятся к позиции целиком (не к конкретной
# характеристике) — именно их ячейки объединяются в Excel, если у позиции
# несколько характеристик и, соответственно, несколько строк.
_ITEM_LEVEL_COLUMNS = [col for col in EXPECTED_COLUMNS if col != "characteristics"]


def _extract_json_block(raw_text: str) -> str:
    """
    Модели иногда оборачивают JSON в ```json ... ``` несмотря на промпт.
    Эта функция отделяет чистый JSON от возможного markdown-обрамления.
    """
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ResponseParsingError("В ответе модели не найден JSON-объект.")
    return match.group(0)


def _normalize_characteristics(raw_characteristics) -> list[str]:
    """
    Приводит поле 'characteristics' к списку строк.
    Модель обязана возвращать список, но на случай отклонения от промпта
    (например, строка вместо списка) — подстраховываемся, а не падаем.
    """
    if raw_characteristics is None:
        return []
    if isinstance(raw_characteristics, list):
        return [str(c) for c in raw_characteristics if c is not None and str(c).strip() != ""]
    text = str(raw_characteristics).strip()
    return [text] if text else []


def parse_ai_response_to_df(raw_response: str) -> pd.DataFrame:
    """
    Преобразует сырой ответ ИИ в pandas.DataFrame с фиксированным набором колонок.

    Ожидаемые форматы ответа:
      - {"items": [...]}          — успешный разбор
      - {"error": "текст ошибки"} — модель не смогла прочитать файл

    Каждая характеристика позиции выносится в отдельную строку таблицы.
    Если у позиции несколько характеристик, поля уровня позиции
    (item_number, name, габариты, quantity) физически дублируются в
    DataFrame только в первой строке, а для последующих строк того же
    объекта — оставляются пустыми (None), т.к. при рендере в Excel такие
    ячейки объединяются в одну (см. dataframe_to_excel_bytes).

    Raises:
        ResponseParsingError: при невалидном JSON, отсутствии 'items'
            или явной ошибке модели ("error").
    """
    json_block = _extract_json_block(raw_response)

    try:
        data = json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(f"Ответ модели — невалидный JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ResponseParsingError("Ответ модели должен быть JSON-объектом.")

    # Модель явно сообщила, что не смогла разобрать файл ТЗ.
    if "error" in data:
        error_text = data.get("error") or "Модель сообщила об ошибке анализа файла."
        raise ResponseParsingError(str(error_text))

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ResponseParsingError("В JSON отсутствует непустой список 'items'.")

    rows: list[dict] = []
    merge_ranges: list[tuple[int, int]] = []  # (первая, последняя) строка одной позиции, 0-based

    for item in items:
        if not isinstance(item, dict):
            continue

        base_values = {col: item.get(col) for col in _ITEM_LEVEL_COLUMNS}
        characteristics = _normalize_characteristics(item.get("characteristics"))

        if not characteristics:
            # Пустой список характеристик -> null (пустая ячейка) в одной строке.
            rows.append({**base_values, "characteristics": None})
            continue

        start_row = len(rows)
        for idx, characteristic in enumerate(characteristics):
            if idx == 0:
                rows.append({**base_values, "characteristics": characteristic})
            else:
                # Соседние строки той же позиции: поля уровня позиции не дублируем
                # текстом, т.к. ячейки будут объединены в Excel.
                blank_values = {col: None for col in _ITEM_LEVEL_COLUMNS}
                rows.append({**blank_values, "characteristics": characteristic})
        end_row = len(rows) - 1
        if end_row > start_row:
            merge_ranges.append((start_row, end_row))

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    # Диапазоны строк для объединения ячеек сохраняем в атрибутах DataFrame,
    # чтобы dataframe_to_excel_bytes мог их использовать без пересчёта.
    df.attrs["merge_ranges"] = merge_ranges

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Позиции закупки") -> bytes:
    """
    Сериализует DataFrame в xlsx и возвращает байты — удобно передавать
    напрямую в st.download_button без временных файлов на диске.

    Если у позиции несколько характеристик (несколько строк в df),
    ячейки колонок уровня позиции (item_number, name, габариты, quantity)
    объединяются по вертикали в одну — на основе df.attrs['merge_ranges'].
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        worksheet = writer.sheets[sheet_name]

        # Автоширина колонок для читаемости.
        # Не используем df[col].astype(str).map(len) напрямую: при наличии
        # None/NaN в колонке map может получить float('nan'), у которого
        # нет len(), и упасть с TypeError. Считаем длину вручную,
        # безопасно приводя каждое значение к строке.
        def _safe_len(value) -> int:
            if value is None:
                return 0
            try:
                if pd.isna(value):
                    return 0
            except (TypeError, ValueError):
                pass
            return len(str(value))

        for idx, col in enumerate(df.columns, start=1):
            column_values = df[col].tolist()
            max_content_len = max((_safe_len(v) for v in column_values), default=0)
            max_len = max(max_content_len, len(str(col))) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(max_len, 60)

        # Объединение ячеек для полей уровня позиции, когда у неё несколько
        # строк-характеристик. +2, т.к. строка 1 — заголовок, а df использует 0-based индекс.
        merge_ranges = df.attrs.get("merge_ranges", [])
        item_level_col_indices = [
            idx for idx, col in enumerate(df.columns, start=1) if col in _ITEM_LEVEL_COLUMNS
        ]
        for start_row, end_row in merge_ranges:
            excel_start = start_row + 2
            excel_end = end_row + 2
            for col_idx in item_level_col_indices:
                worksheet.merge_cells(
                    start_row=excel_start, end_row=excel_end,
                    start_column=col_idx, end_column=col_idx,
                )

    buffer.seek(0)
    return buffer.getvalue()
