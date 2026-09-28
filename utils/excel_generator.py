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


EXPECTED_COLUMNS = ["id", "section", "requirement", "description", "priority"]


def _extract_json_block(raw_text: str) -> str:
    """
    Модели иногда оборачивают JSON в ```json ... ``` несмотря на промпт.
    Эта функция отделяет чистый JSON от возможного markdown-обрамления.
    """
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ResponseParsingError("В ответе модели не найден JSON-объект.")
    return match.group(0)


def parse_ai_response_to_df(raw_response: str) -> pd.DataFrame:
    """
    Преобразует сырой ответ ИИ (ожидается JSON вида {"items": [...]})
    в pandas.DataFrame с фиксированным набором колонок.
    """
    json_block = _extract_json_block(raw_response)

    try:
        data = json.loads(json_block)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(f"Ответ модели — невалидный JSON: {exc}") from exc

    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ResponseParsingError("В JSON отсутствует непустой список 'items'.")

    df = pd.DataFrame(items)

    # Гарантируем наличие всех ожидаемых колонок в правильном порядке,
    # даже если модель что-то забыла вернуть.
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[EXPECTED_COLUMNS]

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Требования ТЗ") -> bytes:
    """
    Сериализует DataFrame в xlsx и возвращает байты — удобно передавать
    напрямую в st.download_button без временных файлов на диске.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

        # Автоширина колонок для читаемости
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns, start=1):
            max_len = max(
                df[col].astype(str).map(len).max() if len(df) else 0,
                len(str(col)),
            ) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = min(max_len, 60)

    buffer.seek(0)
    return buffer.getvalue()