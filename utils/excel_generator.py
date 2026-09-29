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

from config import EXPECTED_COLUMNS, COLUMN_LABELS, COLUMN_GROUPS


class ResponseParsingError(Exception):
    """Ошибка разбора ответа ИИ в табличный вид."""


# EXPECTED_COLUMNS и COLUMN_LABELS вынесены в config.py — это часть контракта
# данных (соответствует DEFAULT_SYSTEM_PROMPT), а не деталь сериализации в xlsx.

# Колонки, значения которых относятся к позиции целиком (не к конкретной
# характеристике) — именно их ячейки объединяются в Excel, если у позиции
# несколько характеристик и, соответственно, несколько строк.
_ITEM_LEVEL_COLUMNS = [col for col in EXPECTED_COLUMNS if col != "characteristics"]

# Колонка 'sketch' (Эскиз) не приходит от ИИ и не заполняется программно —
# она нужна только как место в таблице для последующей ручной вставки
# изображения/чертежа пользователем прямо в Excel.
# 'volume' тоже не приходит от ИИ — это вычисляемая колонка: в DataFrame
# остаётся пустой (None), а формула Excel проставляется только при
# экспорте в xlsx (см. dataframe_to_excel_bytes).
# 'tz_price' / 'sketch_price' — вводятся пользователем вручную в Excel после
# экспорта; 'tz_sum' / 'sketch_sum' — вычисляются формулой Excel как
# 'volume' * соответствующая цена (см. dataframe_to_excel_bytes).
_UNFILLED_COLUMNS = {"sketch", "volume", "tz_price", "tz_sum", "sketch_price", "sketch_sum"}


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
    item_start_rows: list[int] = []  # 0-based индексы первых строк каждой позиции (для формулы объёма)

    for item in items:
        if not isinstance(item, dict):
            continue

        base_values = {
            col: (None if col in _UNFILLED_COLUMNS else item.get(col))
            for col in _ITEM_LEVEL_COLUMNS
        }
        characteristics = _normalize_characteristics(item.get("characteristics"))

        if not characteristics:
            # Пустой список характеристик -> null (пустая ячейка) в одной строке.
            item_start_rows.append(len(rows))
            rows.append({**base_values, "characteristics": None})
            continue

        start_row = len(rows)
        item_start_rows.append(start_row)
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
    # Строки-начала каждой позиции — именно в них dataframe_to_excel_bytes проставит
    # формулу расчёта объёма (для продолжений той же позиции формула не нужна,
    # т.к. ячейки будут объединены).
    df.attrs["item_start_rows"] = item_start_rows

    return df


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Позиции закупки") -> bytes:
    """
    Сериализует DataFrame в xlsx и возвращает байты — удобно передавать
    напрямую в st.download_button без временных файлов на диске.

    Если у позиции несколько характеристик (несколько строк в df),
    ячейки колонок уровня позиции (item_number, name, габариты, quantity)
    объединяются по вертикали в одну — на основе df.attrs['merge_ranges'].
    """
    # Заголовки переводятся на русский только на выходе, в самом DataFrame
    # (result_df) и остальном коде продолжают использоваться английские
    # имена колонок — это упрощает работу с данными (data_editor, фильтры и т.д.).
    export_df = df.rename(columns=COLUMN_LABELS)
    merge_ranges = df.attrs.get("merge_ranges", [])
    item_start_rows = df.attrs.get("item_start_rows", list(range(len(df))))

    # Обратная карта 'внутреннее имя колонки' -> 'название группы' (для колонок
    # вроде tz_price/tz_sum, объединяемых общим заголовком 'Вариант 1'/'Вариант 2').
    col_to_group: dict[str, str] = {
        col: group_name for group_name, cols in COLUMN_GROUPS.items() for col in cols
    }

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # startrow=1 (0-based) сдвигает заголовки колонок на строку 2 и оставляет
        # строку 1 свободной под общие заголовки групп ('Вариант 1' / 'Вариант 2').
        export_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=1)

        worksheet = writer.sheets[sheet_name]

        # Строка 1: для колонок без группы — объединяем с их заголовком в строке 2
        # по вертикали (чтобы подпись не "провисала" рядом с пустой ячейкой сверху);
        # для колонок в группе — пишем общее название группы и объединяем по горизонтали
        # на ширину всех колонок этой группы, сама группа не трогает строку 2.
        for idx, col in enumerate(df.columns, start=1):
            if col not in col_to_group:
                header_cell = worksheet.cell(row=2, column=idx)
                worksheet.cell(row=1, column=idx).value = header_cell.value
                header_cell.value = None
                worksheet.merge_cells(start_row=1, end_row=2, start_column=idx, end_column=idx)

        for group_name, group_cols in COLUMN_GROUPS.items():
            group_indices = [i for i, col in enumerate(df.columns, start=1) if col in group_cols]
            if not group_indices:
                continue
            start_col, end_col = min(group_indices), max(group_indices)
            worksheet.cell(row=1, column=start_col).value = group_name
            if end_col > start_col:
                worksheet.merge_cells(start_row=1, end_row=1, start_column=start_col, end_column=end_col)

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

        for idx, col in enumerate(export_df.columns, start=1):
            column_values = export_df[col].tolist()
            max_content_len = max((_safe_len(v) for v in column_values), default=0)
            max_len = max(max_content_len, len(str(col))) + 2
            worksheet.column_dimensions[get_column_letter(idx)].width = min(max_len, 60)

        # Колонку 'Эскиз' делаем чуть шире, т.к. в неё пользователь будет вручную
        # вставлять изображение — узкая колонка по ширине заголовка для этого мала.
        sketch_col_idx = list(export_df.columns).index(COLUMN_LABELS["sketch"]) + 1
        worksheet.column_dimensions[get_column_letter(sketch_col_idx)].width = 20

        # Перенос строк и выравнивание по верхнему краю для колонки характеристик,
        # чтобы многострочный текст был читаемым.
        from openpyxl.styles import Alignment

        characteristics_col_idx = list(export_df.columns).index(COLUMN_LABELS["characteristics"]) + 1
        for row_idx in range(3, worksheet.max_row + 1):
            worksheet.cell(row=row_idx, column=characteristics_col_idx).alignment = Alignment(
                wrap_text=True, vertical="top"
            )

        # Формула расчёта объёма (м³) по габаритам и количеству из той же строки.
        # Адреса ячеек берутся по фактическому индексу колонок, а не жёстко — если
        # порядок колонок в config.EXPECTED_COLUMNS изменится, формула перестроится автоматически.
        width_col = list(export_df.columns).index(COLUMN_LABELS["max_width_mm"]) + 1
        depth_col = list(export_df.columns).index(COLUMN_LABELS["max_depth_mm"]) + 1
        height_col = list(export_df.columns).index(COLUMN_LABELS["max_height_mm"]) + 1
        quantity_col = list(export_df.columns).index(COLUMN_LABELS["quantity"]) + 1
        volume_col = list(export_df.columns).index(COLUMN_LABELS["volume"]) + 1
        tz_price_col = list(export_df.columns).index(COLUMN_LABELS["tz_price"]) + 1
        tz_sum_col = list(export_df.columns).index(COLUMN_LABELS["tz_sum"]) + 1
        sketch_price_col = list(export_df.columns).index(COLUMN_LABELS["sketch_price"]) + 1
        sketch_sum_col = list(export_df.columns).index(COLUMN_LABELS["sketch_sum"]) + 1

        width_letter = get_column_letter(width_col)
        depth_letter = get_column_letter(depth_col)
        height_letter = get_column_letter(height_col)
        quantity_letter = get_column_letter(quantity_col)
        volume_letter = get_column_letter(volume_col)
        tz_price_letter = get_column_letter(tz_price_col)
        sketch_price_letter = get_column_letter(sketch_price_col)

        # +3, а не +2: строка 1 теперь занята общими заголовками групп
        # ('Вариант 1' / 'Вариант 2'), строка 2 — заголовками колонок,
        # данные (0-based индекс df) начинаются со строки 3.
        for start_row in item_start_rows:
            excel_row = start_row + 3
            w = f"{width_letter}{excel_row}"
            d = f"{depth_letter}{excel_row}"
            h = f"{height_letter}{excel_row}"
            q = f"{quantity_letter}{excel_row}"
            v = f"{volume_letter}{excel_row}"
            tz_price_ref = f"{tz_price_letter}{excel_row}"
            sketch_price_ref = f"{sketch_price_letter}{excel_row}"
            # Excel всегда хранит в xlsx формулы с английскими именами функций
            # (IF/AND/ISNUMBER), независимо от языка интерфейса: локализованные имена
            # (ЕСЛИ/И/ЕЧИСЛО) openpyxl пишет буквально, Excel их не распознаёт
            # и показывает #ИМЯ?. Excel сам отобразит IF/AND/ISNUMBER как
            # ЕСЛИ/И/ЕЧИСЛО при русской локали интерфейса.
            volume_formula = (
                f'=IF(AND(ISNUMBER({w});ISNUMBER({d});ISNUMBER({h});ISNUMBER({q}));'
                f'({w}/1000)*({d}/1000)*({h}/1000)*{q};"ПУСТО")'
            )
            worksheet.cell(row=excel_row, column=volume_col).value = volume_formula

            # 'Сумма по ТЗ' / 'Сумма по эскизу' = 'Объём' * соответствующая цена.
            # Проверка ISNUMBER здесь не нужна: пустая ячейка цены при умножении
            # даёт 0, что для суммы — ожидаемое поведение (а не ошибка).
            worksheet.cell(row=excel_row, column=tz_sum_col).value = f"={v}*{tz_price_ref}"
            worksheet.cell(row=excel_row, column=sketch_sum_col).value = f"={v}*{sketch_price_ref}"

        # Объединение ячеек для полей уровня позиции, когда у неё несколько
        # строк-характеристик. +3, т.к. строки 1–2 — заголовки групп/колонок, а df использует 0-based индекс.
        item_level_col_indices = [
            idx for idx, col in enumerate(df.columns, start=1) if col in _ITEM_LEVEL_COLUMNS
        ]
        for start_row, end_row in merge_ranges:
            excel_start = start_row + 3
            excel_end = end_row + 3
            for col_idx in item_level_col_indices:
                worksheet.merge_cells(
                    start_row=excel_start, end_row=excel_end,
                    start_column=col_idx, end_column=col_idx,
                )

    buffer.seek(0)
    return buffer.getvalue()
