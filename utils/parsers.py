# utils/parsers.py
"""
Модуль отвечает ТОЛЬКО за извлечение текста из загруженных файлов.
Никакой бизнес-логики (обращения к ИИ, генерации Excel) — единственная ответственность.
"""
from __future__ import annotations
import io

import streamlit as st
#import mammoth
import docx
import pandas as pd
from pypdf import PdfReader


class ParsingError(Exception):
    """Ошибка извлечения текста из файла."""


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages_text).strip()


'''def _read_docx(file_bytes: bytes) -> str:
    """Извлекает текст и таблицы из DOCX, сохраняя порядок и структуру Markdown."""
    try:
        file_obj = io.BytesIO(file_bytes)
        # Конвертируем DOCX в Markdown формат
        result = mammoth.convert_to_markdown(file_obj)
        return result.value.strip()
    except Exception as exc:
        raise ParsingError(f"Ошибка парсинга DOCX через Mammoth: {exc}") from exc'''


def _read_docx(file_bytes: bytes) -> str:
    """
    Парсит DOCX напрямую через структуру документа.
    Сохраняет точный порядок абзацев и таблиц. Таблицы конвертируются в Markdown.
    """
    try:
        # Загружаем файл из байтов в объект документа python-docx
        doc = docx.Document(io.BytesIO(file_bytes))
        full_document_elements = []

        # Перебираем все элементы в теле документа строго по порядку их появления
        for element in doc.element.body:

            # 1. Если элемент — обычный абзац текста (Paragraph)
            if element.tag.endswith('p'):
                p = docx.text.paragraph.Paragraph(element, doc)
                text = p.text.strip()
                if text:
                    full_document_elements.append(text)

            # 2. Если элемент — таблица (Table)
            elif element.tag.endswith('tbl'):
                t = docx.table.Table(element, doc)
                table_data = []

                # Извлекаем текст из каждой ячейки построчно
                for row in t.rows:
                    # Очищаем текст ячеек от лишних переносов строк внутри самой ячейки,
                    # чтобы не ломать структуру Markdown-таблицы
                    row_text = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                    table_data.append(row_text)

                if table_data:
                    # Превращаем сырую матрицу строк в DataFrame
                    df_table = pd.DataFrame(table_data)

                    # Делаем первую строку заголовком таблицы (если в ней есть данные)
                    if len(df_table) > 1:
                        df_table.columns = df_table.iloc[0]
                        df_table = df_table[1:]

                    # Конвертируем DataFrame в идеальную Markdown-таблицу (формат GitHub Pipe)
                    # index=False убирает лишнюю техническую колонку с индексами строк
                    markdown_table = df_table.to_markdown(index=False, tablefmt="pipe")

                    # Добавляем пустые строки вокруг таблицы для лучшего восприятия моделью
                    full_document_elements.append(f"\n{markdown_table}\n")

        # Собираем весь документ воедино с разделением через два переноса строки
        return "\n\n".join(full_document_elements).strip()

    except Exception as exc:
        raise ParsingError(f"Ошибка парсинга DOCX через python-docx: {exc}") from exc

@st.cache_data(show_spinner=False)
def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """
    Кэшируется по хешу байтов файла: при rerun (например, смена модели
    в селекторе) повторный парсинг не выполняется.
    """
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    try:
        if suffix == "pdf":
            text = _read_pdf(file_bytes)
        elif suffix == "docx":
            text = _read_docx(file_bytes)
        else:
            raise ParsingError(f"Неподдерживаемый формат: .{suffix}")
    except ParsingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParsingError(f"Не удалось прочитать '{file_name}': {exc}") from exc

    if not text:
        raise ParsingError(
            f"Из файла '{file_name}' не удалось извлечь текст "
            "(возможно, это скан без текстового слоя)."
        )
    return text