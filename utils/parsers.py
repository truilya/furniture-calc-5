# utils/parsers.py
"""
Модуль отвечает ТОЛЬКО за извлечение текста из загруженных файлов.
Никакой бизнес-логики (обращения к ИИ, генерации Excel) — единственная ответственность.
"""
from __future__ import annotations
import io

import streamlit as st
import mammoth
from pypdf import PdfReader
from bs4 import BeautifulSoup  # Используем для очистки вложенных таблиц


class ParsingError(Exception):
    """Ошибка извлечения текста из файла."""


def _flatten_html_to_markdown(html_content: str) -> str:
    """
    Принимает тяжелый HTML от mammoth со вложенными таблицами
    и преобразует его в плоский, читаемый для ИИ Markdown-текст.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    markdown_lines = []

    # Находим главную (внешнюю) таблицу объекта закупки
    main_table = soup.find('table')
    if not main_table:
        # Если таблиц нет вообще, возвращаем просто очищенный от тегов текст
        return soup.get_text(separator="\n").strip()

    rows = main_table.find_all('tr', recursive=False)

    for row in rows:
        cells = row.find_all(['td', 'th'], recursive=False)
        row_data = []

        for cell in cells:
            # Проверяем, есть ли ВНУТРИ текущей ячейки вложенная таблица с характеристиками
            nested_table = cell.find('table')

            if nested_table:
                # Если нашли вложенную таблицу, парсим её строки отдельно
                nested_lines = []
                nested_rows = nested_table.find_all('tr')
                for n_row in nested_rows:
                    n_cells = [c.get_text(strip=True) for c in n_row.find_all(['td', 'th'])]
                    # Фильтруем пустые ячейки
                    n_cells = [c for c in n_cells if c.strip()]
                    if n_cells:
                        # Объединяем наименование характеристики, значение и инструкцию через тире
                        nested_lines.append(" — ".join(n_cells))

                # Соединяем все характеристики позиции через перенос строки
                row_data.append("\n" + "\n".join(nested_lines) + "\n")
            else:
                # Если вложенной таблицы нет, просто берем очищенный текст ячейки
                text = cell.get_text(strip=True)
                if text:
                    row_data.append(text)

        if row_data:
            # Склеиваем колонки основной строки через стандартный пайп (|)
            markdown_lines.append(" | ".join(row_data))
            markdown_lines.append("-" * 20)  # Визуальный разделитель строк для ИИ

    return "\n".join(markdown_lines).strip()


def _read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages_text).strip()


def _read_docx(file_bytes: bytes) -> str:
    """Извлекает текст и таблицы из DOCX, сохраняя порядок и структуру Markdown."""
    try:
        file_obj = io.BytesIO(file_bytes)
        # Конвертируем DOCX в Markdown формат
        #result = mammoth.convert_to_markdown(file_obj)
        result = mammoth.convert_to_html(file_obj)
        #return result.value.strip()

        raw_html = result.value.strip()

        # 2. Выпрямляем вложенные таблицы HTML в плоский Markdown-текст
        flat_markdown = _flatten_html_to_markdown(raw_html)
        return flat_markdown
    except Exception as exc:
        raise ParsingError(f"Ошибка парсинга DOCX через Mammoth: {exc}") from exc



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