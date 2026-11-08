# utils/parsers.py
"""
Модуль отвечает за извлечение текста из загруженных файлов.
Поддерживает постраничное чтение для агентской архитектуры, чтобы не перегружать контекст ИИ.
"""
from __future__ import annotations
import io
import re

import streamlit as st
import mammoth
from pypdf import PdfReader
from bs4 import BeautifulSoup


class ParsingError(Exception):
    """Ошибка извлечения текста из файла."""


def _flatten_html_to_markdown_chunks(html_content: str) -> list[str]:
    """
    Принимает тяжелый HTML от mammoth со вложенными таблицами
    и преобразует его в список текстовых блоков (строк таблицы), имитируя страницы.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []

    main_table = soup.find('table')
    if not main_table:
        # Если таблиц нет, делим просто текст по крупным абзацам
        paragraphs = soup.get_text(separator="\n").strip().split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    rows = main_table.find_all('tr', recursive=False)

    current_chunk = []
    # Группируем строки (например, по 3 строки основной таблицы в один блок), чтобы не плодить слишком мелкие куски
    for idx, row in enumerate(rows):
        cells = row.find_all(['td', 'th'], recursive=False)
        row_data = []

        for cell in cells:
            nested_table = cell.find('table')
            if nested_table:
                nested_lines = []
                nested_rows = nested_table.find_all('tr')
                for n_row in nested_rows:
                    n_cells = [c.get_text(strip=True) for c in n_row.find_all(['td', 'th'])]
                    n_cells = [c for c in n_cells if c.strip()]
                    if n_cells:
                        nested_lines.append(" — ".join(n_cells))
                row_data.append("\n" + "\n".join(nested_lines) + "\n")
            else:
                text = cell.get_text(strip=True)
                if text:
                    row_data.append(text)

        if row_data:
            current_chunk.append(" | ".join(row_data))
            current_chunk.append("-" * 20)

        # Каждые 3 строки основной спецификации формируют "страницу" для ИИ
        if (idx + 1) % 3 == 0 and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks


def _read_pdf_pages(file_bytes: bytes) -> list[str]:
    """Извлекает текст из PDF поэлементно (список страниц)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _read_docx_pages(file_bytes: bytes) -> list[str]:
    """Конвертирует DOCX в HTML и режет на логические чанки-страницы."""
    try:
        file_obj = io.BytesIO(file_bytes)
        result = mammoth.convert_to_html(file_obj)
        raw_html = result.value.strip()
        return _flatten_html_to_markdown_chunks(raw_html)
    except Exception as exc:
        raise ParsingError(f"Ошибка парсинга DOCX через Mammoth: {exc}") from exc


@st.cache_data(show_spinner=False)
def get_file_metadata_and_pages(file_bytes: bytes, file_name: str) -> dict:
    """
    Главная агентская функция. Вместо одной огромной строки возвращает словарь с метаданными
    и списком страниц/чанков. Кэшируется Streamlit.

    Returns:
        dict: {
            "total_pages": int,
            "pages": list[str],
            "summary": str  # Первые 2 страницы для ознакомления ИИ
        }
    """
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    try:
        if suffix == "pdf":
            pages = _read_pdf_pages(file_bytes)
        elif suffix == "docx":
            pages = _read_docx_pages(file_bytes)
        else:
            raise ParsingError(f"Неподдерживаемый формат: .{suffix}")
    except ParsingError:
        raise
    except Exception as exc:
        raise ParsingError(f"Не удалось прочитать '{file_name}': {exc}") from exc

    # Фильтруем абсолютно пустые страницы
    pages = [p for p in pages if p.strip()]

    if not pages:
        raise ParsingError(
            f"Из файла '{file_name}' не удалось извлечь текст "
            "(возможно, это скан без текстового слоя)."
        )

    # Формируем краткую выжимку (первые 2 страницы), чтобы агент понял контекст документа
    summary_limit = min(2, len(pages))
    summary_text = f"Файл: {file_name}. Всего логических страниц/секций: {len(pages)}.\n\n"
    summary_text += "--- НАЧАЛО ДОКУМЕНТА (ПЕРВЫЕ СТРАНИЦЫ) ---\n"
    summary_text += "\n\n=== Новая секция ===\n\n".join(pages[:summary_limit])
    summary_text += "\n\n--- КОНЕЦ ВВОДНОЙ ЧАСТИ ---"

    return {
        "total_pages": len(pages),
        "pages": pages,
        "summary": summary_text
    }


def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """
    Сохраняем обратную совместимость со старым кодом.
    Если где-то в streamlit_app.py всё еще запрашивается сплошной текст.
    """
    data = get_file_metadata_and_pages(file_bytes, file_name)
    return "\n\n=== Новая секция/Страница ===\n\n".join(data["pages"])
