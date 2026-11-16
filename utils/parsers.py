# utils/parsers.py
"""
Модуль отвечает за извлечение текста и таблиц из загруженных файлов (.pdf, .docx).
Преобразует таблицы в чистый HTML с сохранением структуры ячеек (colspan/rowspan)
и дублированием шапки таблицы в каждом логическом чанке для точного подсчета количества.
"""
from __future__ import annotations
import io
import html as html_lib

import streamlit as st
import mammoth
from pypdf import PdfReader
from bs4 import BeautifulSoup

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class ParsingError(Exception):
    """Ошибка извлечения текста из файла."""


def _table_rows_to_html(rows: list[list[str | None]]) -> str:
    """Преобразует матрицу строк таблицы из pdfplumber в чистый HTML."""
    if not rows:
        return ""
    html_parts = ["<table>"]
    for r_idx, row in enumerate(rows):
        tag = "th" if r_idx == 0 else "td"
        cells_html = []
        for cell in row:
            cell_text = html_lib.escape((cell or "").strip()).replace("\n", "<br>")
            cells_html.append(f"<{tag}>{cell_text}</{tag}>")
        html_parts.append(f"<tr>{''.join(cells_html)}</tr>")
    html_parts.append("</table>")
    return "\n".join(html_parts)


def _clean_html_table(table_tag) -> tuple[str, list[str]]:
    """
    Очищает HTML-таблицу от лишних стилей и атрибутов, но СОХРАНЯЕТ rowspan, colspan
    и пустые ячейки, чтобы столбцы (особенно 'Количество') не сдвигались.
    Возвращает HTML-шапку (первую строку) и список очищенных HTML-строк <tr>.
    """
    clean_rows = []
    for tr in table_tag.find_all("tr", recursive=False):
        cells_out = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            tag_name = cell.name
            attrs = []
            if cell.has_attr("colspan"):
                attrs.append(f'colspan="{cell["colspan"]}"')
            if cell.has_attr("rowspan"):
                attrs.append(f'rowspan="{cell["rowspan"]}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""

            nested_tables = cell.find_all("table")
            if nested_tables:
                nested_parts = []
                for nt in nested_tables:
                    for ntr in nt.find_all("tr"):
                        ncells = [c.get_text(" ", strip=True) for c in ntr.find_all(["td", "th"])]
                        ncells = [c for c in ncells if c]
                        if ncells:
                            nested_parts.append(" — ".join(ncells))
                cell_content = "<br>".join(html_lib.escape(p) for p in nested_parts)
            else:
                text = cell.get_text("\n", strip=True)
                cell_content = html_lib.escape(text).replace("\n", "<br>")

            cells_out.append(f"<{tag_name}{attr_str}>{cell_content}</{tag_name}>")

        if cells_out:
            clean_rows.append(f"<tr>{''.join(cells_out)}</tr>")

    if not clean_rows:
        return "", []

    header_row = clean_rows[0]
    body_rows = clean_rows[1:] if len(clean_rows) > 1 else []
    return header_row, body_rows


def _parse_docx_html_to_chunks(html_content: str, rows_per_chunk: int = 4) -> list[str]:
    """
    Проходит по ВСЕМ элементам DOCX (заголовки, параграфы и ВСЕ таблицы).
    Каждую таблицу разбивает на логические страницы по rows_per_chunk строк,
    приклеивая сверху исходную шапку таблицы (header_row), чтобы модель всегда видела названия колонок.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body if soup.body else soup

    chunks = []
    text_buffer = []

    for elem in body.find_all(["p", "h1", "h2", "h3", "h4", "table"], recursive=False):
        if elem.name == "table":
            header_row, body_rows = _clean_html_table(elem)
            if not header_row and not body_rows:
                continue

            prefix_text = "\n".join(text_buffer).strip()
            text_buffer = []

            if not body_rows:
                table_html = f"<table>\n{header_row}\n</table>"
                block = f"{prefix_text}\n{table_html}".strip() if prefix_text else table_html
                chunks.append(block)
                continue

            for i in range(0, len(body_rows), rows_per_chunk):
                slice_rows = body_rows[i : i + rows_per_chunk]
                table_html = "<table>\n" + header_row + "\n" + "\n".join(slice_rows) + "\n</table>"
                if i == 0 and prefix_text:
                    chunks.append(f"{prefix_text}\n{table_html}")
                else:
                    chunks.append(table_html)
        else:
            txt = elem.get_text(" ", strip=True)
            if txt:
                text_buffer.append(txt)
                if len("\n".join(text_buffer)) > 2500:
                    chunks.append("\n".join(text_buffer).strip())
                    text_buffer = []

    if text_buffer:
        chunks.append("\n".join(text_buffer).strip())

    return chunks


def _read_pdf_pages(file_bytes: bytes) -> list[str]:
    """
    Извлекает текст и таблицы из PDF построчно.
    При наличии pdfplumber преобразует таблицы страницы в чистый HTML,
    чтобы сохранить точную привязку колонок (включая 'Количество').
    """
    if HAS_PDFPLUMBER:
        pages_out = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                raw_text = (page.extract_text() or "").strip()

                if tables:
                    html_tables = [_table_rows_to_html(t) for t in tables if t]
                    combined = raw_text + "\n\n" + "\n\n".join(html_tables)
                    pages_out.append(combined.strip())
                else:
                    pages_out.append(raw_text)
        return pages_out

    reader = PdfReader(io.BytesIO(file_bytes))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _read_docx_pages(file_bytes: bytes) -> list[str]:
    """Конвертирует DOCX в очищенный структурный HTML и режет на логические страницы."""
    try:
        file_obj = io.BytesIO(file_bytes)
        result = mammoth.convert_to_html(file_obj)
        raw_html = result.value.strip()
        return _parse_docx_html_to_chunks(raw_html, rows_per_chunk=4)
    except Exception as exc:
        raise ParsingError(f"Ошибка парсинга DOCX через Mammoth: {exc}") from exc


@st.cache_data(show_spinner=False)
def get_file_metadata_and_pages(file_bytes: bytes, file_name: str) -> dict:
    """
    Главная агентская функция. Возвращает словарь с метаданными и списком страниц/чанков.
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

    pages = [p for p in pages if p.strip()]

    if not pages:
        raise ParsingError(
            f"Из файла '{file_name}' не удалось извлечь текст "
            "(возможно, это скан без текстового слоя)."
        )

    summary_limit = min(2, len(pages))
    summary_text = f"Файл: {file_name}. Всего логических страниц/секций: {len(pages)}.\n\n"
    summary_text += "--- НАЧАЛО ДОКУМЕНТА (ПЕРВЫЕ СТРАНИЦЫ) ---\n"
    summary_text += "\n\n=== Новая секция ===\n\n".join(pages[:summary_limit])
    summary_text += "\n\n--- КОНЕЦ ВВОДНОЙ ЧАСТИ ---"

    return {
        "total_pages": len(pages),
        "pages": pages,
        "summary": summary_text,
    }


def extract_text_from_file(file_bytes: bytes, file_name: str) -> str:
    """Обратная совместимость со старым кодом."""
    data = get_file_metadata_and_pages(file_bytes, file_name)
    return "\n\n=== Новая секция/Страница ===\n\n".join(data["pages"])
