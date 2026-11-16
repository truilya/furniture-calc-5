# utils/parsers.py
"""
Модуль отвечает за извлечение текста и таблиц из загруженных файлов (.pdf, .docx).
Преобразует таблицы в компактный выровненный табличный текст с сохранением пустых ячеек
(чтобы столбцы не сдвигались) и без дублирования текста.
"""
from __future__ import annotations
import io

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


def _rows_to_compact_table(rows: list[list[str | None]]) -> str:
    """
    Преобразует матрицу строк таблицы из pdfplumber в компактный Markdown-подобный вид.
    Пустые ячейки заменяются на '-', чтобы сохранить строгую привязку колонок (включая 'Количество').
    """
    if not rows:
        return ""
    lines = []
    for r_idx, row in enumerate(rows):
        cells = []
        has_content = False
        for cell in row:
            cleaned = " ".join((cell or "").split()).strip()
            if cleaned:
                has_content = True
                cells.append(cleaned)
            else:
                cells.append("-")
        if has_content:
            lines.append("| " + " | ".join(cells) + " |")
            if r_idx == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


def _read_pdf_pages(file_bytes: bytes) -> list[str]:
    """
    Извлекает текст и таблицы из PDF постранично.
    При наличии pdfplumber извлекает таблицы с сохранением сетки ячеек,
    а текст вне таблиц добавляет отдельно без дублирования содержимого ячеек.
    """
    if HAS_PDFPLUMBER:
        pages_out = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                found_tables = page.find_tables() or []
                if found_tables:
                    table_blocks = []
                    for tbl in found_tables:
                        extracted = tbl.extract()
                        if extracted:
                            md_tbl = _rows_to_compact_table(extracted)
                            if md_tbl:
                                table_blocks.append(md_tbl)

                    # Фильтруем текст вне рамок таблиц, чтобы не дублировать цифры количества
                    bboxes = [tbl.bbox for tbl in found_tables]

                    def _not_inside_tables(obj):
                        x0 = obj.get("x0", 0)
                        top = obj.get("top", 0)
                        x1 = obj.get("x1", 0)
                        bottom = obj.get("bottom", 0)
                        cx = (x0 + x1) / 2
                        cy = (top + bottom) / 2
                        for bx0, btop, bx1, bbottom in bboxes:
                            if bx0 <= cx <= bx1 and btop <= cy <= bbottom:
                                return False
                        return True

                    outside_page = page.filter(_not_inside_tables)
                    outside_text = (outside_page.extract_text() or "").strip()

                    parts = []
                    if outside_text:
                        parts.append(outside_text)
                    if table_blocks:
                        parts.append("\n\n".join(table_blocks))
                    pages_out.append("\n\n".join(parts).strip())
                else:
                    raw_text = (page.extract_text(layout=True) or "").strip()
                    pages_out.append(raw_text)
        return pages_out

    reader = PdfReader(io.BytesIO(file_bytes))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _flatten_docx_table_rows(table_tag) -> list[str]:
    """
    Преобразует HTML-таблицу из mammoth в список строк с разделителем '|',
    разворачивая вложенные таблицы (где часто хранится количество по адресам)
    и сохраняя пустые ячейки как '-' для защиты от сдвига столбцов.
    """
    out_lines = []
    for r_idx, tr in enumerate(table_tag.find_all("tr", recursive=False)):
        cells = tr.find_all(["td", "th"], recursive=False)
        row_cells = []
        has_content = False

        for cell in cells:
            nested_tables = cell.find_all("table")
            if nested_tables:
                nested_lines = []
                for nt in nested_tables:
                    for ntr in nt.find_all("tr"):
                        ncells = [" ".join(c.get_text(" ", strip=True).split()) for c in ntr.find_all(["td", "th"])]
                        ncells = [c for c in ncells if c]
                        if ncells:
                            nested_lines.append(" — ".join(ncells))
                val = "; ".join(nested_lines).strip()
            else:
                val = " ".join(cell.get_text(" ", strip=True).split()).strip()

            if val:
                has_content = True
                row_cells.append(val)
            else:
                row_cells.append("-")

        if has_content:
            out_lines.append("| " + " | ".join(row_cells) + " |")
            if r_idx == 0:
                out_lines.append("| " + " | ".join(["---"] * len(row_cells)) + " |")

    return out_lines


def _parse_docx_to_chunks(html_content: str, rows_per_chunk: int = 5) -> list[str]:
    """
    Проходит по ВСЕМ элементам DOCX (заголовки, параграфы и ВСЕ таблицы).
    Разбивает таблицы на логические страницы с дублированием шапки таблицы сверху.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.body if soup.body else soup

    chunks = []
    text_buffer = []

    for elem in body.find_all(["p", "h1", "h2", "h3", "h4", "table"], recursive=False):
        if elem.name == "table":
            table_lines = _flatten_docx_table_rows(elem)
            if not table_lines:
                continue

            prefix_text = "\n".join(text_buffer).strip()
            text_buffer = []

            if len(table_lines) <= 2:
                block = f"{prefix_text}\n" + "\n".join(table_lines) if prefix_text else "\n".join(table_lines)
                chunks.append(block.strip())
                continue

            header_block = table_lines[:2]
            body_lines = table_lines[2:]

            for i in range(0, len(body_lines), rows_per_chunk):
                slice_lines = body_lines[i : i + rows_per_chunk]
                chunk_str = "\n".join(header_block + slice_lines)
                if i == 0 and prefix_text:
                    chunks.append(f"{prefix_text}\n{chunk_str}")
                else:
                    chunks.append(chunk_str)
        else:
            txt = " ".join(elem.get_text(" ", strip=True).split()).strip()
            if txt:
                text_buffer.append(txt)
                if len("\n".join(text_buffer)) > 2500:
                    chunks.append("\n".join(text_buffer).strip())
                    text_buffer = []

    if text_buffer:
        chunks.append("\n".join(text_buffer).strip())

    return chunks


def _read_docx_pages(file_bytes: bytes) -> list[str]:
    """Конвертирует DOCX в компактный табличный текст и режет на логические страницы."""
    try:
        file_obj = io.BytesIO(file_bytes)
        result = mammoth.convert_to_html(file_obj)
        raw_html = result.value.strip()
        return _parse_docx_to_chunks(raw_html, rows_per_chunk=5)
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
