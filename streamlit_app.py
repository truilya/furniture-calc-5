# streamlit_app.py
"""
Главный модуль интерфейса. Только UI и координация шагов — никакой
бизнес-логики (она в utils/*). Использует конечный автомат processing_status
в session_state, чтобы интерфейс не сбрасывался между rerun'ами.
"""
from __future__ import annotations

import streamlit as st

from config import AVAILABLE_MODELS, DEFAULT_MODEL_LABEL, DEFAULT_SYSTEM_PROMPT
from utils.parsers import get_file_metadata_and_pages, ParsingError
from utils.ai_client import analyze_tz_with_ai, AIRequestError
from utils.excel_generator import (
    parse_ai_response_to_df,
    dataframe_to_excel_bytes,
    ResponseParsingError,
)

st.set_page_config(page_title="ТЗ → Excel анализатор", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Инициализация session_state — единая точка правды состояния приложения.
# Делается один раз через setdefault, чтобы не затирать значения при rerun.
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "uploaded_file_bytes": None,
        "uploaded_file_name": None,
        "tz_data": None,                       # Хранит структурированный dict вместо плоской строки extracted_text
        "selected_model": AVAILABLE_MODELS[DEFAULT_MODEL_LABEL],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "processing_status": "idle",  # idle | parsing | analyzing | done | error
        "error_message": None,
        "ai_raw_response": None,
        "result_df": None,
        "excel_bytes": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_pipeline_state() -> None:
    """Сброс результатов при загрузке нового файла — старые данные не должны 'протекать'."""
    st.session_state["tz_data"] = None
    st.session_state["ai_raw_response"] = None
    st.session_state["result_df"] = None
    st.session_state["excel_bytes"] = None
    st.session_state["processing_status"] = "idle"
    st.session_state["error_message"] = None


# ---------------------------------------------------------------------------
# Боковая панель: выбор модели + (в будущем) кастомизация промпта
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Настройки анализа")

        model_label = st.selectbox(
            "Модель ИИ",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_LABEL),
            help="Список моделей, доступных через агрегатор GPTunneL",
        )
        st.session_state["selected_model"] = AVAILABLE_MODELS[model_label]

        with st.expander("Системный промпт (продвинутая настройка)"):
            st.session_state["system_prompt"] = st.text_area(
                "Промпт для анализа ТЗ",
                value=st.session_state["system_prompt"],
                height=280,
            )

        st.divider()
        st.caption("В следующих версиях здесь появится: авторизация, история файлов.")


# ---------------------------------------------------------------------------
# Основной пайплайн обработки
# ---------------------------------------------------------------------------
def run_pipeline(uploaded_file) -> None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    # Если загружен новый файл — сбрасываем предыдущие результаты
    if st.session_state["uploaded_file_name"] != file_name:
        reset_pipeline_state()
        st.session_state["uploaded_file_bytes"] = file_bytes
        st.session_state["uploaded_file_name"] = file_name

    # Шаг 1: извлечение текста и разбивка на страницы/секции
    st.session_state["processing_status"] = "parsing"
    try:
        with st.spinner("Извлекаю текст и формирую структуру документа..."):
            # Вызываем новую агентскую функцию парсинга вместо extract_text_from_file
            st.session_state["tz_data"] = get_file_metadata_and_pages(file_bytes, file_name)
    except ParsingError as exc:
        st.session_state["processing_status"] = "error"
        st.session_state["error_message"] = str(exc)
        return

    # Шаг 2: многошаговый анализ ИИ (локализация + сбор порциями)
    st.session_state["processing_status"] = "analyzing"
    try:
        with st.spinner(f"Запуск ИИ-агента с моделью «{st.session_state['selected_model']}»..."):
            # Передаем словарь tz_data вместо плоского текста
            raw_response = analyze_tz_with_ai(
                tz_data=st.session_state["tz_data"],
                system_prompt=st.session_state["system_prompt"],
                model_name=st.session_state["selected_model"],
            )
            st.session_state["ai_raw_response"] = raw_response
    except AIRequestError as exc:
        st.session_state["processing_status"] = "error"
        st.session_state["error_message"] = str(exc)
        return

    # Шаг 3: разбор ответа в таблицу + генерация xlsx
    try:
        df = parse_ai_response_to_df(st.session_state["ai_raw_response"])
        st.session_state["result_df"] = df
        st.session_state["excel_bytes"] = dataframe_to_excel_bytes(df)
        st.session_state["processing_status"] = "done"
    except ResponseParsingError as exc:
        st.session_state["processing_status"] = "error"
        st.session_state["error_message"] = str(exc)


# ---------------------------------------------------------------------------
# Отрисовка результата
# ---------------------------------------------------------------------------
def render_result() -> None:
    status = st.session_state["processing_status"]

    if status == "error":
        st.error(f"Ошибка: {st.session_state['error_message']}")
        return

    if status == "done" and st.session_state["result_df"] is not None:
        st.success("Анализ завершён!")
        st.subheader("Результат анализа ТЗ")

        # В будущем: заменить st.dataframe на st.data_editor для редактирования
        st.dataframe(st.session_state["result_df"], use_container_width=True)

        st.download_button(
            label="📥 Скачать Excel",
            data=st.session_state["excel_bytes"],
            file_name="tz_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Показать извлечённый текст ТЗ"):
            if st.session_state["tz_data"] and "pages" in st.session_state["tz_data"]:
                # Восстанавливаем сплошной текст из списка страниц для UI
                full_text = "\n\n=== Новая секция/Страница ===\n\n".join(st.session_state["tz_data"]["pages"])
                st.text(full_text)

        with st.expander("Показать возвращенный JSON"):
            st.text(st.session_state["ai_raw_response"])


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("📊 Анализатор ТЗ → Excel")
    st.caption("Загрузите файл ТЗ (.pdf или .docx), выберите модель в боковой панели и получите таблицу требований.")

    uploaded_file = st.file_uploader("Загрузите файл ТЗ", type=["pdf", "docx"])

    if uploaded_file is not None:
        if st.button("🚀 Проанализировать", type="primary"):
            run_pipeline(uploaded_file)

    render_result()


if __name__ == "__main__":
    main()
