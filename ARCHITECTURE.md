# Архитектура проекта: AI TZ Analyzer (Streamlit)

## 1. Обзор системы
Приложение на Streamlit для парсинга файлов ТЗ (.pdf, .docx), анализа текста через API GPTunneL с выбором моделей и экспорта результатов в .xlsx.

## 2. Структура проекта
tz_analyzer/
├── app.py                      # UI-оркестратор: сайдбар, шаги, session_state
├── config.py                   # константы: список моделей, дефолтный промпт
├── requirements.txt
├── .streamlit/
│   └── secrets.toml            # секреты (не в git)
├── utils/
│   ├── __init__.py
│   ├── parsers.py              # извлечение текста из pdf/docx
│   ├── ai_client.py            # обёртка над GPTunneL API (OpenAI-совместимый)
│   └── excel_generator.py      # JSON ответа ИИ -> DataFrame -> xlsx
├── auth/                       # (будущее) модуль авторизации
│   └── __init__.py
├── history/                    # (будущее) история обработанных файлов
│   └── storage.py
└── tests/
    ├── test_parsers.py
    └── test_excel_generator.py

## 3. Глобальное состояние (st.session_state)
Ключ	Тип	Назначение
st.session_state = {
    "uploaded_file_bytes": bytes | None,   # сырые байты файла (для кэш-парсинга)
    "uploaded_file_name": str | None,
    "extracted_text": str | None,          # текст ТЗ после парсинга
    "selected_model": str,                 # id модели из селектора
    "system_prompt": str,                  # дефолт из config, позже — редактируемый
    "processing_status": str,              # "idle" | "parsing" | "analyzing" | "done" | "error"
    "error_message": str | None,
    "ai_raw_response": str | None,         # сырой JSON-ответ модели
    "result_df": pd.DataFrame | None,      # финальная таблица (источник правды для data_editor)
    "excel_bytes": bytes | None,           # сгенерированный xlsx, чтобы download_button не пересобирал файл
}

## 4. Формат обмена данными (ИИ -> Excel)
