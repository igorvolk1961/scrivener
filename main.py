"""
Точка входа для запуска FastAPI приложения.
"""

import uvicorn
from utils.config import get_config
from utils.logging import setup_logging


def main():
    """Запуск FastAPI приложения."""
    # Загружаем конфигурацию
    config = get_config()
    
    # Настраиваем логирование
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(level=log_level)
    
    # Автозапуск локального Qdrant при необходимости
    from utils.qdrant_runner import ensure_qdrant_started
    qdrant_config = config.get("qdrant", {})
    ensure_qdrant_started(qdrant_config)
    
    # Получаем параметры API из конфигурации
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    debug = api_config.get("debug", False)
    
    # Запускаем сервер
    # Отключаем логирование uvicorn, так как мы используем loguru
    # Это предотвратит переопределение нашей конфигурации логирования
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level.lower(),
        log_config=None  # Отключаем конфигурацию логирования uvicorn
    )


if __name__ == "__main__":
    main()
