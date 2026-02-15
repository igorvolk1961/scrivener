"""
Настройка логирования для RAG-системы.
"""

import sys
import logging
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
    clear_log_on_start: bool = True
) -> None:
    """
    Настройка логирования с использованием loguru.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат логов (если None, используется стандартный)
        log_file: Путь к файлу для сохранения логов (если None, только консоль)
        clear_log_on_start: Очищать лог-файл при старте приложения (по умолчанию True)
    """
    # Удаляем стандартный обработчик
    logger.remove()
    
    # Перехватываем все сообщения стандартного logging и перенаправляем в loguru
    class InterceptHandler(logging.Handler):
        def __init__(self, min_level):
            super().__init__()
            self.min_level = min_level
        
        def emit(self, record):
            # КРИТИЧЕСКИ ВАЖНО: Фильтруем ВСЕ DEBUG сообщения от внешних библиотек
            # независимо от того, что думает объект Trace или логгер
            # Это последняя линия защиты от DEBUG сообщений
            # Проверяем имя логгера ПЕРВЫМ делом, до любых других проверок
            # Список библиотек, которые могут логировать на DEBUG: httpcore, httpx, openai, urllib3, matplotlib
            if record.name.startswith(("httpcore", "httpx", "openai", "urllib3", "matplotlib")):
                # Для внешних библиотек применяем строгую фильтрацию
                # Игнорируем ВСЕ сообщения уровня ниже min_level
                if record.levelno < self.min_level:
                    # Отладочное логирование (можно удалить после решения проблемы)
                    # print(f"FILTERED external library: {record.name} {record.levelname} {record.getMessage()}")
                    return  # Полностью игнорируем DEBUG сообщения от внешних библиотек
            
            # Также фильтруем DEBUG сообщения от rag.* логгеров (если они не нужны)
            if record.name.startswith("rag.") and record.levelno < self.min_level:
                return  # Фильтруем DEBUG сообщения от rag.* логгеров
            
            # Фильтруем сообщения по уровню перед передачей в loguru
            if record.levelno < self.min_level:
                return
            # Получаем соответствующий уровень loguru
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Находием фрейм, откуда был вызван логгер
            frame, depth = sys._getframe(6), 6
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
    
    # Устанавливаем перехватчик для всех стандартных логгеров
    # Устанавливаем уровень стандартного logging в соответствии с требуемым уровнем
    # Это предотвратит появление DEBUG сообщений от библиотек (например, httpcore)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Получаем корневой логгер и очищаем все существующие handlers ДО настройки
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(numeric_level)
    
    # Создаем InterceptHandler с фильтрацией по уровню
    intercept_handler = InterceptHandler(min_level=numeric_level)
    logging.basicConfig(handlers=[intercept_handler], level=numeric_level, force=True)
    
    # Убеждаемся, что корневой логгер использует только наш handler
    root_logger.handlers = [intercept_handler]
    root_logger.setLevel(numeric_level)
    
    # Устанавливаем уровень для родительских логгеров внешних библиотек
    # Это нужно, чтобы isEnabledFor(DEBUG) возвращал False при создании объектов Trace
    # Достаточно установить только для родительских логгеров - дочерние наследуют уровень
    for parent_logger_name in ["httpcore", "httpx", "openai", "urllib3", "matplotlib"]:
        parent_logger = logging.getLogger(parent_logger_name)
        parent_logger.setLevel(numeric_level)
        parent_logger.handlers = []
        parent_logger.propagate = True
    
    # Формат по умолчанию
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Добавляем обработчик для консоли с фильтрацией DEBUG сообщений от внешних библиотек
    def filter_external_library_debug(record):
        """Фильтр для loguru, который отфильтровывает DEBUG сообщения от внешних библиотек."""
        # КРИТИЧЕСКИ ВАЖНО: Фильтруем ВСЕ DEBUG сообщения от внешних библиотек
        # Проверяем имя логгера (может быть "httpcore._trace", "httpcore.http11", "openai._base_client", "urllib3.connectionpool", "matplotlib" и т.д.)
        logger_name = record.get("name", "")
        if logger_name.startswith(("httpcore", "httpx", "openai", "urllib3", "matplotlib")):
            # Если это DEBUG сообщение от внешних библиотек, фильтруем его
            # В loguru record["level"] - это объект Level, у которого есть атрибут name и no
            level = record.get("level")
            if level:
                level_name = level.name if hasattr(level, "name") else str(level)
                level_no = level.no if hasattr(level, "no") else 0
                # Фильтруем DEBUG сообщения (level_name == "DEBUG" или level_no < numeric_level)
                if level_name == "DEBUG" or level_no < numeric_level:
                    return False
        # Также фильтруем DEBUG сообщения от rag.* логгеров
        if logger_name.startswith("rag."):
            level = record.get("level")
            if level:
                level_name = level.name if hasattr(level, "name") else str(level)
                level_no = level.no if hasattr(level, "no") else 0
                if level_name == "DEBUG" or level_no < numeric_level:
                    return False
        return True
    
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        filter=filter_external_library_debug  # Добавляем фильтр для loguru
    )
    
    # Добавляем обработчик для файла (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Очистка лог-файла при старте, если указано
        if clear_log_on_start and log_path.exists():
            log_path.unlink()
        
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            filter=filter_external_library_debug  # Добавляем фильтр для loguru
        )
    
    logger.info(f"Логирование настроено. Уровень: {level}")


def get_logger(name: str):
    """
    Получение логгера с указанным именем.
    
    Args:
        name: Имя логгера (обычно __name__)
    
    Returns:
        Логгер loguru
    """
    return logger.bind(name=name)

