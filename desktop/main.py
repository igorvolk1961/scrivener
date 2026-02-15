"""
Точка входа для десктопного приложения Scrivener.
"""

import sys
import tkinter as tk
from pathlib import Path

from loguru import logger

from desktop.api_client import ScrivenerClient
from desktop.config_manager import ConfigManager
from desktop.cfx_emulator_manager import CfxEmulatorManager
from desktop.ui.main_window import MainWindow


def setup_logging():
    """Настройка логирования."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="INFO"
    )


def main():
    """Главная функция приложения."""
    setup_logging()
    logger.info("Запуск десктопного приложения Scrivener")
    
    # Проверяем существование директории для хранилища
    storage_dir = Path("data/debug_storage")
    storage_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Директория хранилища: {storage_dir.absolute()}")
    
    # Загружаем конфигурацию
    config_manager = ConfigManager()
    
    # Запускаем эмулятор КФО
    cfx_emulator_port = 8001
    cfx_emulator_manager = CfxEmulatorManager(port=cfx_emulator_port)
    
    logger.info("Запуск эмулятора КФО...")
    if not cfx_emulator_manager.start():
        logger.error("Не удалось запустить эмулятор КФО")
        print("Ошибка: Не удалось запустить эмулятор КФО")
        return 1
    
    cfx_emulator_url = cfx_emulator_manager.get_url()
    logger.info(f"Эмулятор КФО запущен: {cfx_emulator_url}")
    
    # Получаем URL API из конфигурации
    api_url = config_manager.get("api_url", "http://localhost:8000")
    
    # Создаем API клиент
    api_client = ScrivenerClient(api_url=api_url, cfx_emulator_url=cfx_emulator_url)
    
    # Создаем главное окно
    root = tk.Tk()
    
    def on_closing():
        """Обработчик закрытия окна."""
        logger.info("Закрытие приложения")
        cfx_emulator_manager.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        app = MainWindow(root, api_client, config_manager)
        logger.info("Приложение запущено")
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Прерывание пользователем")
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        return 1
    finally:
        cfx_emulator_manager.stop()
        logger.info("Приложение завершено")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

