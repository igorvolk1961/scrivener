"""
Менеджер для запуска и остановки эмулятора КФО.
"""

import threading
import time
from typing import Optional

import uvicorn
from loguru import logger


class CfxEmulatorManager:
    """Менеджер для управления эмулятором КФО."""
    
    def __init__(self, port: int = 8001):
        """
        Инициализация менеджера.
        
        Args:
            port: Порт для запуска эмулятора
        """
        self.port = port
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> bool:
        """
        Запускает эмулятор КФО в отдельном потоке.
        
        Returns:
            True если запуск успешен
        """
        if self._running:
            logger.warning("Эмулятор КФО уже запущен")
            return True
        
        try:
            from desktop.cfx_emulator.main import app
            
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.port,
                log_level="info",
            )
            self.server = uvicorn.Server(config)
            
            def run_server():
                self._running = True
                try:
                    self.server.run()
                except Exception as e:
                    logger.error(f"Ошибка запуска эмулятора КФО: {e}")
                finally:
                    self._running = False
            
            self.thread = threading.Thread(target=run_server, daemon=True)
            self.thread.start()
            
            # Ждем немного, чтобы сервер успел запуститься
            time.sleep(1)
            
            if self._running:
                logger.info(f"Эмулятор КФО запущен на порту {self.port}")
                return True
            else:
                logger.error("Не удалось запустить эмулятор КФО")
                return False
                
        except Exception as e:
            logger.exception(f"Ошибка при запуске эмулятора КФО: {e}")
            return False
    
    def stop(self):
        """Останавливает эмулятор КФО."""
        if not self._running:
            return
        
        if self.server:
            self.server.should_exit = True
        
        self._running = False
        logger.info("Эмулятор КФО остановлен")
    
    def is_running(self) -> bool:
        """Проверяет, запущен ли эмулятор."""
        return self._running
    
    def get_url(self) -> str:
        """Возвращает URL эмулятора."""
        return f"http://localhost:{self.port}"

