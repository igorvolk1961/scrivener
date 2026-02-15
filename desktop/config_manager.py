"""
Менеджер конфигурации для десктопного приложения.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

CONFIG_FILE = Path("data/debug_config.json")


class ConfigManager:
    """Менеджер конфигурации приложения."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """Загружает конфигурацию из файла."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                logger.info(f"Конфигурация загружена из {CONFIG_FILE}")
            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации: {e}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
            self.save()
        
        return self.config
    
    def save(self):
        """Сохраняет конфигурацию в файл."""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            logger.info(f"Конфигурация сохранена в {CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение конфигурации."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Устанавливает значение конфигурации."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Обновляет конфигурацию."""
        self.config.update(updates)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            "api_url": "http://localhost:8000",
            "cfx_emulator_url": "http://localhost:8001",
            "llm": {
                "url": "",
                "api_key": "",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 8000,
            },
            "embeddings": {
                "api_key": "",
                "url": "https://gigachat.devices.sberbank.ru/api/v1",
                "model": "Embeddings",
                "batch_size": 10,
            },
            "qdrant": {
                "url": "http://localhost:6333",
            },
            "search": {
                "api_key": "",
                "url": "https://api.tavily.com",
            },
        }

