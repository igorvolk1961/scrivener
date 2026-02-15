"""
Абстрактный интерфейс для работы с векторным хранилищем.
Позволяет абстрагироваться от конкретной реализации (Qdrant, Pinecone, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStoreInterface(ABC):
    """
    Абстрактный интерфейс для работы с векторным хранилищем.
    
    Определяет методы, необходимые для работы с чанками:
    - получение чанков по фильтрам
    - работа с метаданными
    """
    
    @abstractmethod
    def get_chunks_by_filter(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по фильтрам.
        
        Args:
            filters: Словарь с фильтрами. Поддерживаемые ключи:
                - irvf_id: ID файла (обязательно)
                - section_number: Номер раздела (опционально, точное совпадение)
                - section_prefix: Префикс номера раздела (опционально, для LIKE запросов)
                - chunk_indices: Список индексов чанков (опционально)
                - chunk_index_range: Кортеж (min_index, max_index) для диапазона (опционально)
            limit: Максимальное количество результатов
        
        Returns:
            Список словарей с данными чанков в формате:
            {
                "text": str,
                "metadata": dict,
                "id": str (опционально)
            }
        """
        pass
    
    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Имя коллекции/индекса."""
        pass

