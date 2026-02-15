"""
Модуль для работы с чанками в векторном хранилище.
Абстракция для получения чанков по различным критериям.
"""

import logging
from typing import List, Dict, Any

from rag.vector_store_interface import VectorStoreInterface

logger = logging.getLogger(__name__)


class ChunkRepository:
    """
    Репозиторий для работы с чанками в векторном хранилище.
    
    Предоставляет методы для получения чанков по различным критериям,
    изолируя детали работы с конкретной БД от бизнес-логики инструментов.
    """
    
    def __init__(self, vector_store: VectorStoreInterface):
        """
        Инициализация репозитория.
        
        Args:
            vector_store: Адаптер векторного хранилища, реализующий VectorStoreInterface
        """
        self.vector_store = vector_store
        self.collection_name = vector_store.collection_name
    
    def get_chunks_by_indices(
        self,
        irvf_id: str,
        section_number: str,
        indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по irvf_id, section_number и списку индексов.
        
        Args:
            irvf_id: ID файла
            section_number: Номер раздела
            indices: Список индексов чанков
        
        Returns:
            Список словарей с данными чанков в формате:
            {
                "text": str,
                "metadata": dict,
                "id": str (опционально)
            }
        """
        if not indices:
            return []
        
        try:
            filters = {
                "irvf_id": irvf_id,
                "section_number": section_number,
                "chunk_indices": indices
            }
            
            chunks = self.vector_store.get_chunks_by_filter(filters, limit=len(indices) * 2)
            
            logger.debug(
                f"Получено {len(chunks)} чанков для irvf_id={irvf_id}, "
                f"section_number={section_number}, indices={indices}"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении чанков по индексам: {e}",
                exc_info=True
            )
            return []
    
    def get_chunks_by_section_prefix(
        self,
        irvf_id: str,
        section_prefix: str
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по irvf_id и префиксу section_number.
        
        Args:
            irvf_id: ID файла
            section_prefix: Префикс номера раздела (например, "6.4" для "6.4.1", "6.4.2", etc.)
        
        Returns:
            Список словарей с данными чанков
        """
        try:
            filters = {
                "irvf_id": irvf_id,
                "section_prefix": section_prefix
            }
            
            chunks = self.vector_store.get_chunks_by_filter(filters, limit=10000)
            
            logger.debug(
                f"Получено {len(chunks)} чанков для irvf_id={irvf_id}, "
                f"section_prefix={section_prefix}"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении чанков по префиксу раздела: {e}",
                exc_info=True
            )
            return []
    
    def get_chunks_by_section_and_index_range(
        self,
        irvf_id: str,
        section_number: str,
        min_index: int,
        max_index: int
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по irvf_id, section_number и диапазону индексов.
        
        Args:
            irvf_id: ID файла
            section_number: Номер раздела
            min_index: Минимальный индекс (включительно)
            max_index: Максимальный индекс (включительно)
        
        Returns:
            Список словарей с данными чанков
        """
        try:
            filters = {
                "irvf_id": irvf_id,
                "section_number": section_number,
                "chunk_index_range": (min_index, max_index)
            }
            
            chunks = self.vector_store.get_chunks_by_filter(
                filters,
                limit=(max_index - min_index + 1) * 2
            )
            
            logger.debug(
                f"Получено {len(chunks)} чанков для irvf_id={irvf_id}, "
                f"section_number={section_number}, range=[{min_index}, {max_index}]"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении чанков по диапазону индексов: {e}",
                exc_info=True
            )
            return []

