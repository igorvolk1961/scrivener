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
    
    def get_chunks_metadata_by_indices(
        self,
        chunk_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Получение метаданных чанков только по списку индексов (без irvf_id и section_number).
        Используется для получения полных метаданных по chunk_index из векторной БД.
        
        Args:
            chunk_indices: Список индексов чанков
            
        Returns:
            Список словарей с метаданными: {"irvf_id": str, "section_number": str, "chunk_index": int}
        """
        if not chunk_indices:
            return []
        
        try:
            # Ищем чанки только по chunk_index (без фильтра по irvf_id и section_number)
            filters = {
                "chunk_indices": chunk_indices
            }
            
            chunks = self.vector_store.get_chunks_by_filter(filters, limit=len(chunk_indices) * 2)
            
            # Извлекаем только метаданные
            metadata_list = []
            found_indices = set()
            
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                chunk_index = metadata.get("chunk_index")
                
                if chunk_index is not None and chunk_index in chunk_indices and chunk_index not in found_indices:
                    metadata_list.append({
                        "irvf_id": metadata.get("irvf_id", ""),
                        "section_number": metadata.get("section_number", ""),
                        "chunk_index": chunk_index
                    })
                    found_indices.add(chunk_index)
            
            logger.debug(
                f"Получено {len(metadata_list)} метаданных для индексов {chunk_indices}"
            )
            
            return metadata_list
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении метаданных по индексам: {e}",
                exc_info=True
            )
            return []
    
    def get_chunks_by_ids(
        self,
        chunk_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по их уникальным ID (point.id из векторной БД).
        
        Args:
            chunk_ids: Список уникальных ID чанков (UUID строки)
        
        Returns:
            Список словарей с данными чанков в формате:
            {
                "text": str,
                "metadata": dict,
                "id": str
            }
        """
        if not chunk_ids:
            return []
        
        try:
            # Используем метод адаптера для получения по ID
            if hasattr(self.vector_store, 'get_chunks_by_ids'):
                chunks = self.vector_store.get_chunks_by_ids(chunk_ids)
            else:
                logger.warning("Адаптер не поддерживает get_chunks_by_ids, возвращаем пустой список")
                return []
            
            logger.debug(
                f"Получено {len(chunks)} чанков для {len(chunk_ids)} ID"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении чанков по ID: {e}",
                exc_info=True
            )
            return []

