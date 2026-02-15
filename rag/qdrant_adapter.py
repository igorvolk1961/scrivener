"""
Адаптер Qdrant для работы через VectorStoreInterface.
Реализует абстрактный интерфейс для Qdrant.
"""

import logging
from typing import List, Dict, Any, Optional

from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from rag.vector_store_interface import VectorStoreInterface
from rag.vector_store import QdrantVectorStoreManager

logger = logging.getLogger(__name__)


class QdrantAdapter(VectorStoreInterface):
    """
    Адаптер для Qdrant, реализующий VectorStoreInterface.
    
    Инкапсулирует работу с Qdrant API и преобразует вызовы
    в формат, понятный Qdrant.
    """
    
    def __init__(self, vector_store_manager: QdrantVectorStoreManager):
        """
        Инициализация адаптера.
        
        Args:
            vector_store_manager: Менеджер векторного хранилища Qdrant
        """
        self._vector_store_manager = vector_store_manager
        self._collection_name = vector_store_manager.collection_name
    
    @property
    def collection_name(self) -> str:
        """Имя коллекции."""
        return self._collection_name
    
    def get_chunks_by_filter(
        self,
        filters: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение чанков по фильтрам через Qdrant.
        
        Args:
            filters: Словарь с фильтрами
            limit: Максимальное количество результатов
        
        Returns:
            Список словарей с данными чанков
        """
        try:
            # Строим фильтр Qdrant
            search_filter = self._build_qdrant_filter(filters)
            
            # Определяем лимит
            if limit is None:
                # Если есть chunk_indices, используем их количество
                if "chunk_indices" in filters and filters["chunk_indices"]:
                    limit = len(filters["chunk_indices"]) * 2
                elif "chunk_index_range" in filters and filters["chunk_index_range"]:
                    min_idx, max_idx = filters["chunk_index_range"]
                    limit = (max_idx - min_idx + 1) * 2
                else:
                    limit = 10000  # Большой лимит по умолчанию
            
            # Получаем чанки через scroll
            results = self._vector_store_manager.client.scroll(
                collection_name=self._collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Преобразуем результаты в нужный формат
            chunks = []
            for point in results[0]:
                payload = point.payload if point.payload else {}
                text = payload.get("text", "")
                
                chunk_data = {
                    "text": text,
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "id": str(point.id) if hasattr(point, 'id') else None
                }
                
                # Дополнительная фильтрация на стороне Python (для section_prefix и chunk_indices)
                if self._matches_filters(chunk_data, filters):
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(
                f"Ошибка при получении чанков через Qdrant адаптер: {e}",
                exc_info=True
            )
            return []
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """
        Построение фильтра Qdrant из словаря фильтров.
        
        Args:
            filters: Словарь с фильтрами
        
        Returns:
            Объект Filter для Qdrant или None
        """
        must_conditions = []
        
        # irvf_id - обязательный фильтр
        if "irvf_id" in filters and filters["irvf_id"]:
            must_conditions.append(
                FieldCondition(
                    key="irvf_id",
                    match=MatchValue(value=filters["irvf_id"])
                )
            )
        
        # section_number - точное совпадение (если указан, но не указан section_prefix)
        if "section_number" in filters and filters["section_number"] and "section_prefix" not in filters:
            must_conditions.append(
                FieldCondition(
                    key="section_number",
                    match=MatchValue(value=filters["section_number"])
                )
            )
        
        # chunk_index_range - диапазон индексов
        if "chunk_index_range" in filters and filters["chunk_index_range"]:
            min_idx, max_idx = filters["chunk_index_range"]
            must_conditions.append(
                FieldCondition(
                    key="chunk_index",
                    range=Range(gte=min_idx, lte=max_idx)
                )
            )
        # chunk_indices - список индексов (через should для OR условий)
        elif "chunk_indices" in filters and filters["chunk_indices"]:
            indices = filters["chunk_indices"]
            if len(indices) == 1:
                # Один индекс - простое условие
                must_conditions.append(
                    FieldCondition(
                        key="chunk_index",
                        match=MatchValue(value=indices[0])
                    )
                )
            else:
                # Несколько индексов - используем should
                chunk_conditions = [
                    FieldCondition(
                        key="chunk_index",
                        match=MatchValue(value=idx)
                    )
                    for idx in indices
                ]
                # Возвращаем фильтр с should
                return Filter(
                    must=must_conditions,
                    should=chunk_conditions,
                    min_should=1
                )
        
        # section_prefix - обрабатывается на стороне Python, но можем добавить фильтр по section_number если нужно
        # (для оптимизации можно добавить фильтр, но основная фильтрация будет в _matches_filters)
        
        if must_conditions:
            return Filter(must=must_conditions)
        return None
    
    def _matches_filters(self, chunk_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Дополнительная проверка фильтров на стороне Python.
        Используется для фильтров, которые сложно выразить в Qdrant (например, section_prefix).
        
        Args:
            chunk_data: Данные чанка
            filters: Словарь с фильтрами
        
        Returns:
            True, если чанк соответствует всем фильтрам
        """
        metadata = chunk_data.get("metadata", {})
        
        # Проверка section_prefix (LIKE запрос)
        if "section_prefix" in filters and filters["section_prefix"]:
            section_number = metadata.get("section_number", "")
            if not section_number or not section_number.startswith(filters["section_prefix"]):
                return False
        
        # Проверка chunk_indices (если указан список)
        if "chunk_indices" in filters and filters["chunk_indices"]:
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None or chunk_index not in filters["chunk_indices"]:
                return False
        
        # Проверка chunk_index_range (если указан диапазон)
        if "chunk_index_range" in filters and filters["chunk_index_range"]:
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                return False
            min_idx, max_idx = filters["chunk_index_range"]
            if not (min_idx <= chunk_index <= max_idx):
                return False
        
        return True

