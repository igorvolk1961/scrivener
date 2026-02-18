"""
Chunk Horizontal Extension Tool для горизонтального расширения чанков.
Заполняет пропуски между имеющимися чанками и добавляет соседние чанки
для формирования непрерывного блока текста.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, List, Dict, Any, Set, Tuple, ClassVar

from pydantic import Field

from api.agents.base_tool import BaseTool

if TYPE_CHECKING:
    from api.agents.agent_definition import AgentConfig
    from api.agents.models import AgentContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _merge_index_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Объединяет пересекающиеся и соприкасающиеся диапазоны индексов.
    
    Args:
        ranges: Список кортежей (min_index, max_index)
    
    Returns:
        Список объединенных диапазонов (может быть один или несколько)
    
    Примеры:
        [(1, 5), (3, 8)] -> [(1, 8)]  # пересекающиеся
        [(1, 5), (6, 10)] -> [(1, 10)]  # соприкасающиеся
        [(1, 5), (10, 15)] -> [(1, 5), (10, 15)]  # отдельные
    """
    if not ranges:
        return []
    
    # Сортировка по min_index
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    
    for current_min, current_max in sorted_ranges[1:]:
        last_min, last_max = merged[-1]
        
        # Если диапазоны пересекаются или соприкасаются (разница <= 1)
        if current_min <= last_max + 1:
            # Объединяем диапазоны
            merged[-1] = (last_min, max(last_max, current_max))
        else:
            # Добавляем новый диапазон
            merged.append((current_min, current_max))
    
    return merged


def _get_chunk_key(chunk: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    Возвращает уникальный ключ для чанка.
    
    Args:
        chunk: Словарь с данными чанка
    
    Returns:
        Кортеж (irvf_id, section_number, chunk_index)
    """
    metadata = chunk.get("metadata", {})
    return (
        metadata.get("irvf_id", ""),
        metadata.get("section_number", ""),
        metadata.get("chunk_index", -1)
    )


class ChunkHorizontalExtensionTool(BaseTool):
    """
    Горизонтальное расширение: заполняет пропуски между имеющимися чанками
    и добавляет соседние чанки для формирования непрерывного блока текста.
    Принимает список всех имеющихся чанков, возвращает дополненный список.
    """

    tool_name: ClassVar[str] = "chunk_horizontal_extension"
    description: ClassVar[str] = __doc__

    reasoning: str = Field(description="Why horizontal extension is needed")
    chunks: List[Dict[str, Any]] = Field(description="List of existing chunks to extend")

    # Кэш для переиспользуемых компонентов (на уровне класса)
    _vector_store_cache: ClassVar[Dict[str, Any]] = {}
    _config_cache: ClassVar[Any] = None

    async def __call__(
        self,
        context: "AgentContext",
        config: "AgentConfig",
        **kwargs
    ) -> str:
        """
        Выполняет горизонтальное расширение чанков.
        
        Args:
            context: Контекст агента
            config: Конфигурация агента
        
        Returns:
            JSON строка с результатами расширения
        """
        logger.info(f"Горизонтальное расширение чанков: {len(self.chunks)} исходных чанков")
        
        try:
            # Получаем параметры из custom_context
            vdb_url = self._get_vdb_url(context)
            if not vdb_url:
                return self._format_error(
                    "VDB URL не указан",
                    "Для выполнения горизонтального расширения необходимо указать vdb_url в параметрах запроса или custom_context"
                )
            
            # Инициализируем компоненты
            vector_store = self._get_vector_store_manager(vdb_url)
            chunk_repository = self._get_chunk_repository(vector_store)
            
            # Получаем лимит символов из конфигурации
            max_chars = config.get("expansion_max_chars", 3000) if hasattr(config, "get") else 3000
            
            # Группируем по файлу (irvf_id)
            by_file = defaultdict(list)
            for ch in self.chunks:
                file_id = ch.get("metadata", {}).get("irvf_id") or ch.get("irvf_id")
                if file_id:
                    by_file[file_id].append(ch)
            
            result_chunks = []
            total_chars = 0
            added_chunks_keys: Set[Tuple[str, str, int]] = set()
            
            # Добавляем исходные чанки в результат и отслеживаем их ключи
            for ch in self.chunks:
                chunk_key = _get_chunk_key(ch)
                if chunk_key not in added_chunks_keys:
                    result_chunks.append(ch)
                    added_chunks_keys.add(chunk_key)
                    total_chars += len(ch.get("text", ch.get("content", "")))
            
            # Обрабатываем каждый файл
            for file_id, file_chunks in by_file.items():
                # Группируем по section_number
                by_section = defaultdict(list)
                for ch in file_chunks:
                    section = ch.get("metadata", {}).get("section_number", "")
                    by_section[section].append(ch)
                
                # Обрабатываем каждый section_number отдельно
                for section_number, section_chunks in by_section.items():
                    if not section_number:
                        # Если нет section_number, просто добавляем как есть
                        continue
                    
                    # Сортируем по индексу
                    section_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", x.get("chunk_index", 0)))
                    indices = [
                        c.get("metadata", {}).get("chunk_index", c.get("chunk_index"))
                        for c in section_chunks
                        if c.get("metadata", {}).get("chunk_index") is not None or c.get("chunk_index") is not None
                    ]
                    
                    if not indices:
                        continue
                    
                    min_idx = min(indices)
                    max_idx = max(indices)
                    existing_indices = set(indices)
                    
                    # 1. Заполняем пропуски между min_idx и max_idx
                    missing = [i for i in range(min_idx, max_idx + 1) if i not in existing_indices]
                    if missing:
                        missing_chunks = chunk_repository.get_chunks_by_indices(
                            file_id, section_number, missing
                        )
                        for new_chunk in missing_chunks:
                            chunk_key = _get_chunk_key(new_chunk)
                            if chunk_key not in added_chunks_keys:
                                chunk_text = new_chunk.get("text", "")
                                if total_chars + len(chunk_text) <= max_chars:
                                    result_chunks.append(new_chunk)
                                    added_chunks_keys.add(chunk_key)
                                    total_chars += len(chunk_text)
                        
                        # Обновляем границы после заполнения пропусков
                        all_indices = existing_indices | set(missing)
                        if all_indices:
                            min_idx = min(all_indices)
                            max_idx = max(all_indices)
                    
                    # 2. Пытаемся расширить диапазон назад
                    current_min = min_idx
                    while total_chars < max_chars:
                        next_idx = current_min - 1
                        if next_idx < 0:
                            break
                        
                        more = chunk_repository.get_chunks_by_indices(file_id, section_number, [next_idx])
                        if not more:
                            break
                        
                        new_chunk = more[0]
                        chunk_key = _get_chunk_key(new_chunk)
                        if chunk_key in added_chunks_keys:
                            break
                        
                        chunk_text = new_chunk.get("text", "")
                        if total_chars + len(chunk_text) <= max_chars:
                            result_chunks.append(new_chunk)
                            added_chunks_keys.add(chunk_key)
                            total_chars += len(chunk_text)
                            current_min = next_idx
                        else:
                            break
                    
                    # 3. Пытаемся расширить диапазон вперёд
                    current_max = max_idx
                    while total_chars < max_chars:
                        next_idx = current_max + 1
                        more = chunk_repository.get_chunks_by_indices(file_id, section_number, [next_idx])
                        if not more:
                            break
                        
                        new_chunk = more[0]
                        chunk_key = _get_chunk_key(new_chunk)
                        if chunk_key in added_chunks_keys:
                            break
                        
                        chunk_text = new_chunk.get("text", "")
                        if total_chars + len(chunk_text) <= max_chars:
                            result_chunks.append(new_chunk)
                            added_chunks_keys.add(chunk_key)
                            total_chars += len(chunk_text)
                            current_max = next_idx
                        else:
                            break
            
            # Сортируем итоговый список (по документу, section_number, индексу)
            result_chunks.sort(key=lambda x: (
                x.get("metadata", {}).get("irvf_id", x.get("irvf_id", "")),
                x.get("metadata", {}).get("section_number", ""),
                x.get("metadata", {}).get("chunk_index", x.get("chunk_index", 0))
            ))
            
            return json.dumps({
                "original_count": len(self.chunks),
                "final_count": len(result_chunks),
                "total_chars": total_chars,
                "max_chars_limit": max_chars,
                "chunks": result_chunks,
                "note": "Горизонтальное расширение выполнено"
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.exception(f"Ошибка при горизонтальном расширении чанков: {e}")
            return self._format_error(
                "Ошибка при горизонтальном расширении",
                str(e)
            )
    
    def _get_vdb_url(self, context: "AgentContext") -> str | None:
        """Получение vdb_url из контекста."""
        try:
            context_dict = context.model_dump()
            custom_context_value = context_dict.get("custom_context")
            
            if custom_context_value is not None:
                if isinstance(custom_context_value, dict):
                    return custom_context_value.get("vdb_url")
                elif hasattr(custom_context_value, "model_dump"):
                    custom_dict = custom_context_value.model_dump()
                    if isinstance(custom_dict, dict):
                        return custom_dict.get("vdb_url")
        except Exception as e:
            logger.debug(f"Ошибка при извлечении vdb_url: {e}")
        
        return None
    
    def _get_vector_size(self, normalized_url: str, qdrant_config: dict) -> int:
        """
        Получение размера вектора из существующей коллекции или из embedding объекта.
        
        Args:
            normalized_url: Нормализованный URL Qdrant
            qdrant_config: Конфигурация Qdrant
        
        Returns:
            Размер вектора эмбеддинга
        """
        from qdrant_client import QdrantClient
        
        # Сначала пробуем получить размер из существующей коллекции
        try:
            client = QdrantClient(
                url=normalized_url,
                api_key=qdrant_config.get("api_key"),
                timeout=qdrant_config.get("timeout", 30)
            )
            collection_name = qdrant_config.get("collection_name", "scrivener_documents")
            collection_info = client.get_collection(collection_name)
            
            if hasattr(collection_info, 'config'):
                config = collection_info.config
                if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                    vectors_config = config.params.vectors
                    if hasattr(vectors_config, 'size'):
                        vector_size = vectors_config.size
                        logger.debug(f"Размер вектора получен из существующей коллекции: {vector_size}")
                        return int(vector_size)
        except Exception as e:
            logger.debug(f"Не удалось получить размер из существующей коллекции: {e}")
        
        # Если коллекция не существует, создаем embedding объект для получения размера
        try:
            from rag.giga_embeddings import GigaEmbedding
            import os
            
            # Создаем временный embedding объект для получения размера
            credentials = os.getenv("GIGACHAT_AUTH_KEY")
            if credentials:
                embeddings_config = self._get_cached_config().get("embeddings", {}).get("giga", {})
                embedding = GigaEmbedding(
                    credentials=credentials,
                    scope=embeddings_config.get("scope", "GIGACHAT_API_PERS"),
                    api_url=embeddings_config.get("api_url", "https://gigachat.devices.sberbank.ru/api/v1"),
                    model=embeddings_config.get("model", "Embeddings"),
                    batch_size=embeddings_config.get("batch_size", 10),
                    max_retries=embeddings_config.get("max_retries", 3),
                    timeout=embeddings_config.get("timeout", 60)
                )
                from rag.vector_store import get_embedding_dimension
                vector_size = get_embedding_dimension(embedding)
                logger.debug(f"Размер вектора получен из embedding объекта: {vector_size}")
                return vector_size
        except Exception as e:
            logger.warning(f"Не удалось получить размер из embedding объекта: {e}")
        
        # Fallback на конфиг
        vector_size = qdrant_config.get("vector_size", 1024)
        logger.warning(f"Используется размер вектора из конфига (fallback): {vector_size}")
        return vector_size
    
    def _get_cached_config(self):
        """Получение конфигурации с кэшированием."""
        if ChunkHorizontalExtensionTool._config_cache is None:
            from utils.config import get_config
            ChunkHorizontalExtensionTool._config_cache = get_config()
        return ChunkHorizontalExtensionTool._config_cache
    
    def _get_vector_store_manager(self, vdb_url: str):
        """Получение или создание адаптера векторного хранилища с кэшированием."""
        from rag.vector_store import QdrantVectorStoreManager
        from rag.qdrant_adapter import QdrantAdapter
        from utils.config import get_config
        
        # Загрузка конфигурации для qdrant (кэшируется)
        if ChunkHorizontalExtensionTool._config_cache is None:
            ChunkHorizontalExtensionTool._config_cache = get_config()
        config = ChunkHorizontalExtensionTool._config_cache
        
        # Извлечение конфигурации Qdrant
        qdrant_config = {}
        try:
            if isinstance(config, dict):
                qdrant_config = config.get("qdrant", {})
            else:
                qdrant_section = getattr(config, "qdrant", None)
                if qdrant_section is not None:
                    if hasattr(qdrant_section, "model_dump"):
                        qdrant_config = qdrant_section.model_dump()
                    elif isinstance(qdrant_section, dict):
                        qdrant_config = qdrant_section
        except Exception:
            pass
        
        if not isinstance(qdrant_config, dict):
            qdrant_config = {}
        
        normalized_url = vdb_url.strip().rstrip("/")
        if not normalized_url.startswith("http"):
            normalized_url = f"http://{normalized_url}"
        
        # Получаем размер вектора из embedding объекта или из существующей коллекции
        vector_size = self._get_vector_size(normalized_url, qdrant_config)
        
        vector_store_cache_key = f"{normalized_url}:{qdrant_config.get('collection_name', 'scrivener_documents')}:{vector_size}"
        
        if vector_store_cache_key not in ChunkHorizontalExtensionTool._vector_store_cache:
            # Создаем QdrantVectorStoreManager
            vector_store_manager = QdrantVectorStoreManager(
                url=normalized_url,
                api_key=qdrant_config.get("api_key"),
                collection_name=qdrant_config.get("collection_name", "scrivener_documents"),
                vector_size=vector_size,
                timeout=qdrant_config.get("timeout", 30)
            )
            
            vector_store_manager.ensure_collection_exists()
            
            # Создаем адаптер
            adapter = QdrantAdapter(vector_store_manager)
            
            ChunkHorizontalExtensionTool._vector_store_cache[vector_store_cache_key] = adapter
            logger.debug(f"Создан новый адаптер Qdrant для {normalized_url} (кэширован)")
        
        return ChunkHorizontalExtensionTool._vector_store_cache[vector_store_cache_key]
    
    def _get_chunk_repository(self, vector_store):
        """Создание ChunkRepository."""
        from rag.chunk_repository import ChunkRepository
        return ChunkRepository(vector_store)
    
    def _format_error(self, error: str, detail: str) -> str:
        """Форматирование ошибки в JSON."""
        return json.dumps({
            "error": error,
            "detail": detail,
            "original_count": len(self.chunks),
            "final_count": len(self.chunks),
            "chunks": self.chunks,
            "note": "Ошибка при горизонтальном расширении"
        }, ensure_ascii=False)

