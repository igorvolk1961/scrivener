"""
Chunk Vertical Extension Tool для вертикального расширения чанков.
Добавляет чанки из иерархически связанных разделов для формирования
полной картины по теме.
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


def _get_related_section_prefixes(section: str) -> Set[str]:
    """
    Возвращает набор префиксов разделов, связанных с данным.
    Включает сам раздел (для поиска всех его подразделов), родителя,
    и префикс для поиска соседей (родитель + '.').
    
    Args:
        section: Номер раздела (например, "6.4.1")
    
    Returns:
        Множество связанных префиксов
    """
    related = set()
    parts = section.split('.')
    
    # Сам раздел — для получения всех его подразделов
    related.add(section)
    
    # Родитель
    if len(parts) > 1:
        parent = '.'.join(parts[:-1])
        related.add(parent)
        # Для соседей того же уровня используем родительский префикс
        related.add(parent + '.')  # все подразделы родителя
    
    return related


def _prefix_length(s1: str, s2: str) -> int:
    """
    Длина общего префикса (по количеству совпадающих частей до точки).
    
    Args:
        s1: Первый номер раздела
        s2: Второй номер раздела
    
    Returns:
        Количество совпадающих частей
    """
    p1 = s1.split('.')
    p2 = s2.split('.')
    i = 0
    while i < min(len(p1), len(p2)) and p1[i] == p2[i]:
        i += 1
    return i


class ChunkVerticalExtensionTool(BaseTool):
    """
    Вертикальное расширение: добавляет чанки из иерархически связанных разделов.
    Анализирует имеющиеся чанки и дополняет недостающие разделы (родительские,
    дочерние, соседние) для формирования полной картины по теме.
    """

    tool_name: ClassVar[str] = "chunk_vertical_extension"
    description: ClassVar[str] = __doc__

    reasoning: str = Field(description="Why vertical extension is needed")
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
        Выполняет вертикальное расширение чанков.
        
        Args:
            context: Контекст агента
            config: Конфигурация агента
        
        Returns:
            JSON строка с результатами расширения
        """
        logger.info(f"Вертикальное расширение чанков: {len(self.chunks)} исходных чанков")
        
        try:
            # Получаем параметры из custom_context
            vdb_url = self._get_vdb_url(context)
            if not vdb_url:
                return self._format_error(
                    "VDB URL не указан",
                    "Для выполнения вертикального расширения необходимо указать vdb_url в параметрах запроса или custom_context"
                )
            
            # Инициализируем компоненты
            vector_store = self._get_vector_store_manager(vdb_url)
            chunk_repository = self._get_chunk_repository(vector_store)
            
            # Получаем лимит символов из конфигурации
            max_chars = config.get("expansion_max_chars", 4000) if hasattr(config, "get") else 4000
            
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
                # Множество уже представленных разделов
                present_sections = {
                    c.get("metadata", {}).get("section_number", c.get("section_number", ""))
                    for c in file_chunks
                    if c.get("metadata", {}).get("section_number") or c.get("section_number")
                }
                
                # Определяем диапазоны индексов для каждого имеющегося раздела
                section_ranges: Dict[str, Tuple[int, int]] = {}
                for ch in file_chunks:
                    section = ch.get("metadata", {}).get("section_number", ch.get("section_number", ""))
                    if not section:
                        continue
                    
                    chunk_index = ch.get("metadata", {}).get("chunk_index", ch.get("chunk_index"))
                    if chunk_index is None:
                        continue
                    
                    if section not in section_ranges:
                        section_ranges[section] = (chunk_index, chunk_index)
                    else:
                        min_idx, max_idx = section_ranges[section]
                        section_ranges[section] = (min(min_idx, chunk_index), max(max_idx, chunk_index))
                
                # Для каждого представленного раздела определяем связанные (недостающие)
                target_sections = set()
                for sec in present_sections:
                    target_sections.update(_get_related_section_prefixes(sec))
                
                missing_prefixes = target_sections - present_sections
                
                if not missing_prefixes:
                    # Нет недостающих — просто добавляем имеющиеся чанки (уже добавлены)
                    continue
                
                # Получаем все недостающие чанки (по префиксам разделов)
                missing_chunks = []
                for prefix in missing_prefixes:
                    chunks_for_prefix = chunk_repository.get_chunks_by_section_prefix(file_id, prefix)
                    
                    # Фильтруем по диапазону индексов, если есть информация о диапазоне
                    # Используем диапазон из родительского или соседнего раздела
                    filtered_chunks = []
                    for ch in chunks_for_prefix:
                        chunk_section = ch.get("metadata", {}).get("section_number", "")
                        chunk_index = ch.get("metadata", {}).get("chunk_index", -1)
                        
                        # Ищем подходящий диапазон индексов
                        use_range = False
                        min_idx, max_idx = None, None
                        
                        # Проверяем, есть ли диапазон для этого раздела или родительского
                        if chunk_section in section_ranges:
                            min_idx, max_idx = section_ranges[chunk_section]
                            use_range = True
                        else:
                            # Ищем диапазон родительского раздела
                            parts = chunk_section.split('.')
                            for i in range(len(parts) - 1, 0, -1):
                                parent_section = '.'.join(parts[:i])
                                if parent_section in section_ranges:
                                    min_idx, max_idx = section_ranges[parent_section]
                                    use_range = True
                                    break
                        
                        # Если нашли диапазон, фильтруем по нему
                        if use_range and min_idx is not None and max_idx is not None:
                            if min_idx <= chunk_index <= max_idx:
                                filtered_chunks.append(ch)
                        else:
                            # Если диапазона нет, добавляем все чанки этого раздела
                            filtered_chunks.append(ch)
                    
                    missing_chunks.extend(filtered_chunks)
                
                # Оцениваем близость каждого недостающего чанка к имеющимся разделам
                scored = []
                for ch in missing_chunks:
                    ch_sec = ch.get("metadata", {}).get("section_number", "")
                    # Максимальная длина общего префикса с любым из имеющихся разделов
                    max_overlap = 0
                    for psec in present_sections:
                        overlap = _prefix_length(psec, ch_sec)
                        if overlap > max_overlap:
                            max_overlap = overlap
                    scored.append((max_overlap, ch))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                
                # Добавляем чанки в порядке близости, пока есть место
                for _, ch in scored:
                    chunk_key = _get_chunk_key(ch)
                    if chunk_key in added_chunks_keys:
                        continue
                    
                    chunk_text = ch.get("text", "")
                    if total_chars + len(chunk_text) <= max_chars:
                        result_chunks.append(ch)
                        added_chunks_keys.add(chunk_key)
                        total_chars += len(chunk_text)
                    else:
                        break
            
            # Финальная сортировка
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
                "note": "Вертикальное расширение выполнено"
            }, ensure_ascii=False)
            
        except Exception as e:
            logger.exception(f"Ошибка при вертикальном расширении чанков: {e}")
            return self._format_error(
                "Ошибка при вертикальном расширении",
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
    
    def _get_vector_store_manager(self, vdb_url: str):
        """Получение или создание адаптера векторного хранилища с кэшированием."""
        from rag.vector_store import QdrantVectorStoreManager
        from rag.qdrant_adapter import QdrantAdapter
        from utils.config import get_config
        
        # Загрузка конфигурации для qdrant (кэшируется)
        if ChunkVerticalExtensionTool._config_cache is None:
            ChunkVerticalExtensionTool._config_cache = get_config()
        config = ChunkVerticalExtensionTool._config_cache
        
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
        
        vector_store_cache_key = f"{normalized_url}:{qdrant_config.get('collection_name', 'scrivener_documents')}:{qdrant_config.get('vector_size', 1024)}"
        
        if vector_store_cache_key not in ChunkVerticalExtensionTool._vector_store_cache:
            # Создаем QdrantVectorStoreManager
            vector_store_manager = QdrantVectorStoreManager(
                url=normalized_url,
                api_key=qdrant_config.get("api_key"),
                collection_name=qdrant_config.get("collection_name", "scrivener_documents"),
                vector_size=qdrant_config.get("vector_size", 1024),
                timeout=qdrant_config.get("timeout", 30)
            )
            
            vector_store_manager.ensure_collection_exists()
            
            # Создаем адаптер
            adapter = QdrantAdapter(vector_store_manager)
            
            ChunkVerticalExtensionTool._vector_store_cache[vector_store_cache_key] = adapter
            logger.debug(f"Создан новый адаптер Qdrant для {normalized_url} (кэширован)")
        
        return ChunkVerticalExtensionTool._vector_store_cache[vector_store_cache_key]
    
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
            "note": "Ошибка при вертикальном расширении"
        }, ensure_ascii=False)

