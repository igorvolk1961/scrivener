"""
Модуль для работы с Qdrant векторным хранилищем.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CollectionStatus,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
import httpx

logger = logging.getLogger(__name__)


def get_embedding_dimension(embedding) -> int:
    """
    Получение размера вектора эмбеддинга из объекта эмбеддинга.
    
    ВАЖНО: Выполняет пробный запрос к API для определения размера ТОЛЬКО один раз,
    затем сохраняет результат в объекте эмбеддинга и использует его для всех последующих вызовов.
    Не использует жестко закодированные значения или привязки к конкретным моделям.
    
    Args:
        embedding: Объект эмбеддинга (GigaEmbedding или BaseEmbedding)
    
    Returns:
        Размер вектора эмбеддинга
    
    Raises:
        ValueError: Если не удалось определить размер вектора через пробный запрос
    """
    # Проверяем, есть ли уже сохраненный размер вектора в объекте эмбеддинга
    if hasattr(embedding, '_cached_vector_dimension') and embedding._cached_vector_dimension is not None:
        logger.debug(f"Используется сохраненный размер вектора из предыдущего пробного запроса: {embedding._cached_vector_dimension}")
        return embedding._cached_vector_dimension
    
    # Выполняем пробный запрос к API для определения размера
    # Это единственный надежный способ определить реальный размер вектора
    try:
        # Пробуем получить эмбеддинг для тестового текста
        test_embedding = embedding.get_query_embedding("test")
        if test_embedding and isinstance(test_embedding, list):
            actual_dim = len(test_embedding)
            logger.info(f"Размер вектора определен из пробного запроса к API: {actual_dim}")
            
            # Сохраняем результат пробного запроса в объекте эмбеддинга для дальнейшего использования
            object.__setattr__(embedding, '_cached_vector_dimension', actual_dim)
            # Также обновляем embedding_dim для совместимости
            if hasattr(embedding, 'embedding_dim'):
                object.__setattr__(embedding, 'embedding_dim', actual_dim)
            if hasattr(embedding, 'embed_dim'):
                object.__setattr__(embedding, 'embed_dim', actual_dim)
            
            return actual_dim
    except Exception as e:
        logger.error(f"Не удалось определить размер вектора через пробный запрос к API: {e}")
        raise ValueError(
            f"Не удалось определить размер вектора эмбеддинга через пробный запрос к API. "
            f"Ошибка: {e}"
        )
    
    # Если запрос не вернул список, выбрасываем ошибку
    raise ValueError(
        f"Не удалось определить размер вектора эмбеддинга из объекта {type(embedding).__name__}. "
        "Пробный запрос к API не вернул валидный вектор."
    )


class QdrantVectorStoreManager:
    """
    Менеджер для работы с Qdrant векторным хранилищем.
    
    Управляет подключением, созданием коллекций и настройкой индексов.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = "scrivener_documents",
        vector_size: Optional[int] = None,
        timeout: int = 30
    ):
        """
        Инициализация менеджера Qdrant.
        
        Args:
            url: URL Qdrant сервера
            api_key: API ключ (если требуется)
            collection_name: Имя коллекции
            vector_size: Размер вектора эмбеддинга (может быть None, будет определен через пробный запрос к API)
            timeout: Таймаут подключения
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        # Создание клиента Qdrant
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout
        )
        
        logger.info(
            f"QdrantVectorStoreManager инициализирован: url={url}, "
            f"collection={collection_name}, vector_size={vector_size}"
        )
    
    def ensure_collection_exists(self, recreate: bool = False, embedding = None) -> None:
        """
        Создание коллекции, если она не существует.
        Проверяет соответствие размера вектора существующей коллекции.
        Если коллекция не существует и передан объект эмбеддинга, выполняет пробный запрос
        для определения размера вектора перед созданием коллекции.
        
        Args:
            recreate: Пересоздать коллекцию, если она уже существует
            embedding: Опциональный объект эмбеддинга для определения размера вектора
        """
        try:
            # Проверяем существование коллекции
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )
            
            if collection_exists:
                if recreate:
                    logger.info(f"Удаление существующей коллекции: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                    collection_exists = False
                else:
                    # Проверяем соответствие размера вектора существующей коллекции
                    collection_info = self.client.get_collection(self.collection_name)
                    existing_vector_size = None
                    
                    if hasattr(collection_info, 'config'):
                        config = collection_info.config
                        if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                            vectors_config = config.params.vectors
                            if hasattr(vectors_config, 'size'):
                                existing_vector_size = vectors_config.size
                    
                    # Если размер вектора не был определен при инициализации, используем размер из коллекции
                    if self.vector_size is None and existing_vector_size is not None:
                        self.vector_size = existing_vector_size
                        logger.info(f"Размер вектора определен из существующей коллекции: {self.vector_size}")
                    elif existing_vector_size is not None and self.vector_size is not None and existing_vector_size != self.vector_size:
                        error_msg = (
                            f"Несоответствие размера вектора эмбеддинга! "
                            f"Коллекция '{self.collection_name}' имеет размер вектора {existing_vector_size}, "
                            f"а текущий эмбеддер создает векторы размером {self.vector_size}. "
                            f"Нельзя использовать эмбеддер с другим размером вектора для существующей коллекции. "
                            f"Либо используйте другой эмбеддер (с размером {existing_vector_size}), "
                            f"либо пересоздайте коллекцию с параметром recreate=True."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    logger.info(f"Коллекция {self.collection_name} уже существует (размер вектора: {existing_vector_size or 'неизвестен'})")
                    return
            
            if not collection_exists:
                # Если коллекция не существует, ОБЯЗАТЕЛЬНО нужен объект эмбеддинга для пробного запроса
                if embedding is None:
                    raise ValueError(
                        f"Для создания коллекции '{self.collection_name}' необходим объект эмбеддинга "
                        f"для определения размера вектора через пробный запрос к API. "
                        f"Передайте параметр embedding в ensure_collection_exists()."
                    )
                
                # Выполняем пробный запрос для определения размера вектора перед созданием коллекции
                try:
                    logger.info(f"Выполнение пробного запроса к сервису эмбеддингов для определения размера вектора...")
                    actual_vector_size = get_embedding_dimension(embedding)
                    logger.info(f"Размер вектора определен из сервиса эмбеддингов через пробный запрос: {actual_vector_size}")
                    
                    # Обновляем размер вектора в менеджере
                    self.vector_size = actual_vector_size
                except Exception as e:
                    logger.error(f"Не удалось определить размер вектора из сервиса эмбеддингов через пробный запрос: {e}")
                    raise ValueError(
                        f"Не удалось определить размер вектора эмбеддинга через пробный запрос к API. "
                        f"Ошибка: {e}"
                    )
                
                logger.info(f"Создание коллекции: {self.collection_name} с размером вектора: {self.vector_size}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Коллекция {self.collection_name} успешно создана с размером вектора: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}", exc_info=True)
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получение информации о коллекции.
        
        Returns:
            Словарь с информацией о коллекции
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Используем имя коллекции напрямую, так как оно уже известно
            # и проверяем наличие атрибутов перед доступом
            info_dict = {
                "name": self.collection_name,
                "vectors_count": getattr(collection_info, 'points_count', 0),
            }
            
            # Статус коллекции
            if hasattr(collection_info, 'status'):
                status = collection_info.status
                if hasattr(status, 'name'):
                    info_dict["status"] = status.name
                elif isinstance(status, str):
                    info_dict["status"] = status
                else:
                    info_dict["status"] = str(status)
            
            # Конфигурация векторов
            if hasattr(collection_info, 'config'):
                config = collection_info.config
                if hasattr(config, 'params') and hasattr(config.params, 'vectors'):
                    vectors_config = config.params.vectors
                    vector_info = {}
                    
                    if hasattr(vectors_config, 'size'):
                        vector_info["vector_size"] = vectors_config.size
                    
                    if hasattr(vectors_config, 'distance'):
                        distance = vectors_config.distance
                        if hasattr(distance, 'name'):
                            vector_info["distance"] = distance.name
                        elif isinstance(distance, str):
                            vector_info["distance"] = distance
                        else:
                            vector_info["distance"] = str(distance)
                    
                    if vector_info:
                        info_dict["config"] = vector_info
            
            return info_dict
            
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}", exc_info=True)
            raise
    
    def delete_collection(self) -> None:
        """
        Удаление коллекции.
        """
        try:
            logger.info(f"Удаление коллекции: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            logger.info(f"Коллекция {self.collection_name} успешно удалена")
        except Exception as e:
            logger.error(f"Ошибка при удалении коллекции: {e}", exc_info=True)
            raise
    
    def get_vector_store(self) -> QdrantVectorStore:
        """
        Получение объекта QdrantVectorStore для LlamaIndex.
        
        Returns:
            Объект QdrantVectorStore
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
    
    def search_by_metadata(
        self,
        field: str,
        value: Any,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Поиск точек по метаданным.
        
        Args:
            field: Поле для фильтрации
            value: Значение для поиска
            limit: Максимальное количество результатов
        
        Returns:
            Список найденных точек
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    ]
                ),
                limit=limit
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                }
                for point in results[0]
            ]
        except Exception as e:
            logger.error(f"Ошибка при поиске по метаданным: {e}", exc_info=True)
            return []
    
    def get_points_count(self) -> int:
        """
        Получение количества точек в коллекции.
        
        Returns:
            Количество точек
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Ошибка при получении количества точек: {e}", exc_info=True)
            return 0
    
    def check_connection(self, timeout: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Быстрая проверка доступности Qdrant сервера.
        
        Args:
            timeout: Таймаут проверки в секундах (по умолчанию 5 секунд)
            
        Returns:
            Кортеж (доступен, сообщение_об_ошибке)
        """
        try:
            # Используем HTTP запрос для быстрой проверки
            response = httpx.get(f"{self.url}/", timeout=timeout)
            if response.status_code == 200:
                return True, None
            else:
                return False, f"Qdrant вернул статус {response.status_code}"
        except httpx.ConnectError:
            return False, f"Не удалось подключиться к Qdrant на {self.url}. Убедитесь, что сервер запущен."
        except httpx.TimeoutException:
            return False, f"Таймаут подключения к Qdrant на {self.url}. Сервер не отвечает."
        except Exception as e:
            return False, f"Ошибка при проверке подключения к Qdrant: {str(e)}"

