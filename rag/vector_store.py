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
    
    Args:
        embedding: Объект эмбеддинга (GigaEmbedding, OllamaEmbedding или BaseEmbedding)
    
    Returns:
        Размер вектора эмбеддинга
    
    Raises:
        ValueError: Если не удалось определить размер вектора
    """
    # Пробуем получить из атрибута embedding_dim (используется в GigaEmbedding и OllamaEmbedding)
    if hasattr(embedding, 'embedding_dim'):
        dim = getattr(embedding, 'embedding_dim')
        if dim is not None:
            return int(dim)
    
    # Пробуем получить из атрибута embed_dim (из BaseEmbedding)
    if hasattr(embedding, 'embed_dim'):
        dim = getattr(embedding, 'embed_dim')
        if dim is not None:
            return int(dim)
    
    # Если не удалось получить, пробуем получить реальный размер из первого эмбеддинга
    try:
        # Пробуем получить эмбеддинг для тестового текста
        test_embedding = embedding.get_query_embedding("test")
        if test_embedding and isinstance(test_embedding, list):
            return len(test_embedding)
    except Exception:
        pass
    
    # Если ничего не помогло, выбрасываем ошибку
    raise ValueError(
        f"Не удалось определить размер вектора эмбеддинга из объекта {type(embedding).__name__}. "
        "Убедитесь, что объект эмбеддинга имеет атрибут embedding_dim или embed_dim."
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
        vector_size: int = 2560,
        timeout: int = 30
    ):
        """
        Инициализация менеджера Qdrant.
        
        Args:
            url: URL Qdrant сервера
            api_key: API ключ (если требуется)
            collection_name: Имя коллекции
            vector_size: Размер вектора эмбеддинга
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
    
    def ensure_collection_exists(self, recreate: bool = False) -> None:
        """
        Создание коллекции, если она не существует.
        Проверяет соответствие размера вектора существующей коллекции.
        
        Args:
            recreate: Пересоздать коллекцию, если она уже существует
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
                    
                    if existing_vector_size is not None and existing_vector_size != self.vector_size:
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

