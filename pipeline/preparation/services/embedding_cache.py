"""Qdrant-backed embedding cache services for WebQSP training artifacts."""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from helpers.constants import DEFAULT_EMBEDDING_BATCH_SIZE
from helpers.env_variables import QDRANT_API_KEY, QDRANT_COLLECTION_PREFIX, QDRANT_URL
from helpers.logging_config import get_logger
from pipeline.preparation.exceptions import QdrantEmbeddingStoreException
from pipeline.preparation.helpers.configuration_definitions import OPENAI_EMBEDDING_MODELS
from pipeline.preparation.helpers.dataset_definitions import WEBQSP_DATASET_ID
from pipeline.services import AbstractService
from pipeline.preparation.services.openai_text_embedding import (
    LangChainOpenAiTextEmbeddingService,
)

logger = get_logger(__name__)


class TextEmbeddingCache(BaseModel):
    """Lightweight handle for one text category stored in Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_id: str = Field(..., description="Dataset identifier.")
    model_id: str = Field(..., description="Embedding model id used for this cache.")
    cache_kind: str = Field(..., description="Text category stored in this cache.")
    vocabulary: dict[str, int] = Field(default_factory=dict)
    collection_name: str = Field(..., description="Qdrant collection name.")
    vector_size: int = Field(..., description="Embedding vector dimensionality.")


class WebQSPEmbeddingCacheService(AbstractService):
    """Load, populate, and read WebQSP embeddings through Qdrant."""

    default_batch_size = DEFAULT_EMBEDDING_BATCH_SIZE
    point_namespace = uuid.UUID("8edee705-0dc6-5db5-b1d9-82a4ea62197d")

    def __init__(
        self,
        embedding_service: LangChainOpenAiTextEmbeddingService | None = None,
        batch_size: int = default_batch_size,
        qdrant_client: Any | None = None,
        qdrant_url: str = QDRANT_URL,
        qdrant_api_key: str | None = QDRANT_API_KEY,
        collection_prefix: str = QDRANT_COLLECTION_PREFIX,
    ):
        self.embedding_service = embedding_service or LangChainOpenAiTextEmbeddingService()
        if batch_size <= 0:
            raise ValueError("Embedding cache batch size must be greater than zero.")

        self.batch_size = batch_size
        self._qdrant_client = qdrant_client
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection_prefix = collection_prefix

    def load_node_cache(
        self,
        cache_root,
        model_id: str,
        vocabulary: dict[str, int],
        dataset_id: str = WEBQSP_DATASET_ID,
        ensure_collection: bool = True,
    ) -> TextEmbeddingCache:
        """Return the entity/node embedding cache handle for a model."""
        return self._load_cache(
            dataset_id=dataset_id,
            model_id=model_id,
            cache_kind="nodes",
            vocabulary=vocabulary,
            ensure_collection=ensure_collection,
        )

    def load_relation_cache(
        self,
        cache_root,
        model_id: str,
        vocabulary: dict[str, int],
        dataset_id: str = WEBQSP_DATASET_ID,
        ensure_collection: bool = True,
    ) -> TextEmbeddingCache:
        """Return the relation embedding cache handle for a model."""
        return self._load_cache(
            dataset_id=dataset_id,
            model_id=model_id,
            cache_kind="relations",
            vocabulary=vocabulary,
            ensure_collection=ensure_collection,
        )

    def load_question_cache(
        self,
        cache_root,
        model_id: str,
        vocabulary: dict[str, int],
        dataset_id: str = WEBQSP_DATASET_ID,
        ensure_collection: bool = True,
    ) -> TextEmbeddingCache:
        """Return the question embedding cache handle for a model."""
        return self._load_cache(
            dataset_id=dataset_id,
            model_id=model_id,
            cache_kind="questions",
            vocabulary=vocabulary,
            ensure_collection=ensure_collection,
        )

    def ensure_embeddings(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
        preprocess: bool = False,
    ) -> None:
        """Populate missing embeddings for the requested texts."""
        unique_texts = list(dict.fromkeys(texts))
        existing_point_ids = self._existing_point_ids(cache, unique_texts)
        missing_texts = [
            text
            for text in unique_texts
            if self.point_id(cache=cache, text=text) not in existing_point_ids
        ]
        if not missing_texts:
            logger.info(
                f"Qdrant embedding cache hit for {cache.cache_kind}/{cache.model_id}: "
                f"requested={len(unique_texts)} existing={len(existing_point_ids)} missing=0"
            )
            return

        total_batches = (len(missing_texts) + self.batch_size - 1) // self.batch_size
        logger.info(
            f"Qdrant embedding cache fill for {cache.cache_kind}/{cache.model_id}: "
            f"requested={len(unique_texts)} existing={len(existing_point_ids)} "
            f"missing={len(missing_texts)} batch_size={self.batch_size} "
            f"collection={cache.collection_name} preview={self._preview_texts(missing_texts)}"
        )
        for batch_index, start_index in enumerate(
            range(0, len(missing_texts), self.batch_size),
            start=1,
        ):
            batch_texts = missing_texts[start_index : start_index + self.batch_size]
            embedding_inputs = [
                self.preprocess_relation_text(text) if preprocess else text
                for text in batch_texts
            ]
            logger.info(
                f"Embedding {cache.cache_kind}/{cache.model_id} batch "
                f"{batch_index}/{total_batches}: original_text_count={len(batch_texts)} "
                f"endpoint_text_count={len(embedding_inputs)} "
                f"preview={self._preview_texts(batch_texts)}"
            )
            embedded_inputs = self.embedding_service.embed_texts(
                texts=embedding_inputs,
                model_id=cache.model_id,
            )
            self.upsert_embeddings(
                cache=cache,
                original_texts=batch_texts,
                embedding_inputs=embedding_inputs,
                embedded_inputs=embedded_inputs,
            )
            logger.info(
                f"Upserted Qdrant embedding batch for {cache.cache_kind}/{cache.model_id}: "
                f"batch={batch_index}/{total_batches} upserted={len(batch_texts)} "
                f"collection={cache.collection_name}"
            )

    def ensure_cache_available(self, cache: TextEmbeddingCache) -> None:
        """Create the backing Qdrant collection only when remote access is needed."""
        self._ensure_collection(
            collection_name=cache.collection_name,
            vector_size=cache.vector_size,
        )

    def upsert_embeddings(
        self,
        cache: TextEmbeddingCache,
        original_texts: list[str],
        embedding_inputs: list[str],
        embedded_inputs: dict[str, list[float]],
    ) -> None:
        """Persist precomputed embeddings into Qdrant."""
        records = [
            (
                original_text,
                embedding_input,
                embedded_inputs[embedding_input],
            )
            for original_text, embedding_input in zip(
                original_texts,
                embedding_inputs,
                strict=True,
            )
        ]
        self.upsert_embedding_records(cache=cache, records=records)

    def upsert_embedding_records(
        self,
        cache: TextEmbeddingCache,
        records: list[tuple[str, str, list[float]]],
    ) -> None:
        """Persist explicit original text, embedding input, and vector records."""
        try:
            from qdrant_client.models import PointStruct
        except ModuleNotFoundError:
            PointStruct = None

        points = []
        for original_text, embedding_input, vector in records:
            text_id = self._get_or_add_text_id(cache, original_text)
            point_payload = {
                "dataset_id": cache.dataset_id,
                "model_id": cache.model_id,
                "cache_kind": cache.cache_kind,
                "text": original_text,
                "embedding_input": embedding_input,
                "text_id": text_id,
            }
            point_id = self.point_id(cache=cache, text=original_text)
            if PointStruct is None:
                points.append(
                    {
                        "id": point_id,
                        "vector": vector,
                        "payload": point_payload,
                    }
                )
            else:
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=point_payload,
                    )
                )

        try:
            self.client.upsert(
                collection_name=cache.collection_name,
                points=points,
                wait=True,
            )
        except Exception as error:
            raise QdrantEmbeddingStoreException(
                f"Could not upsert embeddings into Qdrant collection "
                f"{cache.collection_name}. Is Qdrant running at {self.qdrant_url}? "
                f"Original error: {error}"
            ) from error

    def embeddings_for_texts(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> list[list[float]]:
        """Return cached embeddings for texts in the same order as requested."""
        if not texts:
            return []

        point_ids = [self.point_id(cache=cache, text=text) for text in texts]
        vectors_by_id: dict[str, list[float]] = {}
        for batch_ids in self._chunk(point_ids, self.batch_size):
            try:
                records = self.client.retrieve(
                    collection_name=cache.collection_name,
                    ids=batch_ids,
                    with_vectors=True,
                    with_payload=False,
                )
            except Exception as error:
                raise QdrantEmbeddingStoreException(
                    f"Could not retrieve embeddings from Qdrant collection "
                    f"{cache.collection_name}. Is Qdrant running at {self.qdrant_url}? "
                    f"Original error: {error}"
                ) from error

            for record in records:
                vector = getattr(record, "vector", None)
                if vector is not None:
                    vectors_by_id[str(record.id)] = list(vector)

        missing_texts = [
            text
            for text, point_id in zip(texts, point_ids, strict=True)
            if point_id not in vectors_by_id
        ]
        if missing_texts:
            raise QdrantEmbeddingStoreException(
                f"Missing {len(missing_texts)} embeddings in Qdrant collection "
                f"{cache.collection_name}. First missing values: "
                f"{self._preview_texts(missing_texts)}"
            )

        return [vectors_by_id[point_id] for point_id in point_ids]

    def embedding_for_text(
        self,
        cache: TextEmbeddingCache,
        text: str,
    ) -> list[float]:
        """Return one cached embedding for compatibility with existing callers."""
        return self.embeddings_for_texts(cache=cache, texts=[text])[0]

    @staticmethod
    def preprocess_relation_text(relation: str) -> str:
        """Normalize relation identifiers before embedding."""
        normalized = re.sub(r"[._/]+", " ", relation)
        return " ".join(normalized.split())

    def point_id(self, cache: TextEmbeddingCache, text: str) -> str:
        """Return the deterministic Qdrant point id for a cached text."""
        return self.point_id_for(
            dataset_id=cache.dataset_id,
            model_id=cache.model_id,
            cache_kind=cache.cache_kind,
            text=text,
        )

    @classmethod
    def point_id_for(
        cls,
        dataset_id: str,
        model_id: str,
        cache_kind: str,
        text: str,
    ) -> str:
        raw_key = f"{dataset_id}|{model_id}|{cache_kind}|{text}"
        return str(uuid.uuid5(cls.point_namespace, raw_key))

    def collection_name_for_model(self, model_id: str) -> str:
        safe_model_id = re.sub(r"[^a-zA-Z0-9_]+", "_", model_id).strip("_")
        return f"{self.collection_prefix}_{safe_model_id}"

    def _load_cache(
        self,
        dataset_id: str,
        model_id: str,
        cache_kind: str,
        vocabulary: dict[str, int],
        ensure_collection: bool,
    ) -> TextEmbeddingCache:
        vector_size = self._vector_size_for_model(model_id)
        collection_name = self.collection_name_for_model(model_id)
        if ensure_collection:
            self._ensure_collection(
                collection_name=collection_name,
                vector_size=vector_size,
            )
        return TextEmbeddingCache(
            dataset_id=dataset_id,
            model_id=model_id,
            cache_kind=cache_kind,
            vocabulary=dict(vocabulary),
            collection_name=collection_name,
            vector_size=vector_size,
        )

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        try:
            if self.client.collection_exists(collection_name):
                return
            from qdrant_client.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except Exception as error:
            raise QdrantEmbeddingStoreException(
                f"Could not prepare Qdrant collection {collection_name}. "
                f"Is Qdrant running at {self.qdrant_url}? Original error: {error}"
            ) from error

    def _existing_point_ids(
        self,
        cache: TextEmbeddingCache,
        texts: list[str],
    ) -> set[str]:
        existing_ids: set[str] = set()
        point_ids = [self.point_id(cache=cache, text=text) for text in texts]
        for batch_ids in self._chunk(point_ids, self.batch_size):
            try:
                records = self.client.retrieve(
                    collection_name=cache.collection_name,
                    ids=batch_ids,
                    with_vectors=False,
                    with_payload=False,
                )
            except Exception as error:
                raise QdrantEmbeddingStoreException(
                    f"Could not check existing embeddings in Qdrant collection "
                    f"{cache.collection_name}. Is Qdrant running at {self.qdrant_url}? "
                    f"Original error: {error}"
                ) from error
            existing_ids.update(str(record.id) for record in records)

        return existing_ids

    @property
    def client(self) -> Any:
        """Return a Qdrant client, creating it lazily for clearer pipeline errors."""
        if self._qdrant_client is not None:
            return self._qdrant_client

        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError as error:
            raise QdrantEmbeddingStoreException(
                "qdrant-client is required for the embedding vector store. "
                "Install dependencies from requirements.txt."
            ) from error

        self._qdrant_client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        return self._qdrant_client

    @staticmethod
    def _vector_size_for_model(model_id: str) -> int:
        model_definition = OPENAI_EMBEDDING_MODELS.get(model_id)
        if model_definition is None:
            raise QdrantEmbeddingStoreException(
                f"Unknown embedding model {model_id}; cannot determine Qdrant vector size."
            )
        return model_definition.dimensions

    @staticmethod
    def _get_or_add_text_id(cache: TextEmbeddingCache, text: str) -> int:
        if text not in cache.vocabulary:
            cache.vocabulary[text] = len(cache.vocabulary)

        return cache.vocabulary[text]

    @staticmethod
    def _chunk(items: list[str], batch_size: int) -> list[list[str]]:
        return [
            items[start_index : start_index + batch_size]
            for start_index in range(0, len(items), batch_size)
        ]

    @staticmethod
    def _preview_texts(texts: list[str]) -> list[str]:
        return [
            text[:117] + "..." if len(text) > 120 else text
            for text in texts[:3]
        ]
