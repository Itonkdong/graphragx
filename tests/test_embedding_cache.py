"""Tests for Qdrant-backed embedding cache behavior."""

from types import SimpleNamespace
import unittest

from pipeline.preparation.services.embedding_cache import (
    TextEmbeddingCache,
    WebQSPEmbeddingCacheService,
)


class FakeEmbeddingService:
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    def embed_texts(self, texts: list[str], model_id: str) -> dict[str, list[float]]:
        self.calls.append((texts, model_id))
        return {
            text: [float(index), 1.0]
            for index, text in enumerate(texts)
        }


class FakeQdrantClient:
    def __init__(self):
        self.points: dict[str, dict] = {}
        self.upserts: list[list[dict]] = []
        self.created_collections: list[str] = []
        self.collection_exists_calls = 0

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls += 1
        return True

    def create_collection(self, collection_name: str, vectors_config):
        self.created_collections.append(collection_name)

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_vectors: bool,
        with_payload: bool,
    ):
        records = []
        for point_id in ids:
            point = self.points.get(str(point_id))
            if point is None:
                continue
            records.append(
                SimpleNamespace(
                    id=str(point_id),
                    vector=point["vector"] if with_vectors else None,
                    payload=point["payload"] if with_payload else None,
                )
            )
        return records

    def upsert(self, collection_name: str, points: list, wait: bool):
        normalized_points = []
        for point in points:
            if isinstance(point, dict):
                normalized = point
            else:
                normalized = {
                    "id": str(point.id),
                    "vector": point.vector,
                    "payload": point.payload,
                }
            normalized["id"] = str(normalized["id"])
            self.points[normalized["id"]] = normalized
            normalized_points.append(normalized)
        self.upserts.append(normalized_points)


class EmbeddingCacheTests(unittest.TestCase):
    def _service_and_cache(self):
        embedding_service = FakeEmbeddingService()
        qdrant_client = FakeQdrantClient()
        service = WebQSPEmbeddingCacheService(
            embedding_service=embedding_service,
            qdrant_client=qdrant_client,
            batch_size=2,
        )
        cache = TextEmbeddingCache(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="nodes",
            vocabulary={"existing": 0},
            collection_name="graphragx_embeddings_text_embedding_3_small",
            vector_size=1536,
        )
        return service, cache, embedding_service, qdrant_client

    def test_point_id_is_deterministic_and_cache_kind_scoped(self) -> None:
        first_id = WebQSPEmbeddingCacheService.point_id_for(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="nodes",
            text="Paris",
        )
        second_id = WebQSPEmbeddingCacheService.point_id_for(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="nodes",
            text="Paris",
        )
        relation_id = WebQSPEmbeddingCacheService.point_id_for(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="relations",
            text="Paris",
        )

        self.assertEqual(first_id, second_id)
        self.assertNotEqual(first_id, relation_id)

    def test_existing_points_are_not_embedded_again(self) -> None:
        service, cache, embedding_service, qdrant_client = self._service_and_cache()
        existing_id = service.point_id(cache=cache, text="existing")
        qdrant_client.points[existing_id] = {
            "id": existing_id,
            "vector": [9.0, 9.0],
            "payload": {"text": "existing"},
        }

        service.ensure_embeddings(cache=cache, texts=["existing"])

        self.assertEqual(embedding_service.calls, [])
        self.assertEqual(qdrant_client.upserts, [])

    def test_missing_points_are_embedded_and_upserted_with_payload(self) -> None:
        service, cache, embedding_service, qdrant_client = self._service_and_cache()

        service.ensure_embeddings(cache=cache, texts=["existing", "new"])

        self.assertEqual(embedding_service.calls, [(["existing", "new"], cache.model_id)])
        self.assertEqual(len(qdrant_client.upserts), 1)
        payloads = [point["payload"] for point in qdrant_client.upserts[0]]
        self.assertEqual(payloads[0]["text"], "existing")
        self.assertEqual(payloads[1]["text"], "new")
        self.assertEqual(payloads[1]["cache_kind"], "nodes")

    def test_relation_preprocessing_stores_original_and_embedding_input(self) -> None:
        embedding_service = FakeEmbeddingService()
        qdrant_client = FakeQdrantClient()
        service = WebQSPEmbeddingCacheService(
            embedding_service=embedding_service,
            qdrant_client=qdrant_client,
        )
        cache = TextEmbeddingCache(
            dataset_id="WebQSP",
            model_id="text-embedding-3-small",
            cache_kind="relations",
            vocabulary={},
            collection_name="graphragx_embeddings_text_embedding_3_small",
            vector_size=1536,
        )

        service.ensure_embeddings(
            cache=cache,
            texts=["people.person/place_of_birth"],
            preprocess=True,
        )

        self.assertEqual(embedding_service.calls[0][0], ["people person place of birth"])
        payload = qdrant_client.upserts[0][0]["payload"]
        self.assertEqual(payload["text"], "people.person/place_of_birth")
        self.assertEqual(payload["embedding_input"], "people person place of birth")

    def test_embeddings_for_texts_returns_vectors_in_requested_order(self) -> None:
        service, cache, _, qdrant_client = self._service_and_cache()
        for text, vector in [("a", [1.0, 0.0]), ("b", [2.0, 0.0])]:
            point_id = service.point_id(cache=cache, text=text)
            qdrant_client.points[point_id] = {
                "id": point_id,
                "vector": vector,
                "payload": {"text": text},
            }

        vectors = service.embeddings_for_texts(cache=cache, texts=["b", "a"])

        self.assertEqual(vectors, [[2.0, 0.0], [1.0, 0.0]])

    def test_cache_identity_can_be_loaded_without_contacting_qdrant(self) -> None:
        service, _, _, qdrant_client = self._service_and_cache()

        cache = service.load_node_cache(
            cache_root=None,
            model_id="text-embedding-3-small",
            vocabulary={"Paris": 0},
            ensure_collection=False,
        )

        self.assertEqual(cache.cache_kind, "nodes")
        self.assertEqual(qdrant_client.collection_exists_calls, 0)


if __name__ == "__main__":
    unittest.main()
