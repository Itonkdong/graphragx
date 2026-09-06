"""Persistent metadata for incremental GNN embedding tensor shards."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GnnEmbeddingTensorShardReference(BaseModel):
    """Location of one embedding vector in a persisted tensor shard."""

    shard_filename: str = Field(...)
    row_index: int = Field(..., ge=0)


class GnnEmbeddingTensorCacheManifest(BaseModel):
    """Index connecting deterministic embedding IDs to local tensor rows."""

    schema_version: int = Field(..., ge=1)
    dataset_id: str = Field(...)
    model_id: str = Field(...)
    cache_kind: str = Field(...)
    vector_size: int = Field(..., gt=0)
    dtype_name: str = Field(...)
    next_shard_index: int = Field(default=0, ge=0)
    entries: dict[str, GnnEmbeddingTensorShardReference] = Field(
        default_factory=dict
    )
