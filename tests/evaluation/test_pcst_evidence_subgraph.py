"""Tests for rooted PCST evidence-subgraph construction."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch
from unittest.mock import patch

from pipeline.evaluation.models import CandidateNodeScore, EvaluationSample
from pipeline.evaluation.exceptions import ShortestPathExtractionException
from pipeline.evaluation.services.pcst_evidence_subgraph import (
    PcstEvidenceSubgraphService,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance


def _instance(*, reverse: bool = True) -> WebQSPProcessedInstance:
    nodes = ["seed", "connector", "answer_a", "answer_b"]
    edges = [(0, 1), (1, 2), (1, 3), (0, 1)]
    relations = ["r.seed", "r.answer_a", "r.answer_b", "r.parallel"]
    if reverse:
        edges.append((1, 0))
        relations.append("reverse__r.seed")
    return WebQSPProcessedInstance(
        question="question",
        q_entity=["seed"],
        a_entity=["answer_a"],
        nodes=nodes,
        node2id={node: index for index, node in enumerate(nodes)},
        edge_index=torch.tensor(edges, dtype=torch.long).T.contiguous(),
        edge_relations=relations,
        node_labels=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )


def _sample(seeds: list[str] | None = None) -> EvaluationSample:
    return EvaluationSample(
        question="question",
        q_entities=seeds or ["seed"],
        a_entities=["answer_a"],
        graph_triples=[],
    )


def _candidates() -> list[CandidateNodeScore]:
    return [
        CandidateNodeScore(node_id="answer_a", local_node_id=2, score=0.9),
        CandidateNodeScore(node_id="answer_b", local_node_id=3, score=0.8),
    ]


def test_pcst_maps_selected_solver_edges_to_original_directed_triples() -> None:
    captured = {}

    def solver(edges, prizes, costs, root, clusters, pruning, verbosity):
        captured.update(
            edges=edges,
            prizes=prizes,
            costs=costs,
            root=root,
            clusters=clusters,
            pruning=pruning,
            verbosity=verbosity,
        )
        return np.array([0, 1, 2, 3]), np.array([0, 1, 3])

    result = PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
        instance=_instance(),
        sample=_sample(),
        candidates=_candidates(),
        edge_cost_strategy="constant",
        edge_cost_lambda=1.0,
    )

    assert captured["edges"].shape == (4, 2)
    assert captured["edges"].tolist() == [
        [1, 2],
        [1, 3],
        [0, 1],
        [0, 1],
    ]
    assert captured["root"] == 0
    assert captured["clusters"] == 1
    assert captured["pruning"] == "gw"
    assert captured["verbosity"] == 0
    assert captured["prizes"].tolist() == [0.0, 0.0, 2.0, 1.0]
    assert [triple.model_dump() for triple in result.reasoning_subgraph_triples] == [
        {"source": "connector", "relation": "r.answer_a", "target": "answer_a"},
        {"source": "connector", "relation": "r.answer_b", "target": "answer_b"},
        {"source": "seed", "relation": "r.seed", "target": "connector"},
    ]
    assert all("reverse__" not in triple.relation for triple in result.reasoning_subgraph_triples)
    assert result.construction.selected_candidate_ranks == [1, 2]
    assert result.construction.collected_prize == 3.0
    assert result.construction.total_edge_cost == 3.0


def test_semantic_cost_uses_lambda_times_cosine_distance_with_epsilon() -> None:
    records = [(0, "same", 1), (1, "opposite", 2), (2, "orthogonal", 3)]
    costs = PcstEvidenceSubgraphService._edge_costs(
        records=records,
        strategy="semantic",
        edge_cost_lambda=2.0,
        question_embedding=np.array([1.0, 0.0]),
        relation_embeddings={
            "same": np.array([1.0, 0.0]),
            "opposite": np.array([-1.0, 0.0]),
            "orthogonal": np.array([0.0, 1.0]),
        },
    )
    assert costs == pytest.approx([1e-6, 4.0, 2.0])


def test_multiple_seeds_use_virtual_root_and_mandatory_seed_prizes() -> None:
    instance = _instance()
    instance.q_entity = ["seed", "connector"]
    captured = {}

    def solver(edges, prizes, costs, root, *_):
        captured.update(edges=edges, prizes=prizes, costs=costs, root=root)
        return np.arange(len(prizes)), np.arange(len(edges))

    result = PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
        instance=instance,
        sample=_sample(["seed", "connector"]),
        candidates=_candidates(),
        edge_cost_strategy="constant",
        edge_cost_lambda=1.0,
    )

    assert captured["root"] == len(instance.nodes)
    assert captured["edges"][-2:].tolist() == [[4, 0], [4, 1]]
    assert captured["costs"][-2:].tolist() == [0.0, 0.0]
    assert captured["prizes"][0] > 3.0
    assert captured["prizes"][1] > 3.0
    assert all(triple.source != "4" for triple in result.reasoning_subgraph_triples)
    assert result.construction.valid_seed_count == 2


def test_missing_seed_returns_empty_result_without_calling_solver() -> None:
    def solver(*_):
        raise AssertionError("solver must not run")

    result = PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
        instance=_instance(),
        sample=_sample(["missing"]),
        candidates=_candidates(),
        edge_cost_strategy="constant",
        edge_cost_lambda=1.0,
    )
    assert result.reasoning_subgraph_triples == []
    assert result.construction.empty_result_reason == "no_valid_seeds"
    assert result.missing_paths == 2


def test_real_solver_can_drop_candidate_when_path_cost_exceeds_prize() -> None:
    result = PcstEvidenceSubgraphService().extract_from_processed_graph(
        instance=_instance(reverse=False),
        sample=_sample(),
        candidates=[CandidateNodeScore(node_id="answer_a", local_node_id=2, score=0.9)],
        edge_cost_strategy="constant",
        edge_cost_lambda=2.0,
    )
    assert result.reasoning_subgraph_triples == []
    assert result.construction.selected_candidate_count == 0
    assert result.construction.empty_result_reason == "root_only_solution"


def test_rooted_solution_rejects_missing_root() -> None:
    def solver(*_):
        return np.array([2]), np.array([])

    with pytest.raises(ShortestPathExtractionException, match="required root"):
        PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
            instance=_instance(),
            sample=_sample(),
            candidates=[
                CandidateNodeScore(node_id="answer_a", local_node_id=2, score=0.9)
            ],
            edge_cost_strategy="constant",
            edge_cost_lambda=1.0,
        )


def test_rooted_solution_rejects_disconnected_vertices() -> None:
    def solver(*_):
        return np.array([0, 1, 2]), np.array([0])

    with pytest.raises(ShortestPathExtractionException, match="single connected tree"):
        PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
            instance=_instance(),
            sample=_sample(),
            candidates=_candidates(),
            edge_cost_strategy="constant",
            edge_cost_lambda=1.0,
        )


def test_debug_profile_saves_replayable_failure_snapshot(tmp_path) -> None:
    def solver(*_):
        return np.array([2]), np.array([], dtype=np.int64)

    with pytest.raises(ShortestPathExtractionException, match="Debug profile"):
        PcstEvidenceSubgraphService(solver=solver).extract_from_processed_graph(
            instance=_instance(),
            sample=_sample(),
            candidates=_candidates(),
            edge_cost_strategy="constant",
            edge_cost_lambda=1.0,
            debug_directory=tmp_path,
            instance_index=17,
        )

    snapshot = json.loads(
        (tmp_path / "instance_17_failure.json").read_text(encoding="utf-8")
    )
    assert snapshot["root"]["node"] == "seed"
    assert snapshot["root"]["reported_selected"] is False
    assert snapshot["instance"]["structural_edge_count"] == 4
    assert "collapsed_or_removed_edge_count" not in snapshot["instance"]
    assert snapshot["solver_output"]["selected_vertices"] == [2]
    assert snapshot["configuration"]["rooted_solver"] is True


def test_native_solver_rejects_numpy_two_before_importing_pcst_fast() -> None:
    service = PcstEvidenceSubgraphService()
    with patch.object(np, "__version__", "2.4.6"):
        with pytest.raises(ShortestPathExtractionException, match="NumPy >=1.26,<2"):
            service._resolve_solver()
