"""Rooted prize-collecting Steiner-tree evidence construction."""

from __future__ import annotations

from collections import deque
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, NoReturn

import numpy as np

from helpers.logging_config import get_logger
from pipeline.evaluation.exceptions import ShortestPathExtractionException
from pipeline.evaluation.models import (
    CandidateNodeScore,
    EvidenceSubgraphConstruction,
    EvaluationSample,
    ExtractedReasoningPaths,
    GraphTriple,
    ReasoningPath,
)
from pipeline.evaluation.services.shortest_path_extraction import (
    ShortestPathExtractionService,
)
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance
from pipeline.services import AbstractService

logger = get_logger(__name__)


class PcstEvidenceSubgraphService(AbstractService):
    """Build compact evidence with rooted PCST over original WebQSP facts."""

    semantic_cost_epsilon = 1e-6

    def __init__(self, solver: Any | None = None) -> None:
        self._solver = solver

    def extract_from_processed_graph(
        self,
        *,
        instance: WebQSPProcessedInstance,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
        edge_cost_strategy: str,
        edge_cost_lambda: float,
        semantic_embedding_model: str | None = None,
        question_embedding: np.ndarray | None = None,
        relation_embeddings: dict[str, np.ndarray] | None = None,
        debug_directory: Path | None = None,
        instance_index: int | None = None,
    ) -> ExtractedReasoningPaths:
        """Solve rooted PCST and return the common evidence result model."""
        started_at = time.perf_counter()
        self._validate_cost(edge_cost_strategy, edge_cost_lambda)
        records = self._original_edge_records(instance)
        valid_seeds = sorted(
            {
                instance.node2id[seed]
                for seed in sample.q_entities
                if seed in instance.node2id
            }
        )
        missing_seed_count = len(set(sample.q_entities)) - len(valid_seeds)
        valid_candidates = self._valid_reachable_candidates(
            instance=instance,
            records=records,
            seeds=valid_seeds,
            candidates=candidates,
        )

        empty_reason = None
        if not valid_seeds:
            empty_reason = "no_valid_seeds"
        elif not candidates:
            empty_reason = "no_candidates"
        elif not valid_candidates:
            empty_reason = "no_reachable_candidates"
        elif not records:
            empty_reason = "no_original_edges"
        if empty_reason is not None:
            logger.warning(
                "PCST produced empty evidence before solving: "
                f"reason={empty_reason} question={sample.question!r}"
            )
            return self._empty_result(
                sample=sample,
                candidates=candidates,
                valid_candidate_count=len(valid_candidates),
                valid_seed_count=len(valid_seeds),
                missing_seed_count=missing_seed_count,
                strategy=edge_cost_strategy,
                edge_cost_lambda=edge_cost_lambda,
                semantic_embedding_model=semantic_embedding_model,
                started_at=started_at,
                reason=empty_reason,
            )

        edge_costs = self._edge_costs(
            records=records,
            strategy=edge_cost_strategy,
            edge_cost_lambda=edge_cost_lambda,
            question_embedding=question_embedding,
            relation_embeddings=relation_embeddings,
        )
        # Keep every relation-distinct original triple in the structural
        # projection. Parallel relations are meaningful alternatives and map
        # directly back to the evidence triple selected by the solver.
        solver_edges = [(source, target) for source, _, target in records]
        solver_edge_costs = list(edge_costs)
        solver_record_indices = list(range(len(records)))
        prizes = np.zeros(len(instance.nodes), dtype=np.float64)
        candidate_prizes: dict[int, float] = {}
        candidate_ranks: dict[int, int] = {}
        valid_count = len(valid_candidates)
        for valid_rank, (original_rank, node_id, _) in enumerate(
            valid_candidates,
            start=1,
        ):
            prize = float(valid_count - valid_rank + 1)
            prizes[node_id] = max(prizes[node_id], prize)
            candidate_prizes[node_id] = max(candidate_prizes.get(node_id, 0.0), prize)
            candidate_ranks.setdefault(node_id, original_rank)

        synthetic_edge_start = len(solver_edges)
        if len(valid_seeds) == 1:
            root = valid_seeds[0]
        else:
            root = len(prizes)
            prizes = np.concatenate([prizes, np.zeros(1, dtype=np.float64)])
            mandatory_prize = (
                float(sum(candidate_prizes.values()))
                + float(sum(solver_edge_costs))
                + 1.0
            )
            for seed in valid_seeds:
                prizes[seed] += mandatory_prize
                solver_edges.append((root, seed))
                solver_edge_costs.append(0.0)

        try:
            selected_vertices, selected_edges = self._resolve_solver()(
                np.asarray(solver_edges, dtype=np.int64),
                prizes,
                np.asarray(solver_edge_costs, dtype=np.float64),
                root,
                1,
                "gw",
                0,
            )
        except Exception as error:
            raise ShortestPathExtractionException(
                f"PCST solver failed for question {sample.question!r}: {error}"
            ) from error

        reported_vertex_ids = {int(value) for value in selected_vertices}
        selected_edge_ids = [int(value) for value in selected_edges]
        if any(index < 0 or index >= len(prizes) for index in reported_vertex_ids):
            self._raise_solver_output_error(
                "PCST solver returned a vertex index outside the input graph.",
                debug_directory=debug_directory,
                instance_index=instance_index,
                instance=instance,
                sample=sample,
                candidates=candidates,
                valid_candidates=valid_candidates,
                edge_cost_strategy=edge_cost_strategy,
                edge_cost_lambda=edge_cost_lambda,
                records=records,
                record_edge_costs=edge_costs,
                solver_edges=solver_edges,
                solver_edge_costs=solver_edge_costs,
                solver_record_indices=solver_record_indices,
                synthetic_edge_start=synthetic_edge_start,
                prizes=prizes,
                root=root,
                selected_vertices=selected_vertices,
                selected_edges=selected_edges,
            )
        if any(index < 0 or index >= len(solver_edges) for index in selected_edge_ids):
            self._raise_solver_output_error(
                "PCST solver returned an edge index outside the input graph.",
                debug_directory=debug_directory,
                instance_index=instance_index,
                instance=instance,
                sample=sample,
                candidates=candidates,
                valid_candidates=valid_candidates,
                edge_cost_strategy=edge_cost_strategy,
                edge_cost_lambda=edge_cost_lambda,
                records=records,
                record_edge_costs=edge_costs,
                solver_edges=solver_edges,
                solver_edge_costs=solver_edge_costs,
                solver_record_indices=solver_record_indices,
                synthetic_edge_start=synthetic_edge_start,
                prizes=prizes,
                root=root,
                selected_vertices=selected_vertices,
                selected_edges=selected_edges,
            )

        if root not in reported_vertex_ids:
            self._raise_solver_output_error(
                "PCST rooted solution does not contain its required root.",
                debug_directory=debug_directory,
                instance_index=instance_index,
                instance=instance,
                sample=sample,
                candidates=candidates,
                valid_candidates=valid_candidates,
                edge_cost_strategy=edge_cost_strategy,
                edge_cost_lambda=edge_cost_lambda,
                records=records,
                record_edge_costs=edge_costs,
                solver_edges=solver_edges,
                solver_edge_costs=solver_edge_costs,
                solver_record_indices=solver_record_indices,
                synthetic_edge_start=synthetic_edge_start,
                prizes=prizes,
                root=root,
                selected_vertices=selected_vertices,
                selected_edges=selected_edges,
            )
        selected_vertex_ids = set(reported_vertex_ids)
        selected_adjacency: dict[int, set[int]] = {}
        for edge_index in selected_edge_ids:
            source, target = solver_edges[edge_index]
            if source not in selected_vertex_ids or target not in selected_vertex_ids:
                self._raise_solver_output_error(
                    "PCST solver selected an edge whose endpoint is absent from "
                    "the selected vertices.",
                    debug_directory=debug_directory,
                    instance_index=instance_index,
                    instance=instance,
                    sample=sample,
                    candidates=candidates,
                    valid_candidates=valid_candidates,
                    edge_cost_strategy=edge_cost_strategy,
                    edge_cost_lambda=edge_cost_lambda,
                    records=records,
                    record_edge_costs=edge_costs,
                    solver_edges=solver_edges,
                    solver_edge_costs=solver_edge_costs,
                    solver_record_indices=solver_record_indices,
                    synthetic_edge_start=synthetic_edge_start,
                    prizes=prizes,
                    root=root,
                    selected_vertices=selected_vertices,
                    selected_edges=selected_edges,
                )
            selected_adjacency.setdefault(source, set()).add(target)
            selected_adjacency.setdefault(target, set()).add(source)
        reachable_selected = {root}
        selected_queue = deque([root])
        while selected_queue:
            node = selected_queue.popleft()
            for neighbor in selected_adjacency.get(node, set()):
                if neighbor not in reachable_selected:
                    reachable_selected.add(neighbor)
                    selected_queue.append(neighbor)

        if not selected_vertex_ids.issubset(reachable_selected):
            self._raise_solver_output_error(
                "PCST rooted solution is not a single connected tree.",
                debug_directory=debug_directory,
                instance_index=instance_index,
                instance=instance,
                sample=sample,
                candidates=candidates,
                valid_candidates=valid_candidates,
                edge_cost_strategy=edge_cost_strategy,
                edge_cost_lambda=edge_cost_lambda,
                records=records,
                record_edge_costs=edge_costs,
                solver_edges=solver_edges,
                solver_edge_costs=solver_edge_costs,
                solver_record_indices=solver_record_indices,
                synthetic_edge_start=synthetic_edge_start,
                prizes=prizes,
                root=root,
                selected_vertices=selected_vertices,
                selected_edges=selected_edges,
            )

        selected_original_solver_edge_ids = sorted(
            index for index in selected_edge_ids if index < synthetic_edge_start
        )
        selected_record_ids = [
            solver_record_indices[index]
            for index in selected_original_solver_edge_ids
        ]
        selected_triples = [
            GraphTriple(
                source=instance.nodes[records[index][0]],
                relation=records[index][1],
                target=instance.nodes[records[index][2]],
            )
            for index in selected_record_ids
        ]
        selected_candidates = [
            (rank, node_id, candidate)
            for rank, node_id, candidate in valid_candidates
            if node_id in selected_vertex_ids
        ]
        selected_candidate_ids = {node_id for _, node_id, _ in selected_candidates}
        paths = [
            ReasoningPath(
                candidate_node=candidate.node_id,
                candidate_score=candidate.score,
                path_found=(
                    self._candidate_node_id(instance, candidate)
                    in selected_candidate_ids
                ),
            )
            for candidate in candidates
        ]
        total_edge_cost = float(
            sum(
                solver_edge_costs[index]
                for index in selected_original_solver_edge_ids
            )
        )
        collected_prize = float(
            sum(candidate_prizes[node_id] for _, node_id, _ in selected_candidates)
        )
        selected_nodes = {
            node
            for triple in selected_triples
            for node in (triple.source, triple.target)
        }
        construction = EvidenceSubgraphConstruction(
            strategy="pcst",
            edge_cost_strategy=edge_cost_strategy,
            edge_cost_lambda=edge_cost_lambda,
            semantic_embedding_model=semantic_embedding_model,
            input_candidate_count=len(candidates),
            valid_candidate_count=len(valid_candidates),
            selected_candidate_count=len(selected_candidates),
            selected_candidate_ranks=[rank for rank, _, _ in selected_candidates],
            collected_prize=collected_prize,
            selected_node_count=len(selected_nodes),
            selected_triple_count=len(selected_triples),
            total_edge_cost=total_edge_cost,
            objective=collected_prize - total_edge_cost,
            valid_seed_count=len(valid_seeds),
            missing_seed_count=missing_seed_count,
            construction_time_ms=(time.perf_counter() - started_at) * 1000,
            empty_result_reason=("root_only_solution" if not selected_triples else None),
        )
        return ExtractedReasoningPaths(
            sample=sample.model_copy(update={"graph_triples": selected_triples}),
            paths=paths,
            reasoning_subgraph_triples=selected_triples,
            reasoning_paths_text=ShortestPathExtractionService.verbalize_subgraph(
                selected_triples
            ),
            found_paths=len(selected_candidates),
            missing_paths=len(candidates) - len(selected_candidates),
            construction=construction,
        )

    def _raise_solver_output_error(
        self,
        message: str,
        *,
        debug_directory: Path | None,
        instance_index: int | None,
        instance: WebQSPProcessedInstance,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
        valid_candidates: list[tuple[int, int, CandidateNodeScore]],
        edge_cost_strategy: str,
        edge_cost_lambda: float,
        records: list[tuple[int, str, int]],
        record_edge_costs: list[float],
        solver_edges: list[tuple[int, int]],
        solver_edge_costs: list[float],
        solver_record_indices: list[int],
        synthetic_edge_start: int,
        prizes: np.ndarray,
        root: int,
        selected_vertices: Any,
        selected_edges: Any,
    ) -> NoReturn:
        diagnostic_path = None
        if debug_directory is not None:
            try:
                diagnostic_path = self._write_debug_snapshot(
                    debug_directory=debug_directory,
                    instance_index=instance_index,
                    error_message=message,
                    instance=instance,
                    sample=sample,
                    candidates=candidates,
                    valid_candidates=valid_candidates,
                    edge_cost_strategy=edge_cost_strategy,
                    edge_cost_lambda=edge_cost_lambda,
                    records=records,
                    record_edge_costs=record_edge_costs,
                    solver_edges=solver_edges,
                    solver_edge_costs=solver_edge_costs,
                    solver_record_indices=solver_record_indices,
                    synthetic_edge_start=synthetic_edge_start,
                    prizes=prizes,
                    root=root,
                    selected_vertices=selected_vertices,
                    selected_edges=selected_edges,
                )
            except Exception as diagnostic_error:
                logger.exception(
                    "Could not save PCST debug profile: "
                    f"error={diagnostic_error}"
                )
        suffix = (
            f" Debug profile: {diagnostic_path}." if diagnostic_path else ""
        )
        raise ShortestPathExtractionException(f"{message}{suffix}")

    @classmethod
    def _write_debug_snapshot(
        cls,
        *,
        debug_directory: Path,
        instance_index: int | None,
        error_message: str,
        instance: WebQSPProcessedInstance,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
        valid_candidates: list[tuple[int, int, CandidateNodeScore]],
        edge_cost_strategy: str,
        edge_cost_lambda: float,
        records: list[tuple[int, str, int]],
        record_edge_costs: list[float],
        solver_edges: list[tuple[int, int]],
        solver_edge_costs: list[float],
        solver_record_indices: list[int],
        synthetic_edge_start: int,
        prizes: np.ndarray,
        root: int,
        selected_vertices: Any,
        selected_edges: Any,
    ) -> Path:
        """Persist enough state to replay one malformed native solver result."""
        debug_directory.mkdir(parents=True, exist_ok=True)
        selected_vertex_ids = [int(value) for value in selected_vertices]
        selected_edge_ids = [int(value) for value in selected_edges]
        valid_selected_edges = [
            index for index in selected_edge_ids if 0 <= index < len(solver_edges)
        ]
        components = cls._solver_components(
            solver_edges=solver_edges,
            selected_edge_ids=valid_selected_edges,
            selected_vertex_ids=selected_vertex_ids,
        )

        def node_name(node_id: int) -> str:
            if 0 <= node_id < len(instance.nodes):
                return instance.nodes[node_id]
            if node_id == len(instance.nodes):
                return "__synthetic_root__"
            return "__invalid_node__"

        solver_edge_rows = []
        for edge_index, ((source, target), cost) in enumerate(
            zip(solver_edges, solver_edge_costs, strict=True)
        ):
            record_index = (
                solver_record_indices[edge_index]
                if edge_index < synthetic_edge_start
                else None
            )
            relation = records[record_index][1] if record_index is not None else None
            solver_edge_rows.append(
                {
                    "edge_index": edge_index,
                    "source_id": source,
                    "source": node_name(source),
                    "target_id": target,
                    "target": node_name(target),
                    "cost": float(cost),
                    "record_index": record_index,
                    "relation": relation,
                    "selected": edge_index in selected_edge_ids,
                }
            )

        try:
            pcst_fast_version = metadata.version("pcst-fast")
        except metadata.PackageNotFoundError:
            pcst_fast_version = "unknown"
        reverse_edge_count = sum(
            relation.startswith("reverse__")
            for relation in instance.edge_relations
        )
        self_loop_count = sum(source == target for source, _, target in records)
        payload = {
            "schema_version": 1,
            "error": error_message,
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
                "numpy": np.__version__,
                "pcst_fast": pcst_fast_version,
            },
            "instance": {
                "instance_index": instance_index,
                "question": sample.question,
                "question_entities": sample.q_entities,
                "gold_entities": sample.a_entities,
                "node_count": len(instance.nodes),
                "raw_edge_count": int(instance.edge_index.shape[1]),
                "reverse_edge_count": reverse_edge_count,
                "original_record_count": len(records),
                "self_loop_count": self_loop_count,
                "structural_edge_count": synthetic_edge_start,
            },
            "configuration": {
                "edge_cost_strategy": edge_cost_strategy,
                "edge_cost_lambda": edge_cost_lambda,
                "rooted_solver": True,
                "num_clusters": 1,
                "pruning": "gw",
                "verbosity": 0,
            },
            "root": {
                "node_id": root,
                "node": node_name(root),
                "reported_selected": root in selected_vertex_ids,
            },
            "nodes": [
                {
                    "node_id": node_id,
                    "node": node_name(node_id),
                    "prize": float(prize),
                }
                for node_id, prize in enumerate(prizes.tolist())
            ],
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "valid_candidates": [
                {
                    "original_rank": rank,
                    "node_id": node_id,
                    "node": node_name(node_id),
                    "candidate": candidate.model_dump(mode="json"),
                }
                for rank, node_id, candidate in valid_candidates
            ],
            "original_records": [
                {
                    "record_index": index,
                    "source_id": source,
                    "source": node_name(source),
                    "relation": relation,
                    "target_id": target,
                    "target": node_name(target),
                    "cost": float(record_edge_costs[index]),
                }
                for index, (source, relation, target) in enumerate(records)
            ],
            "solver_edges": solver_edge_rows,
            "solver_output": {
                "selected_vertices": selected_vertex_ids,
                "selected_edges": selected_edge_ids,
                "components": components,
            },
        }
        filename_index = "unknown" if instance_index is None else str(instance_index)
        destination = debug_directory / f"instance_{filename_index}_failure.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
        logger.error(f"Saved PCST debug profile: path={destination}")
        return destination

    @staticmethod
    def _solver_components(
        *,
        solver_edges: list[tuple[int, int]],
        selected_edge_ids: list[int],
        selected_vertex_ids: list[int],
    ) -> list[list[int]]:
        adjacency: dict[int, set[int]] = {
            node_id: set() for node_id in selected_vertex_ids
        }
        for edge_index in selected_edge_ids:
            source, target = solver_edges[edge_index]
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        components = []
        remaining = set(adjacency)
        while remaining:
            start = min(remaining)
            component = {start}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adjacency.get(node, set()):
                    if neighbor not in component:
                        component.add(neighbor)
                        queue.append(neighbor)
            remaining.difference_update(component)
            components.append(sorted(component))
        return components

    def _resolve_solver(self):
        if self._solver is not None:
            return self._solver
        numpy_major = int(np.__version__.split(".", maxsplit=1)[0])
        if numpy_major >= 2:
            raise ShortestPathExtractionException(
                "PCST evidence construction requires NumPy >=1.26,<2 because "
                "pcst-fast 1.0.10 corrupts returned node and edge arrays with "
                f"NumPy {np.__version__} on Linux AMD64. Reinstall the project "
                "dependencies before rerunning."
            )
        try:
            from pcst_fast import pcst_fast
        except ModuleNotFoundError as error:
            raise ShortestPathExtractionException(
                "PCST evidence construction requires pcst-fast==1.0.10."
            ) from error
        return pcst_fast

    @staticmethod
    def _validate_cost(strategy: str, edge_cost_lambda: float) -> None:
        if strategy not in {"constant", "semantic"}:
            raise ShortestPathExtractionException(
                f"Unsupported PCST edge-cost strategy {strategy}."
            )
        if not math.isfinite(edge_cost_lambda) or edge_cost_lambda <= 0:
            raise ShortestPathExtractionException(
                "PCST edge cost must be finite and greater than zero."
            )

    @staticmethod
    def _original_edge_records(
        instance: WebQSPProcessedInstance,
    ) -> list[tuple[int, str, int]]:
        sources, targets = instance.edge_index.tolist()
        return sorted(
            {
                (int(source), relation, int(target))
                for source, target, relation in zip(
                    sources,
                    targets,
                    instance.edge_relations,
                    strict=True,
                )
                if not relation.startswith("reverse__")
            },
            key=lambda item: (
                instance.nodes[item[0]],
                item[1],
                instance.nodes[item[2]],
                item[0],
                item[2],
            ),
        )

    @classmethod
    def _edge_costs(
        cls,
        *,
        records: list[tuple[int, str, int]],
        strategy: str,
        edge_cost_lambda: float,
        question_embedding: np.ndarray | None,
        relation_embeddings: dict[str, np.ndarray] | None,
    ) -> list[float]:
        if strategy == "constant":
            return [float(edge_cost_lambda)] * len(records)
        if question_embedding is None or relation_embeddings is None:
            raise ShortestPathExtractionException(
                "Semantic PCST requires prepared question and relation embeddings."
            )
        question = cls._normalized_vector(question_embedding, "question")
        costs = []
        for _, relation, _ in records:
            vector = relation_embeddings.get(relation)
            if vector is None:
                raise ShortestPathExtractionException(
                    f"Semantic PCST is missing an embedding for relation {relation}."
                )
            similarity = float(
                np.dot(question, cls._normalized_vector(vector, relation))
            )
            distance = 1.0 - max(-1.0, min(1.0, similarity))
            costs.append(
                max(cls.semantic_cost_epsilon, edge_cost_lambda * distance)
            )
        return costs

    @staticmethod
    def _normalized_vector(vector: np.ndarray, label: str) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        if value.ndim != 1 or not math.isfinite(norm) or norm <= 0:
            raise ShortestPathExtractionException(
                f"Semantic PCST received an invalid embedding for {label}."
            )
        return value / norm

    @classmethod
    def _valid_reachable_candidates(
        cls,
        *,
        instance: WebQSPProcessedInstance,
        records: list[tuple[int, str, int]],
        seeds: list[int],
        candidates: list[CandidateNodeScore],
    ) -> list[tuple[int, int, CandidateNodeScore]]:
        if not seeds:
            return []
        adjacency: dict[int, list[int]] = {}
        for source, _, target in records:
            adjacency.setdefault(source, []).append(target)
            adjacency.setdefault(target, []).append(source)
        reachable = set(seeds)
        queue = deque(seeds)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        result = []
        seen_nodes: set[int] = set()
        for rank, candidate in enumerate(candidates, start=1):
            node_id = cls._candidate_node_id(instance, candidate)
            if node_id is None or node_id not in reachable or node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            result.append((rank, node_id, candidate))
        return result

    @staticmethod
    def _candidate_node_id(
        instance: WebQSPProcessedInstance,
        candidate: CandidateNodeScore,
    ) -> int | None:
        local_id = candidate.local_node_id
        if (
            local_id is not None
            and 0 <= local_id < len(instance.nodes)
            and instance.nodes[local_id] == candidate.node_id
        ):
            return local_id
        return instance.node2id.get(candidate.node_id)

    @staticmethod
    def _empty_result(
        *,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
        valid_candidate_count: int,
        valid_seed_count: int,
        missing_seed_count: int,
        strategy: str,
        edge_cost_lambda: float,
        semantic_embedding_model: str | None,
        started_at: float,
        reason: str,
    ) -> ExtractedReasoningPaths:
        paths = [
            ReasoningPath(
                candidate_node=candidate.node_id,
                candidate_score=candidate.score,
                path_found=False,
            )
            for candidate in candidates
        ]
        return ExtractedReasoningPaths(
            sample=sample.model_copy(update={"graph_triples": []}),
            paths=paths,
            reasoning_subgraph_triples=[],
            reasoning_paths_text="No reasoning subgraph found.",
            found_paths=0,
            missing_paths=len(candidates),
            construction=EvidenceSubgraphConstruction(
                strategy="pcst",
                edge_cost_strategy=strategy,
                edge_cost_lambda=edge_cost_lambda,
                semantic_embedding_model=semantic_embedding_model,
                input_candidate_count=len(candidates),
                valid_candidate_count=valid_candidate_count,
                valid_seed_count=valid_seed_count,
                missing_seed_count=missing_seed_count,
                construction_time_ms=(time.perf_counter() - started_at) * 1000,
                empty_result_reason=reason,
            ),
        )
