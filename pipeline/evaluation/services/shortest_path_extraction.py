"""Shortest path extraction and reasoning subgraph construction."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import time

from pipeline.evaluation.models import (
    CandidateNodeScore,
    EvidenceSubgraphConstruction,
    EvaluationSample,
    ExtractedReasoningPaths,
    GraphTriple,
    ReasoningPath,
)
from pipeline.services import AbstractService
from pipeline.preparation.models.webqsp_local_graph import WebQSPProcessedInstance


class ShortestPathExtractionService(AbstractService):
    """Find shortest paths and merge them into a compact reasoning subgraph."""

    def extract_paths(
        self,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
    ) -> ExtractedReasoningPaths:
        """
        Extract candidate paths and a readable text block for LLM context.

        Args:
            sample: Evaluation sample containing q-entities and local graph triples.
            candidates: Ranked candidate answer nodes.

        Returns:
            ExtractedReasoningPaths: Structured paths and deduplicated subgraph text.
        """
        started_at = time.perf_counter()
        adjacency = self._build_undirected_adjacency(sample.graph_triples)
        paths = [
            self._build_candidate_path(sample, candidate, adjacency)
            for candidate in candidates
        ]
        reasoning_subgraph_triples = self.build_reasoning_subgraph(paths)
        reasoning_paths_text = self.verbalize_subgraph(reasoning_subgraph_triples)
        found_paths = sum(1 for path in paths if path.path_found)

        return ExtractedReasoningPaths(
            sample=sample,
            paths=paths,
            reasoning_subgraph_triples=reasoning_subgraph_triples,
            reasoning_paths_text=reasoning_paths_text,
            found_paths=found_paths,
            missing_paths=len(paths) - found_paths,
            construction=EvidenceSubgraphConstruction(
                strategy="shortest_path",
                input_candidate_count=len(candidates),
                valid_candidate_count=found_paths,
                selected_candidate_count=found_paths,
                selected_candidate_ranks=[
                    rank for rank, path in enumerate(paths, start=1) if path.path_found
                ],
                selected_node_count=len(
                    {
                        node
                        for triple in reasoning_subgraph_triples
                        for node in (triple.source, triple.target)
                    }
                ),
                selected_triple_count=len(reasoning_subgraph_triples),
                valid_seed_count=len(set(sample.q_entities)),
                construction_time_ms=(time.perf_counter() - started_at) * 1000,
                empty_result_reason=(
                    "no_reasoning_paths" if not reasoning_subgraph_triples else None
                ),
            ),
        )

    def extract_paths_from_processed_graph(
        self,
        *,
        instance: WebQSPProcessedInstance,
        sample: EvaluationSample,
        candidates: list[CandidateNodeScore],
    ) -> ExtractedReasoningPaths:
        """Extract all candidate paths directly over compact integer graph data."""
        started_at = time.perf_counter()
        adjacency, edge_records = self._build_integer_adjacency(instance)
        seed_node_ids = [
            instance.node2id[entity]
            for entity in sorted(set(sample.q_entities))
            if entity in instance.node2id
        ]
        candidate_node_ids = [
            self._resolve_candidate_node_id(instance, candidate)
            for candidate in candidates
        ]
        distances, predecessors = self._multi_target_bfs(
            adjacency=adjacency,
            seed_node_ids=seed_node_ids,
            target_node_ids={
                node_id for node_id in candidate_node_ids if node_id is not None
            },
        )
        graph_triple = self._graph_triple_factory(instance.nodes, edge_records)
        edge_path_cache: dict[int, list[list[int]]] = {
            seed_node_id: [[]] for seed_node_id in seed_node_ids
        }

        def edge_paths_to(node_id: int) -> list[list[int]]:
            cached = edge_path_cache.get(node_id)
            if cached is not None:
                return cached
            paths = [
                [*prefix, edge_id]
                for previous_node_id, edge_id in predecessors.get(node_id, [])
                for prefix in edge_paths_to(previous_node_id)
            ]
            edge_path_cache[node_id] = paths
            return paths

        paths: list[ReasoningPath] = []
        for candidate, candidate_node_id in zip(
            candidates,
            candidate_node_ids,
            strict=True,
        ):
            edge_paths = (
                edge_paths_to(candidate_node_id)
                if candidate_node_id is not None
                and candidate_node_id in distances
                else []
            )
            shortest_paths = [
                [graph_triple(edge_id) for edge_id in edge_path]
                for edge_path in edge_paths
            ]
            paths.append(
                ReasoningPath(
                    candidate_node=candidate.node_id,
                    candidate_score=candidate.score,
                    path_found=bool(shortest_paths),
                    triples=shortest_paths[0] if shortest_paths else [],
                    shortest_paths=shortest_paths,
                )
            )

        reasoning_subgraph_triples = self.build_reasoning_subgraph(paths)
        compact_sample = sample.model_copy(
            update={"graph_triples": reasoning_subgraph_triples}
        )
        found_paths = sum(1 for path in paths if path.path_found)
        return ExtractedReasoningPaths(
            sample=compact_sample,
            paths=paths,
            reasoning_subgraph_triples=reasoning_subgraph_triples,
            reasoning_paths_text=self.verbalize_subgraph(
                reasoning_subgraph_triples
            ),
            found_paths=found_paths,
            missing_paths=len(paths) - found_paths,
            construction=EvidenceSubgraphConstruction(
                strategy="shortest_path",
                input_candidate_count=len(candidates),
                valid_candidate_count=sum(
                    node_id is not None for node_id in candidate_node_ids
                ),
                selected_candidate_count=found_paths,
                selected_candidate_ranks=[
                    rank for rank, path in enumerate(paths, start=1) if path.path_found
                ],
                selected_node_count=len(
                    {
                        node
                        for triple in reasoning_subgraph_triples
                        for node in (triple.source, triple.target)
                    }
                ),
                selected_triple_count=len(reasoning_subgraph_triples),
                valid_seed_count=len(seed_node_ids),
                missing_seed_count=len(set(sample.q_entities)) - len(seed_node_ids),
                construction_time_ms=(time.perf_counter() - started_at) * 1000,
                empty_result_reason=(
                    "no_valid_seeds"
                    if not seed_node_ids
                    else "no_candidates"
                    if not candidates
                    else "no_reasoning_paths"
                    if not reasoning_subgraph_triples
                    else None
                ),
            ),
        )

    @staticmethod
    def _build_integer_adjacency(
        instance: WebQSPProcessedInstance,
    ) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, str, int]]]:
        """Build deduplicated undirected integer adjacency from original edges."""
        source_ids, target_ids = instance.edge_index.tolist()
        adjacency: dict[int, list[tuple[int, int]]] = {}
        edge_records: list[tuple[int, str, int]] = []
        seen_triples: set[tuple[int, str, int]] = set()
        for source_id, target_id, relation in zip(
            source_ids,
            target_ids,
            instance.edge_relations,
            strict=True,
        ):
            if relation.startswith("reverse__"):
                continue
            triple_key = (source_id, relation, target_id)
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            edge_id = len(edge_records)
            edge_records.append(triple_key)
            adjacency.setdefault(source_id, []).append((target_id, edge_id))
            adjacency.setdefault(target_id, []).append((source_id, edge_id))

        for neighbors in adjacency.values():
            neighbors.sort(
                key=lambda item: (
                    instance.nodes[item[0]],
                    edge_records[item[1]][1],
                    edge_records[item[1]][0],
                    edge_records[item[1]][2],
                )
            )
        return adjacency, edge_records

    @staticmethod
    def _multi_target_bfs(
        *,
        adjacency: dict[int, list[tuple[int, int]]],
        seed_node_ids: list[int],
        target_node_ids: set[int],
    ) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]]]:
        """Build one shortest-path predecessor DAG for every requested target."""
        unique_seeds = list(dict.fromkeys(seed_node_ids))
        distances = {node_id: 0 for node_id in unique_seeds}
        predecessors: dict[int, list[tuple[int, int]]] = {}
        queue = deque(unique_seeds)
        remaining_targets = target_node_ids.difference(unique_seeds)
        stop_depth: int | None = 0 if not remaining_targets else None

        while queue:
            current_node_id = queue.popleft()
            current_distance = distances[current_node_id]
            if stop_depth is not None and current_distance >= stop_depth:
                break
            next_distance = current_distance + 1
            for next_node_id, edge_id in adjacency.get(current_node_id, []):
                known_distance = distances.get(next_node_id)
                if known_distance is None:
                    distances[next_node_id] = next_distance
                    predecessors[next_node_id] = [(current_node_id, edge_id)]
                    queue.append(next_node_id)
                    remaining_targets.discard(next_node_id)
                    if not remaining_targets:
                        stop_depth = next_distance
                elif known_distance == next_distance:
                    predecessor = (current_node_id, edge_id)
                    if predecessor not in predecessors[next_node_id]:
                        predecessors[next_node_id].append(predecessor)
        return distances, predecessors

    @staticmethod
    def _resolve_candidate_node_id(
        instance: WebQSPProcessedInstance,
        candidate: CandidateNodeScore,
    ) -> int | None:
        local_node_id = candidate.local_node_id
        if (
            local_node_id is not None
            and 0 <= local_node_id < len(instance.nodes)
            and instance.nodes[local_node_id] == candidate.node_id
        ):
            return local_node_id
        return instance.node2id.get(candidate.node_id)

    @staticmethod
    def _graph_triple_factory(
        nodes: list[str],
        edge_records: list[tuple[int, str, int]],
    ) -> Callable[[int], GraphTriple]:
        cache: dict[int, GraphTriple] = {}

        def build(edge_id: int) -> GraphTriple:
            triple = cache.get(edge_id)
            if triple is None:
                source_id, relation, target_id = edge_records[edge_id]
                triple = GraphTriple(
                    source=nodes[source_id],
                    relation=relation,
                    target=nodes[target_id],
                )
                cache[edge_id] = triple
            return triple

        return build

    @staticmethod
    def build_reasoning_subgraph(paths: list[ReasoningPath]) -> list[GraphTriple]:
        """Merge found shortest-path triples into one deduplicated subgraph."""
        seen_triples: set[tuple[str, str, str]] = set()
        subgraph_triples: list[GraphTriple] = []
        for path in paths:
            if not path.path_found:
                continue
            for shortest_path in path.shortest_paths:
                for triple in shortest_path:
                    triple_key = (triple.source, triple.relation, triple.target)
                    if triple_key in seen_triples:
                        continue
                    seen_triples.add(triple_key)
                    subgraph_triples.append(triple)

        return subgraph_triples

    @staticmethod
    def verbalize_subgraph(triples: list[GraphTriple]) -> str:
        """Convert the deduplicated reasoning subgraph into LLM context text."""
        if not triples:
            return "No reasoning subgraph found."

        triple_lines = [
            f"{triple.source} -> {triple.relation} -> {triple.target}"
            for triple in triples
        ]
        return "\n".join(["Reasoning subgraph:", *triple_lines])

    def _build_candidate_path(
        self,
        sample: EvaluationSample,
        candidate: CandidateNodeScore,
        adjacency: dict[str, list[tuple[str, GraphTriple]]],
    ) -> ReasoningPath:
        """Return all equal-length shortest paths from any q-entity to one candidate."""
        shortest_paths = self._find_shortest_path_triples(
            start_nodes=sample.q_entities,
            target_node=candidate.node_id,
            adjacency=adjacency,
        )
        return ReasoningPath(
            candidate_node=candidate.node_id,
            candidate_score=candidate.score,
            path_found=bool(shortest_paths),
            triples=shortest_paths[0] if shortest_paths else [],
            shortest_paths=shortest_paths,
        )

    @staticmethod
    def _build_undirected_adjacency(
        triples: list[GraphTriple],
    ) -> dict[str, list[tuple[str, GraphTriple]]]:
        """Build undirected traversal adjacency while preserving original triples."""
        adjacency: dict[str, list[tuple[str, GraphTriple]]] = {}
        for triple in triples:
            adjacency.setdefault(triple.source, []).append((triple.target, triple))
            adjacency.setdefault(triple.target, []).append((triple.source, triple))

        for neighbors in adjacency.values():
            neighbors.sort(key=lambda item: (item[0], item[1].relation))

        return adjacency

    def _find_shortest_path_triples(
        self,
        start_nodes: list[str],
        target_node: str,
        adjacency: dict[str, list[tuple[str, GraphTriple]]],
    ) -> list[list[GraphTriple]]:
        """Run deterministic BFS and return every equal-length shortest path."""
        if target_node in start_nodes:
            return [[]]

        shortest_paths: list[list[GraphTriple]] = []
        shortest_length: int | None = None
        seen_path_keys: set[tuple[tuple[str, str, str], ...]] = set()
        queue = deque(
            (start_node, [], (start_node,)) for start_node in sorted(start_nodes)
        )

        while queue:
            current_node, path_triples, path_nodes = queue.popleft()
            if shortest_length is not None and len(path_triples) >= shortest_length:
                continue

            for next_node, triple in adjacency.get(current_node, []):
                if next_node in path_nodes:
                    continue

                next_path = [*path_triples, triple]
                next_path_length = len(next_path)
                if shortest_length is not None and next_path_length > shortest_length:
                    continue

                if next_node == target_node:
                    if shortest_length is None:
                        shortest_length = next_path_length
                    path_key = tuple(
                        (item.source, item.relation, item.target)
                        for item in next_path
                    )
                    if path_key not in seen_path_keys:
                        seen_path_keys.add(path_key)
                        shortest_paths.append(next_path)
                    continue

                queue.append((next_node, next_path, (*path_nodes, next_node)))

        return shortest_paths
