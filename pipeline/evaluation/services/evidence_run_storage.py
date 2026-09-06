"""Persistence and aggregate metrics for evidence-only runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from string import punctuation
from typing import Any

from helpers.path_serialization import make_project_paths_relative
from pipeline.evaluation.models import (
    ExtractedReasoningPathsBatch,
    GnnAnswerRetrieverEvaluationResult,
    SavedEvidenceSubgraphRun,
)
from pipeline.exceptions import PipelineException
from pipeline.preparation.steps.configuration_building import BuiltPipelineConfiguration
from pipeline.services import AbstractService


class EvidenceRunStorageService(AbstractService):
    """Save evidence rows and metrics without creating an inference artifact."""

    config_filename = "evidence_config.json"
    rows_filename = "evidence_subgraphs.jsonl"
    metrics_filename = "evidence_metrics.json"
    whitespace_pattern = re.compile(r"\s+")
    article_pattern = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)

    def save(
        self,
        *,
        paths_batch: ExtractedReasoningPathsBatch,
        evaluation_result: GnnAnswerRetrieverEvaluationResult,
        configuration: BuiltPipelineConfiguration,
        evidence_root: Path,
        run_name: str | None,
    ) -> SavedEvidenceSubgraphRun:
        try:
            run_directory = self._create_run_directory(evidence_root, run_name)
            run_number = self._extract_run_number(run_directory.name)
            rows_path = run_directory / self.rows_filename
            metrics_path = run_directory / self.metrics_filename
            config_path = run_directory / self.config_filename
            rows = [self._build_row(item) for item in paths_batch.items]
            metrics = self.build_metrics(rows)
            evidence_configuration = self._evidence_configuration(configuration)
            with rows_path.open("w", encoding="utf-8") as output:
                for row in rows:
                    output.write(json.dumps(row, sort_keys=True))
                    output.write("\n")
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            config_payload = {
                "dataset_id": paths_batch.dataset_id,
                "gnn_architecture": evaluation_result.gnn_architecture,
                "run_name": run_directory.name,
                "run_number": run_number,
                "evaluation_config": {
                    "evaluation_run_name": evaluation_result.evaluation_run_name,
                    "evaluation_run_number": evaluation_result.evaluation_run_number,
                    "full_config_path": str(evaluation_result.evaluation_config_path),
                    "predictions_path": str(evaluation_result.predictions_path),
                },
                "evidence": {
                    **evidence_configuration,
                    "evaluated_instances": len(rows),
                    "evidence_metrics": metrics,
                },
                "artifacts": {
                    "evidence_subgraphs_path": str(rows_path),
                    "evidence_metrics_path": str(metrics_path),
                },
            }
            config_path.write_text(
                json.dumps(
                    make_project_paths_relative(config_payload),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as error:
            raise PipelineException(f"Could not save evidence run: {error}") from error

        return SavedEvidenceSubgraphRun(
            dataset_id=paths_batch.dataset_id,
            gnn_architecture=evaluation_result.gnn_architecture,
            evaluation_run_name=evaluation_result.evaluation_run_name,
            evidence_run_directory=run_directory,
            evidence_run_name=run_directory.name,
            evidence_run_number=run_number,
            evaluated_instances=len(rows),
            subgraph_algorithm=configuration.subgraph_construction_algorithm,
            evidence_configuration=evidence_configuration,
            evidence_metrics=metrics,
            evidence_config_path=config_path,
            evidence_subgraphs_path=rows_path,
            evidence_metrics_path=metrics_path,
        )

    @classmethod
    def _build_row(cls, item) -> dict[str, Any]:
        extracted = item.extracted_paths
        triples = extracted.reasoning_subgraph_triples
        gold = cls._normalize_set(item.prediction.a_entity)
        context = cls._normalize_set(
            [node for triple in triples for node in (triple.source, triple.target)]
        )
        visible_gold = gold & context
        coverage = len(visible_gold) / len(gold) if gold else 0.0
        return {
            "instance_index": item.instance_index,
            "question": item.prediction.question,
            "q_entity": item.prediction.q_entity,
            "gold_answers": item.prediction.a_entity,
            "answer_candidates": [
                candidate.node for candidate in item.prediction.answer_candidates
            ],
            "subgraph": [triple.model_dump(mode="json") for triple in triples],
            "construction": extracted.construction.model_dump(mode="json"),
            "analytics": {
                "total_subgraph_triples": len(triples),
                "total_distinct_nodes": len({
                    node
                    for triple in triples
                    for node in (triple.source, triple.target)
                }),
                "found_reasoning_paths": extracted.found_paths,
                "missing_reasoning_paths": extracted.missing_paths,
                "candidate_evidence_coverage": (
                    extracted.construction.candidate_evidence_coverage
                ),
                "context_visible_gold_answers": sorted(visible_gold),
                "reasoning_context_gold_coverage": coverage,
                "full_gold_context": bool(gold) and visible_gold == gold,
            },
        }

    @classmethod
    def build_metrics(cls, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        count = len(rows)
        if count == 0:
            return {
                "average_subgraph_triples": 0.0,
                "average_distinct_nodes": 0.0,
                "average_candidate_evidence_coverage": 0.0,
                "candidate_reduction_percentage": 0.0,
                "empty_subgraph_count": 0,
                "empty_subgraph_rate": 0.0,
                "average_construction_time_ms": 0.0,
                "reasoning_context_gold_coverage": 0.0,
                "reasoning_context_full_gold_coverage_rate": 0.0,
            }
        constructions = [row["construction"] for row in rows]
        analytics = [row["analytics"] for row in rows]
        found = sum(int(item["found_reasoning_paths"]) for item in analytics)
        missing = sum(int(item["missing_reasoning_paths"]) for item in analytics)
        total_candidates = found + missing
        eligible = [
            item for row, item in zip(rows, analytics, strict=True)
            if cls._normalize_set(row["gold_answers"])
        ]
        metrics: dict[str, float | int] = {
            "average_subgraph_triples": sum(
                int(item["total_subgraph_triples"]) for item in analytics
            ) / count,
            "average_distinct_nodes": sum(
                int(item["total_distinct_nodes"]) for item in analytics
            ) / count,
            "average_candidate_evidence_coverage": sum(
                float(item["candidate_evidence_coverage"]) for item in analytics
            ) / count,
            "candidate_reduction_percentage": (
                100.0 * missing / total_candidates if total_candidates else 0.0
            ),
            "empty_subgraph_count": sum(
                int(item["total_subgraph_triples"]) == 0 for item in analytics
            ),
            "empty_subgraph_rate": sum(
                int(item["total_subgraph_triples"]) == 0 for item in analytics
            ) / count,
            "average_construction_time_ms": sum(
                float(item["construction_time_ms"]) for item in constructions
            ) / count,
            "reasoning_context_gold_coverage": (
                sum(float(item["reasoning_context_gold_coverage"]) for item in eligible)
                / len(eligible)
                if eligible else 0.0
            ),
            "reasoning_context_full_gold_coverage_rate": (
                sum(bool(item["full_gold_context"]) for item in eligible) / len(eligible)
                if eligible else 0.0
            ),
        }
        pcst = [item for item in constructions if item.get("strategy") == "pcst"]
        if pcst:
            metrics.update(
                average_collected_prize=sum(
                    float(item["collected_prize"]) for item in pcst
                ) / len(pcst),
                average_edge_cost=sum(float(item["total_edge_cost"]) for item in pcst)
                / len(pcst),
                average_objective=sum(float(item["objective"]) for item in pcst)
                / len(pcst),
            )
        return metrics

    @staticmethod
    def _evidence_configuration(
        configuration: BuiltPipelineConfiguration,
    ) -> dict[str, object]:
        algorithm = configuration.subgraph_construction_algorithm
        if algorithm != "pcst":
            return {"algorithm": algorithm}
        return {
            "algorithm": "pcst",
            "pcst": {
                "prize_strategy": "linear_rank",
                "edge_cost_strategy": configuration.pcst_edge_cost_strategy,
                "edge_cost_lambda": configuration.pcst_edge_cost,
                "semantic_embedding_model": (
                    configuration.embedding_model
                    if configuration.pcst_edge_cost_strategy == "semantic"
                    else None
                ),
                "semantic_cost_formula": "max(1e-6, lambda * (1 - cosine))",
                "solver": "pcst_fast",
                "pruning": "gw",
            },
        }

    @classmethod
    def _normalize_set(cls, values: list[str]) -> set[str]:
        normalized: set[str] = set()
        for value in values:
            item = value.lower().strip().translate(str.maketrans("", "", punctuation))
            item = cls.article_pattern.sub(" ", item)
            item = cls.whitespace_pattern.sub(" ", item).strip()
            if item:
                normalized.add(item)
        return normalized

    @classmethod
    def _create_run_directory(cls, root: Path, run_name: str | None) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        number = max(
            (cls._extract_run_number(path.name) for path in root.iterdir() if path.is_dir()),
            default=0,
        ) + 1
        label = cls._sanitize_run_name(run_name)
        path = root / f"{number}_{label}"
        path.mkdir(exist_ok=False)
        return path

    @staticmethod
    def _extract_run_number(name: str) -> int:
        match = re.match(r"^(\d+)_", name)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _sanitize_run_name(run_name: str | None) -> str:
        if run_name and run_name.strip():
            value = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
            if value.strip("._-"):
                return value.strip("._-")
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
