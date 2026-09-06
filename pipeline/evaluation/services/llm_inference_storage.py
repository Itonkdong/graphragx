"""Storage service for post-retrieval LLM inference runs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from helpers.logging_config import get_logger
from helpers.constants import (
    LLM_INFERENCE_ANSWERS_FILENAME,
    LLM_INFERENCE_CONFIG_FILENAME,
    LLM_INFERENCE_REASONING_FILENAME,
)
from helpers.path_serialization import make_project_paths_relative, project_absolute_path
from pipeline.preparation.helpers.gnn_architecture import infer_gnn_architecture
from pipeline.evaluation.models import (
    GeneratedAnswerForPrediction,
    GeneratedFinalAnswersBatch,
)
from pipeline.exceptions import PipelineException
from pipeline.services import AbstractService

logger = get_logger(__name__)


class LlmInferenceStoragePayload(BaseModel):
    """Data persisted for one LLM inference run."""

    answers: GeneratedFinalAnswersBatch


class LlmInferenceStorageResult(BaseModel):
    """Paths and version metadata produced by inference storage."""

    inference_run_directory: Path
    inference_run_name: str
    inference_run_number: int
    reasoning_path: Path
    answers_path: Path
    inference_config_path: Path


class CreatedLlmInferenceRun(BaseModel):
    """Open inference run paths used for batched appends."""

    inference_run_directory: Path
    inference_run_name: str
    inference_run_number: int
    reasoning_path: Path
    answers_path: Path
    inference_config_path: Path


class LlmInferenceStorageService(AbstractService):
    """Persist numbered LLM inference runs."""

    reasoning_filename = LLM_INFERENCE_REASONING_FILENAME
    answers_filename = LLM_INFERENCE_ANSWERS_FILENAME
    inference_config_filename = LLM_INFERENCE_CONFIG_FILENAME

    def save_inference_run(
        self,
        inference_root: Path,
        run_name: str | None,
        payload: LlmInferenceStoragePayload,
    ) -> LlmInferenceStorageResult:
        """Create a numbered inference run directory and persist all outputs."""
        try:
            run = self.create_inference_run(
                inference_root=inference_root,
                run_name=run_name,
            )
            self.append_inference_batch(run=run, answers=payload.answers)
            self.write_inference_config(run=run, answers=payload.answers)
        except OSError as error:
            raise PipelineException(f"Could not save LLM inference run: {error}") from error

        logger.info(
            f"Saved LLM inference run: directory={run.inference_run_directory} "
            f"answers={run.answers_path} reasoning={run.reasoning_path} "
            f"inference_config={run.inference_config_path}"
        )
        return LlmInferenceStorageResult(
            inference_run_directory=run.inference_run_directory,
            inference_run_name=run.inference_run_name,
            inference_run_number=run.inference_run_number,
            reasoning_path=run.reasoning_path,
            answers_path=run.answers_path,
            inference_config_path=run.inference_config_path,
        )

    def create_inference_run(
        self,
        inference_root: Path,
        run_name: str | None,
    ) -> CreatedLlmInferenceRun:
        """Create a numbered inference run directory and empty JSONL files."""
        try:
            inference_run_directory = self._create_inference_run_directory(
                inference_root=inference_root,
                run_name=run_name,
            )
            run = CreatedLlmInferenceRun(
                inference_run_directory=inference_run_directory,
                inference_run_name=inference_run_directory.name,
                inference_run_number=self._extract_run_number(
                    inference_run_directory.name
                ),
                reasoning_path=inference_run_directory / self.reasoning_filename,
                answers_path=inference_run_directory / self.answers_filename,
                inference_config_path=(
                    inference_run_directory / self.inference_config_filename
                ),
            )
            for path in [
                run.reasoning_path,
                run.answers_path,
            ]:
                path.write_text("", encoding="utf-8")
            return run
        except OSError as error:
            raise PipelineException(f"Could not create LLM inference run: {error}") from error

    def append_inference_batch(
        self,
        run: CreatedLlmInferenceRun,
        answers: GeneratedFinalAnswersBatch,
    ) -> None:
        """Append one generated-answer batch to persisted run artifacts."""
        try:
            self._append_jsonl(
                run.reasoning_path,
                self._iter_reasoning(LlmInferenceStoragePayload(answers=answers)),
            )
            self._append_jsonl(
                run.answers_path,
                self._iter_answers(LlmInferenceStoragePayload(answers=answers)),
            )
        except OSError as error:
            raise PipelineException(f"Could not append LLM inference batch: {error}") from error

    def write_inference_config(
        self,
        run: CreatedLlmInferenceRun,
        answers: GeneratedFinalAnswersBatch,
    ) -> None:
        """Write or replace the inference config file."""
        try:
            run.inference_config_path.write_text(
                json.dumps(
                    make_project_paths_relative(
                        self._build_summary(
                            run,
                            LlmInferenceStoragePayload(answers=answers),
                        )
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError as error:
            raise PipelineException(f"Could not write LLM inference config: {error}") from error

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, sort_keys=True, default=str))
                output_file.write("\n")

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("a", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, sort_keys=True, default=str))
                output_file.write("\n")

    @staticmethod
    def _iter_reasoning(
        payload: LlmInferenceStoragePayload,
    ) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "q_entity": item.q_entity,
                "subgraph": [
                    triple.model_dump(mode="json")
                    for triple in item.reasoning_subgraph_triples
                ],
                "construction": item.evidence_construction.model_dump(mode="json"),
                "analytics": LlmInferenceStorageService._build_reasoning_analytics(item),
            }
            for item in payload.answers.items
        ]

    @staticmethod
    def _build_reasoning_analytics(
        item: GeneratedAnswerForPrediction,
    ) -> dict[str, Any]:
        subgraph = item.reasoning_subgraph_triples
        nodes = [
            node
            for triple in subgraph
            for node in (triple.source, triple.target)
        ]
        relation_names = [triple.relation for triple in subgraph]
        path_lengths = item.reasoning_path_lengths
        total_nodes = len(nodes)
        total_distinct_nodes = len(set(nodes))
        total_relations = len(relation_names)
        return {
            "answer_candidate_count": len(item.answer_candidates),
            "gold_answer_count": len(item.a_entity),
            "total_subgraph_triples": len(subgraph),
            "total_relations": total_relations,
            "total_distinct_relations": len(set(relation_names)),
            "total_nodes": total_nodes,
            "total_distinct_nodes": total_distinct_nodes,
            "duplicate_node_references": total_nodes - total_distinct_nodes,
            "found_reasoning_paths": item.found_reasoning_paths,
            "missing_reasoning_paths": item.missing_reasoning_paths,
            "max_length": max(path_lengths, default=0),
            "min_length": min(path_lengths, default=0),
            "average_length": (
                sum(path_lengths) / len(path_lengths)
                if path_lengths
                else 0.0
            ),
            "candidate_path_coverage": (
                item.found_reasoning_paths / len(item.answer_candidates)
                if item.answer_candidates
                else 0.0
            ),
            "has_gold_answer_candidate": any(
                candidate in item.a_entity for candidate in item.answer_candidates
            ),
            "candidate_evidence_coverage": (
                item.evidence_construction.candidate_evidence_coverage
            ),
        }

    @staticmethod
    def _iter_answers(payload: LlmInferenceStoragePayload) -> list[dict[str, Any]]:
        return [
            {
                "instance_index": item.instance_index,
                "question": item.question,
                "q_entity": item.q_entity,
                "gold_answers": item.a_entity,
                "answer_candidates": item.answer_candidates,
                "model_id": item.model_id,
                "llm_provider": item.llm_provider,
                "answers": item.answers,
                "explanation": item.explanation,
                "raw_response": item.raw_response,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "estimated_cost_usd": item.estimated_cost_usd,
                "error_message": item.error_message,
            }
            for item in payload.answers.items
        ]

    @classmethod
    def _build_summary(
        cls,
        run: CreatedLlmInferenceRun,
        payload: LlmInferenceStoragePayload,
    ) -> dict[str, Any]:
        answers = payload.answers
        evaluation_run_directory = (
            run.inference_run_directory.parent.parent
            / "evaluations"
            / answers.evaluation_run_name
        )
        total_prompt_tokens = sum(item.prompt_tokens for item in answers.items)
        total_completion_tokens = sum(
            item.completion_tokens for item in answers.items
        )
        total_tokens = sum(item.total_tokens for item in answers.items)
        total_cost = sum(item.estimated_cost_usd for item in answers.items)
        evaluation_config_path = evaluation_run_directory / "evaluation_config.json"
        gnn_architecture = cls._load_inference_architecture(evaluation_config_path)
        relation_vocabulary_path = cls._load_relation_vocabulary_path(
            evaluation_config_path
        )
        return {
            "dataset_id": answers.dataset_id,
            "gnn_architecture": gnn_architecture,
            "run_name": run.inference_run_name,
            "run_number": run.inference_run_number,
            "evaluation_config": {
                "evaluation_run_name": answers.evaluation_run_name,
                "evaluation_run_number": cls._extract_run_number(
                    answers.evaluation_run_name
                ),
                "full_config_path": str(
                    evaluation_config_path
                ),
                "predictions_path": str(
                    evaluation_run_directory
                    / "predictions.jsonl"
                ),
                **(
                    {"relation_vocabulary_path": str(relation_vocabulary_path)}
                    if relation_vocabulary_path is not None
                    else {}
                ),
            },
            "inference": {
                "model_id": answers.model_id,
                "llm_provider": answers.llm_provider,
                "reasoning_effort": answers.reasoning_effort,
                "generate_explanation": answers.generate_explanation,
                **(
                    {"batch_size": answers.inference_batch_size}
                    if answers.inference_batch_size is not None
                    else {}
                ),
                "parallel_calls": answers.inference_parallel_calls,
                "total_requests": len(answers.items),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 8),
                "evidence_subgraph": answers.evidence_subgraph,
                "evidence_metrics": cls._build_evidence_metrics(answers),
            },
            "successful_answers": answers.successful_answers,
            "failed_answers": answers.failed_answers,
        }

    @staticmethod
    def _build_evidence_metrics(
        answers: GeneratedFinalAnswersBatch,
    ) -> dict[str, float | int]:
        """Aggregate evidence size, coverage, timing, and PCST objective values."""
        items = answers.items
        count = len(items)
        if count == 0:
            return {
                "average_subgraph_triples": 0.0,
                "average_distinct_nodes": 0.0,
                "average_candidate_evidence_coverage": 0.0,
                "candidate_reduction_percentage": 0.0,
                "empty_subgraph_count": 0,
                "empty_subgraph_rate": 0.0,
                "average_construction_time_ms": 0.0,
            }

        constructions = [item.evidence_construction for item in items]
        distinct_node_counts = [
            len({
                node
                for triple in item.reasoning_subgraph_triples
                for node in (triple.source, triple.target)
            })
            for item in items
        ]
        empty_count = sum(
            not item.reasoning_subgraph_triples for item in items
        )
        found_candidates = sum(item.found_reasoning_paths for item in items)
        missing_candidates = sum(item.missing_reasoning_paths for item in items)
        total_candidates = found_candidates + missing_candidates
        metrics: dict[str, float | int] = {
            "average_subgraph_triples": sum(
                len(item.reasoning_subgraph_triples) for item in items
            ) / count,
            "average_distinct_nodes": sum(distinct_node_counts) / count,
            "average_candidate_evidence_coverage": sum(
                construction.candidate_evidence_coverage
                for construction in constructions
            ) / count,
            "candidate_reduction_percentage": (
                100.0 * missing_candidates / total_candidates
                if total_candidates
                else 0.0
            ),
            "empty_subgraph_count": empty_count,
            "empty_subgraph_rate": empty_count / count,
            "average_construction_time_ms": sum(
                construction.construction_time_ms
                for construction in constructions
            ) / count,
        }
        pcst = [
            construction
            for construction in constructions
            if construction.strategy == "pcst"
        ]
        if pcst:
            metrics.update(
                average_collected_prize=sum(
                    item.collected_prize for item in pcst
                ) / len(pcst),
                average_edge_cost=sum(
                    item.total_edge_cost for item in pcst
                ) / len(pcst),
                average_objective=sum(item.objective for item in pcst) / len(pcst),
            )
        return metrics

    @staticmethod
    def _load_inference_architecture(evaluation_config_path: Path) -> str:
        try:
            evaluation_config = json.loads(
                evaluation_config_path.read_text(encoding="utf-8")
            )
            if not isinstance(evaluation_config, dict):
                return "graphsage"
            model_reference = evaluation_config.get("model_config", {})
            model_config_value = (
                model_reference.get("full_config_path")
                if isinstance(model_reference, dict)
                else None
            )
            if isinstance(model_config_value, str):
                model_config = json.loads(
                    project_absolute_path(model_config_value).read_text(encoding="utf-8")
                )
                if isinstance(model_config, dict):
                    return infer_gnn_architecture(model_config)
            explicit = evaluation_config.get("gnn_architecture")
            if isinstance(explicit, str):
                return explicit
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return "graphsage"

    @staticmethod
    def _load_relation_vocabulary_path(
        evaluation_config_path: Path,
    ) -> Path | None:
        """Resolve an optional R-GCN vocabulary reference for inference lineage."""
        try:
            evaluation_config = json.loads(
                evaluation_config_path.read_text(encoding="utf-8")
            )
            model_reference = evaluation_config.get("model_config", {})
            value = (
                model_reference.get("relation_vocabulary_path")
                if isinstance(model_reference, dict)
                else None
            )
            if isinstance(value, str):
                path = project_absolute_path(value)
                if path.exists():
                    return path
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return None

    def _create_inference_run_directory(
        self,
        inference_root: Path,
        run_name: str | None,
    ) -> Path:
        inference_root.mkdir(parents=True, exist_ok=True)
        run_number = self._next_run_number(inference_root)
        run_label = self._resolve_run_label(run_name)
        inference_run_directory = inference_root / f"{run_number}_{run_label}"
        inference_run_directory.mkdir(parents=True, exist_ok=False)
        return inference_run_directory

    @classmethod
    def _next_run_number(cls, inference_root: Path) -> int:
        existing_run_numbers = [
            cls._extract_run_number(path.name)
            for path in inference_root.iterdir()
            if path.is_dir() and cls._extract_run_number(path.name) > 0
        ]
        return max(existing_run_numbers, default=0) + 1

    @classmethod
    def _resolve_run_label(cls, run_name: str | None) -> str:
        if run_name is None or not run_name.strip():
            return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        return cls._sanitize_run_name(run_name)

    @staticmethod
    def _sanitize_run_name(run_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
        sanitized = sanitized.strip("._-")
        return sanitized or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _extract_run_number(run_directory_name: str) -> int:
        run_number_match = re.match(r"^(\d+)_", run_directory_name)
        if run_number_match is None:
            return 0

        return int(run_number_match.group(1))
