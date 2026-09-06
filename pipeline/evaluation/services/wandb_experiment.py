"""Shared resumable Weights & Biases experiment lifecycle."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from helpers.env_variables import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT
from helpers.logging_config import get_logger
from pipeline.exceptions import PipelineException

logger = get_logger(__name__)


class WandbTrackingMetadata(BaseModel):
    """Persisted W&B lineage and latest integration status."""

    status: Literal["logged", "failed", "skipped"]
    run_id: str | None = None
    run_name: str | None = None
    run_url: str | None = None
    project: str | None = None
    entity: str | None = None
    error_message: str | None = None


class WandbRunIdentifierService:
    """Allocate dataset-wide sequential names for newly created W&B runs."""

    def allocate(self, run_root: Path) -> str:
        """Create and return the next ``number_timestamp`` run identifier."""
        run_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        while True:
            run_number = max(
                (
                    self._extract_run_number(path.name)
                    for path in run_root.iterdir()
                    if path.is_dir()
                ),
                default=0,
            ) + 1
            run_name = f"{run_number}_{timestamp}"
            try:
                (run_root / run_name).mkdir(exist_ok=False)
                return run_name
            except FileExistsError:
                continue

    @staticmethod
    def _extract_run_number(name: str) -> int:
        match = re.match(r"^(\d+)_", name)
        return int(match.group(1)) if match else 0


class WandbExperimentCoordinator:
    """Own one W&B run that can be shared and resumed across pipeline stages."""

    def __init__(
        self,
        *,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        enabled: bool = True,
        resume_from_lineage: bool = True,
        run_root: Path | None = None,
        run_identifier_service: WandbRunIdentifierService | None = None,
        architecture_name: str | None = None,
    ) -> None:
        self.requested_project = project
        self.requested_entity = entity
        self.project = project or WANDB_PROJECT
        self.entity = entity if entity is not None else WANDB_ENTITY
        self.mode = mode or WANDB_MODE
        self.enabled = enabled
        self.resume_from_lineage = resume_from_lineage
        self.run_root = run_root
        self.run_identifier_service = (
            run_identifier_service or WandbRunIdentifierService()
        )
        self.architecture_name = architecture_name
        self._wandb: Any | None = None
        self._run: Any | None = None
        self._metadata = WandbTrackingMetadata(status="skipped")
        self._metadata_paths: set[Path] = set()
        self._config_payload: dict[str, Any] = {}
        self._tags: list[str] = ["graphragx"]
        self._logged_training_epochs: set[int] = set()
        self._logged_aggregate_metrics: dict[str, float | int | str] = {}

    @property
    def metadata(self) -> WandbTrackingMetadata:
        """Return the latest immutable tracking metadata snapshot."""
        return self._metadata.model_copy(deep=True)

    @property
    def has_active_run(self) -> bool:
        """Return whether this coordinator currently owns an initialized run."""
        return self._run is not None

    def ensure_run(
        self,
        *,
        source_config_path: Path | None = None,
        architecture_name: str | None = None,
    ) -> WandbTrackingMetadata:
        """Initialize or resume the shared run using optional persisted lineage."""
        if not self.enabled:
            return self.metadata
        if self._run is not None:
            return self.metadata
        if architecture_name:
            self.architecture_name = architecture_name

        lineage = (
            self._load_lineage(source_config_path)
            if self.resume_from_lineage
            else None
        )
        if lineage is not None:
            self._validate_lineage(lineage)
            self.project = lineage.project or self.project
            self.entity = lineage.entity if lineage.entity is not None else self.entity
        run_name = lineage.run_name if lineage is not None else None
        if lineage is None:
            if self.run_root is None:
                raise PipelineException(
                    "Creating a W&B run requires a configured run identifier root."
                )
            run_name = self.run_identifier_service.allocate(self.run_root)
            logger.info(f"Allocated W&B run identifier: {run_name}")
            if self.architecture_name:
                run_name = f"{run_name}_{self.architecture_name}"
        try:
            self._wandb = importlib.import_module("wandb")
            init_kwargs: dict[str, Any] = {
                "project": self.project,
                "entity": self.entity,
                "mode": self.mode,
                "name": run_name,
                "job_type": "graphragx-pipeline",
            }
            if lineage is not None and lineage.run_id:
                init_kwargs.update({"id": lineage.run_id, "resume": "allow"})
            self._run = self._wandb.init(**init_kwargs)
            existing_tags = getattr(self._run, "tags", ())
            if isinstance(existing_tags, str):
                existing_tags = (existing_tags,)
            if isinstance(existing_tags, list | tuple):
                self._tags = list(
                    dict.fromkeys([*existing_tags, *self._tags])
                )
            self._apply_tags()
            run_config = getattr(self._run, "config", None)
            if isinstance(run_config, Mapping) or hasattr(run_config, "get"):
                existing_config: dict[str, Any] = {}
                for key in [
                    "dataset_id",
                    "model_id",
                    "gnn_architecture",
                    "runs",
                    "configs",
                    "source_paths",
                ]:
                    try:
                        value = run_config.get(key)
                    except (AttributeError, KeyError, TypeError):
                        continue
                    if value is not None:
                        existing_config[key] = value
                self._config_payload = existing_config
            if hasattr(self._run, "define_metric"):
                self._run.define_metric("Training/global_step")
                self._run.define_metric(
                    "Training/loss",
                    step_metric="Training/global_step",
                )
                self._run.define_metric("Training/epoch")
                self._run.define_metric(
                    "Training/gnn_training_loss",
                    step_metric="Training/epoch",
                )
            run_id = str(getattr(self._run, "id", "")) or None
            run_url = str(getattr(self._run, "url", "")) or None
            self._metadata = WandbTrackingMetadata(
                status="logged",
                run_id=run_id,
                run_name=run_name,
                run_url=run_url,
                project=self.project,
                entity=self.entity,
            )
        except Exception as error:
            logger.warning(f"WandB experiment initialization failed: {error}")
            self._metadata = WandbTrackingMetadata(
                status="failed",
                run_name=run_name,
                project=self.project,
                entity=self.entity,
                error_message=str(error),
            )
        return self.metadata

    def update_config(
        self,
        payload: dict[str, Any],
        *,
        source_config_path: Path | None = None,
    ) -> None:
        """Merge available stage configuration into the shared W&B run config."""
        self.ensure_run(source_config_path=source_config_path)
        if self._run is None:
            return
        self._config_payload = self._deep_merge(self._config_payload, payload)
        self.update_tags(self._build_tags_from_config(self._config_payload))
        run_config = getattr(self._run, "config", None)
        if run_config is None or not hasattr(run_config, "update"):
            return
        try:
            try:
                run_config.update(self._config_payload, allow_val_change=True)
            except TypeError:
                run_config.update(self._config_payload)
        except Exception as error:
            self._record_failure("WandB config update failed", error)

    def update_tags(self, tags: list[str]) -> None:
        """Add available stage tags without dropping tags from earlier stages."""
        self.ensure_run()
        if self._run is None:
            return
        self._tags = list(dict.fromkeys([*self._tags, *filter(None, tags)]))
        self._apply_tags()

    def update_inference_run_name(
        self,
        *,
        evidence_algorithm: str | None,
        model_id: str | None,
    ) -> None:
        """Append available inference identity to the active W&B run name."""
        self.ensure_run()
        if self._run is None:
            return
        current_name = self._metadata.run_name or getattr(self._run, "name", None)
        if not isinstance(current_name, str) or not current_name:
            return
        updated_name = self.build_inference_run_name(
            current_name,
            evidence_algorithm=evidence_algorithm,
            model_id=model_id,
        )
        if updated_name == current_name:
            return
        try:
            self._run.name = updated_name
            self._metadata = self._metadata.model_copy(
                update={"run_name": updated_name}
            )
            for path in self._metadata_paths:
                self._write_metadata(path)
        except Exception as error:
            self._record_failure("W&B run-name update failed", error)

    @classmethod
    def build_inference_run_name(
        cls,
        run_name: str,
        *,
        evidence_algorithm: str | None,
        model_id: str | None,
    ) -> str:
        """Return a stable title with optional evidence and model suffixes."""
        algorithm_name = (
            "sp" if evidence_algorithm == "shortest_path" else evidence_algorithm
        )
        components = [
            cls._run_name_component(algorithm_name),
            cls._run_name_component(model_id),
        ]
        suffix = "_".join(component for component in components if component)
        if not suffix or run_name.endswith(f"_{suffix}"):
            return run_name
        return f"{run_name}_{suffix}"

    @staticmethod
    def _run_name_component(value: str | None) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_")

    def log(
        self,
        payload: dict[str, float | int | str],
        *,
        source_config_path: Path | None = None,
        step: int | None = None,
        architecture_name: str | None = None,
    ) -> None:
        """Best-effort log scalar stage data to the active experiment."""
        self.ensure_run(
            source_config_path=source_config_path,
            architecture_name=architecture_name,
        )
        if self._run is None:
            return
        try:
            if step is None:
                self._run.log(payload)
            else:
                self._run.log(payload, step=step)
        except Exception as error:
            self._record_failure("WandB metric logging failed", error)

    def log_aggregate_metrics(
        self,
        payload: dict[str, float | int | str],
        *,
        source_config_path: Path | None = None,
        architecture_name: str | None = None,
    ) -> None:
        """History-log each aggregate scalar once so W&B creates bar panels."""
        self.ensure_run(
            source_config_path=source_config_path,
            architecture_name=architecture_name,
        )
        if self._run is None:
            return
        updated_payload = {
            key: value
            for key, value in payload.items()
            if key in self._logged_aggregate_metrics
            and self._logged_aggregate_metrics[key] != value
        }
        new_payload = {
            key: value
            for key, value in payload.items()
            if key not in self._logged_aggregate_metrics
        }
        if not new_payload and not updated_payload:
            return
        try:
            if new_payload:
                self._run.log(new_payload)
                self._logged_aggregate_metrics.update(new_payload)
            if updated_payload:
                summary = getattr(self._run, "summary", None)
                if summary is None or not hasattr(summary, "update"):
                    raise TypeError("Active W&B run has no mutable summary.")
                summary.update(updated_payload)
                self._logged_aggregate_metrics.update(updated_payload)
        except Exception as error:
            self._record_failure("WandB aggregate metric logging failed", error)

    def log_training_progress(self, payload: dict[str, float | int | str]) -> None:
        """Log one live training progress event using optimizer progress as step."""
        global_step = int(payload["global_step"])
        self.log(
            {
                "Training/loss": float(payload["loss"]),
                "Training/epoch": int(payload["epoch"]),
                "Training/instance": int(payload["instance"]),
                "Training/global_step": global_step,
            },
            architecture_name=(
                str(payload["gnn_architecture"])
                if payload.get("gnn_architecture")
                else None
            ),
        )

    def log_training_epoch(self, payload: dict[str, float | int | str]) -> None:
        """Log one completed epoch average immediately on the epoch axis."""
        epoch = int(payload["epoch"])
        if epoch in self._logged_training_epochs:
            return
        self.log(
            {
                "Training/gnn_training_loss": float(payload["average_loss"]),
                "Training/epoch": epoch,
            },
            architecture_name=(
                str(payload["gnn_architecture"])
                if payload.get("gnn_architecture")
                else None
            ),
        )
        if self._run is not None:
            self._logged_training_epochs.add(epoch)

    def log_artifact(
        self,
        *,
        name: str,
        artifact_type: str,
        paths: list[Path],
        source_config_path: Path | None = None,
    ) -> None:
        """Best-effort upload a collection of existing local artifact files."""
        self.ensure_run(source_config_path=source_config_path)
        if self._run is None or self._wandb is None:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            for path in paths:
                if path.exists():
                    artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as error:
            self._record_failure("WandB artifact logging failed", error)

    def persist_metadata(self, path: Path) -> None:
        """Merge the latest W&B tracking metadata into a JSON config artifact."""
        self._metadata_paths.add(path)
        self._write_metadata(path)

    def _write_metadata(self, path: Path) -> None:
        """Write tracking metadata without changing the registered path set."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("config root must be an object")
            payload["wandb"] = self.metadata.model_dump(mode="json")
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            logger.warning(f"Could not persist WandB metadata to {path}: {error}")

    def finish(self) -> None:
        """Finish the active W&B run without changing pipeline success semantics."""
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as error:
            self._record_failure("WandB experiment finalization failed", error)
            for path in self._metadata_paths:
                self._write_metadata(path)
        finally:
            self._run = None

    def _load_lineage(self, path: Path | None) -> WandbTrackingMetadata | None:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tracking = payload.get("wandb") if isinstance(payload, dict) else None
            if not isinstance(tracking, dict) or not tracking.get("run_id"):
                return None
            return WandbTrackingMetadata.model_validate(tracking)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            logger.warning(f"Ignoring invalid WandB lineage in {path}: {error}")
            return None

    def _validate_lineage(self, lineage: WandbTrackingMetadata) -> None:
        conflicts: list[str] = []
        if self.requested_project is not None and lineage.project != self.requested_project:
            conflicts.append("project")
        if self.requested_entity is not None and lineage.entity != self.requested_entity:
            conflicts.append("entity")
        if conflicts:
            raise PipelineException(
                "WandB options conflict with persisted experiment lineage: "
                + ", ".join(conflicts)
            )

    def _record_failure(self, message: str, error: Exception) -> None:
        logger.warning(f"{message}: {error}")
        self._metadata = self._metadata.model_copy(
            update={"status": "failed", "error_message": str(error)}
        )

    def _apply_tags(self) -> None:
        if self._run is None:
            return
        try:
            self._run.tags = tuple(self._tags)
        except Exception as error:
            self._record_failure("WandB tag update failed", error)

    @staticmethod
    def _build_tags_from_config(config: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for key in ["dataset_id", "gnn_architecture", "llm_provider", "model_id"]:
            value = config.get(key)
            if value:
                tags.append(str(value))

        configs = config.get("configs", {})
        model_config = configs.get("model", {}) if isinstance(configs, dict) else {}
        if isinstance(model_config, dict):
            model_architecture = model_config.get("gnn_architecture")
            if model_architecture:
                tags.append(str(model_architecture))
            embedding_model = model_config.get("embedding_model")
            if embedding_model:
                tags.append(str(embedding_model))
            else:
                # Historical payload compatibility until old runs age out.
                for key in [
                    "entity_embedding_model",
                    "question_embedding_model",
                    "relation_embedding_model",
                ]:
                    value = model_config.get(key)
                    if value:
                        tags.append(str(value))
            architecture_context = model_config.get("gnn_architecture_context", {})
            if isinstance(architecture_context, dict):
                encoder_model_id = architecture_context.get("encoder_model_id")
                if encoder_model_id:
                    tags.append(str(encoder_model_id).rsplit("/", 1)[-1])
            training = model_config.get("training", {})
            if not isinstance(training, dict):
                training = {}
            trained_instances = training.get("trained_instances")
            if isinstance(trained_instances, dict):
                trained_instances = trained_instances.get("count")
            if trained_instances is None:
                trained_instances = model_config.get("trained_instances")
            if trained_instances is not None:
                tags.append(f"trained_instances:{trained_instances}")

        evidence = configs.get("evidence", {}) if isinstance(configs, dict) else {}
        if isinstance(evidence, dict):
            algorithm = evidence.get("algorithm")
            if algorithm:
                tags.append(str(algorithm))
            pcst = evidence.get("pcst", {})
            if algorithm == "pcst" and isinstance(pcst, dict):
                strategy = pcst.get("edge_cost_strategy")
                if strategy:
                    tags.append(f"pcst-{strategy}")
                embedding_model = pcst.get("semantic_embedding_model")
                if embedding_model:
                    tags.append(str(embedding_model))

        runs = config.get("runs", {})
        if isinstance(runs, dict):
            for stage in ["model", "evaluation", "inference"]:
                reference = runs.get(stage)
                if not isinstance(reference, dict):
                    continue
                run_number = reference.get("number")
                if run_number is not None:
                    tags.append(f"{stage}_run_number:{run_number}")
        return tags

    @classmethod
    def _deep_merge(
        cls,
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge stage config without dropping upstream sections."""
        merged = dict(existing)
        for key, value in incoming.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge(current, value)
            else:
                merged[key] = value
        return merged
