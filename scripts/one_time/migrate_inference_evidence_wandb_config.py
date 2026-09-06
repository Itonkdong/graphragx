"""One-time W&B config repair for completed inference runs.

Historical inference runs stored evidence construction under
``configs.inference.evidence_subgraph``. The stage-aware shape is
``configs.evidence``. This script moves that payload in the existing W&B run
and removes the obsolete nested key. Local inference artifacts are read-only.

The script is a dry run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.constants import LLM_INFERENCE_CONFIG_FILENAME  # noqa: E402


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _local_evidence_payload(config: dict[str, Any]) -> dict[str, Any]:
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("Local inference config has no inference object.")
    evidence = inference.get("evidence_subgraph")
    if not isinstance(evidence, dict):
        raise ValueError(
            "Local inference config has no inference.evidence_subgraph object."
        )
    return copy.deepcopy(evidence)


def _wandb_lineage(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    tracking = config.get("wandb")
    if not isinstance(tracking, dict) or not tracking.get("run_id"):
        raise ValueError(
            f"Inference config has no resumable W&B run ID: {config_path}"
        )
    return tracking


def _wandb_location_from_url(run_url: object) -> tuple[str | None, str | None]:
    if not isinstance(run_url, str) or not run_url:
        return None, None
    parts = [part for part in urlparse(run_url).path.split("/") if part]
    if len(parts) >= 4 and parts[-2] == "runs":
        return parts[-4], parts[-3]
    return None, None


def _resolve_wandb_path(
    tracking: dict[str, Any],
    *,
    entity_override: str | None = None,
    project_override: str | None = None,
) -> str:
    url_entity, url_project = _wandb_location_from_url(tracking.get("run_url"))
    entity = (
        entity_override
        or tracking.get("entity")
        or os.getenv("WANDB_ENTITY")
        or url_entity
    )
    project = (
        project_override
        or tracking.get("project")
        or os.getenv("WANDB_PROJECT")
        or url_project
    )
    run_id = tracking.get("run_id")
    if not entity or not project or not run_id:
        raise ValueError(
            "W&B lineage must provide entity, project, and run_id (or entity/project "
            "must be available through WANDB_ENTITY and WANDB_PROJECT)."
        )
    return f"{entity}/{project}/{run_id}"


def migrate_remote_config(
    remote_config: dict[str, Any],
    *,
    local_evidence: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Return a repaired W&B config and whether it differs from the input."""
    repaired = copy.deepcopy(remote_config)
    configs = repaired.get("configs")
    if not isinstance(configs, dict):
        raise ValueError("W&B run config has no configs object.")
    inference = configs.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("W&B run config has no configs.inference object.")

    nested_evidence = inference.get("evidence_subgraph")
    evidence = nested_evidence if isinstance(nested_evidence, dict) else local_evidence
    existing_evidence = configs.get("evidence")
    if isinstance(existing_evidence, dict) and existing_evidence != evidence:
        raise ValueError(
            "W&B configs.evidence conflicts with the historical inference evidence."
        )

    inference.pop("evidence_subgraph", None)
    configs["evidence"] = copy.deepcopy(evidence)
    return repaired, repaired != remote_config


def _resolve_run_directory(
    inference_root: Path,
    *,
    run_number: int | None = None,
    run_name: str | None = None,
) -> Path:
    if run_name is not None:
        exact = inference_root / run_name
        if exact.is_dir():
            return exact
        matches = [
            path
            for path in inference_root.iterdir()
            if path.is_dir() and path.name.endswith(f"_{run_name}")
        ]
    else:
        matches = [
            path
            for path in inference_root.iterdir()
            if path.is_dir() and path.name.startswith(f"{run_number}_")
        ]
    if len(matches) != 1:
        selector = f"name={run_name!r}" if run_name is not None else f"number={run_number}"
        raise ValueError(
            f"Expected exactly one inference run for {selector}; found {len(matches)}."
        )
    return matches[0]


def _selected_run_directories(
    *,
    inference_root: Path,
    run_numbers: list[int] | None,
    run_names: list[str] | None,
    all_nested: bool,
) -> list[Path]:
    if not inference_root.exists():
        raise ValueError(f"Inference root does not exist: {inference_root}")
    if all_nested:
        selected: list[Path] = []
        for directory in sorted(inference_root.iterdir(), key=lambda path: path.name):
            config_path = directory / LLM_INFERENCE_CONFIG_FILENAME
            if not directory.is_dir() or not config_path.exists():
                continue
            config = _load_object(config_path)
            inference = config.get("inference")
            if isinstance(inference, dict) and isinstance(
                inference.get("evidence_subgraph"), dict
            ):
                selected.append(directory)
        return selected
    if run_numbers:
        return [
            _resolve_run_directory(inference_root, run_number=run_number)
            for run_number in run_numbers
        ]
    return [
        _resolve_run_directory(inference_root, run_name=run_name)
        for run_name in run_names or []
    ]


def migrate_runs(
    run_directories: list[Path],
    *,
    apply: bool,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> int:
    failures = 0
    if not run_directories:
        print("No matching inference runs contain historical evidence config.")
        return 0

    wandb = importlib.import_module("wandb") if apply else None
    api = wandb.Api() if wandb is not None else None
    for run_directory in run_directories:
        config_path = run_directory / LLM_INFERENCE_CONFIG_FILENAME
        if not config_path.exists():
            print(
                f"SKIPPED {run_directory.name}: no completed "
                f"{LLM_INFERENCE_CONFIG_FILENAME} artifact"
            )
            continue
        try:
            local_config = _load_object(config_path)
            evidence = _local_evidence_payload(local_config)
            tracking = _wandb_lineage(local_config, config_path)
            wandb_path = _resolve_wandb_path(
                tracking,
                entity_override=wandb_entity,
                project_override=wandb_project,
            )
            print(
                f"{'APPLY' if apply else 'DRY RUN'} {run_directory.name}: "
                f"wandb={wandb_path} algorithm={evidence.get('algorithm')} "
                "move=configs.inference.evidence_subgraph->configs.evidence"
            )
            if not apply:
                continue

            remote_run = api.run(wandb_path)
            remote_config = remote_run.config
            if not isinstance(remote_config, dict):
                raise ValueError("W&B returned a non-object run config.")
            repaired, changed = migrate_remote_config(
                remote_config,
                local_evidence=evidence,
            )
            if not changed:
                print(f"ALREADY FIXED {run_directory.name}")
                continue
            remote_run.config.clear()
            remote_run.config.update(repaired)
            remote_run.update()
            print(f"UPDATED {run_directory.name}: {getattr(remote_run, 'url', wandb_path)}")
        except Exception as error:
            failures += 1
            print(f"FAILED {run_directory.name}: {error}", file=sys.stderr)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration of historical inference evidence configuration in "
            "W&B. Without --apply, no remote state is changed."
        )
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "webqsp" / "inference",
    )
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument(
        "--inference-run-number",
        type=int,
        action="append",
        help="Inference run number to patch; repeat for multiple runs.",
    )
    selectors.add_argument(
        "--inference-run-name",
        action="append",
        help="Inference run folder name or label to patch; repeat for multiple runs.",
    )
    selectors.add_argument(
        "--all-with-nested-evidence",
        action="store_true",
        help="Patch every local inference run containing the historical nested block.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update existing W&B runs. Omit for a dry run.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity override when old lineage omitted it.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional W&B project override when old lineage omitted it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_directories = _selected_run_directories(
            inference_root=args.inference_root,
            run_numbers=args.inference_run_number,
            run_names=args.inference_run_name,
            all_nested=args.all_with_nested_evidence,
        )
    except Exception as error:
        print(f"Could not select inference runs: {error}", file=sys.stderr)
        return 2
    return migrate_runs(
        run_directories,
        apply=args.apply,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    raise SystemExit(main())
