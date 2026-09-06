"""Run a small, resumable sequence of existing graphragX CLI commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class ExperimentManifestError(ValueError):
    """Raised when an experiment manifest is invalid."""


@dataclass(frozen=True)
class ExperimentRun:
    """One existing ``main.py`` invocation from a manifest."""

    run_id: str
    args: tuple[str, ...]
    after: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class ExperimentManifest:
    """Validated experiment manifest."""

    path: Path
    default_args: tuple[str, ...]
    runs: tuple[ExperimentRun, ...]

    @property
    def run_map(self) -> dict[str, ExperimentRun]:
        return {run.run_id: run for run in self.runs}


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ExperimentManifestError(f"{field_name} must be an array of strings.")
    if any("\x00" in item for item in value):
        raise ExperimentManifestError(f"{field_name} cannot contain null bytes.")
    return tuple(value)


def load_manifest(path: Path) -> ExperimentManifest:
    """Load and validate a version-1 TOML experiment manifest."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExperimentManifestError(f"Could not load manifest {path}: {error}") from error

    if payload.get("version") != 1:
        raise ExperimentManifestError("Manifest version must be 1.")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ExperimentManifestError("defaults must be a TOML table.")
    default_args = _string_list(defaults.get("args"), "defaults.args")

    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ExperimentManifestError("Manifest must define at least one [[runs]] table.")

    runs: list[ExperimentRun] = []
    seen_ids: set[str] = set()
    for index, raw_run in enumerate(raw_runs, start=1):
        if not isinstance(raw_run, dict):
            raise ExperimentManifestError(f"runs[{index}] must be a TOML table.")
        run_id = raw_run.get("id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise ExperimentManifestError(
                f"runs[{index}].id must contain only letters, digits, '.', '_', or '-'."
            )
        if run_id in seen_ids:
            raise ExperimentManifestError(f"Duplicate run id: {run_id}.")
        seen_ids.add(run_id)
        enabled = raw_run.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ExperimentManifestError(f"runs[{index}].enabled must be a boolean.")
        runs.append(
            ExperimentRun(
                run_id=run_id,
                args=_string_list(raw_run.get("args"), f"runs[{index}].args"),
                after=_string_list(raw_run.get("after"), f"runs[{index}].after"),
                enabled=enabled,
            )
        )

    run_ids = {run.run_id for run in runs}
    for run in runs:
        unknown = sorted(set(run.after) - run_ids)
        if unknown:
            raise ExperimentManifestError(
                f"Run {run.run_id} has unknown dependencies: {', '.join(unknown)}."
            )
        if run.run_id in run.after:
            raise ExperimentManifestError(f"Run {run.run_id} cannot depend on itself.")

    manifest = ExperimentManifest(
        path=path.resolve(),
        default_args=default_args,
        runs=tuple(runs),
    )
    # Validate all enabled dependencies and cycles immediately.
    resolve_execution_order(manifest)
    return manifest


def resolve_execution_order(
    manifest: ExperimentManifest,
    selected_ids: Sequence[str] | None = None,
) -> list[ExperimentRun]:
    """Return selected runs and dependencies in stable topological order."""
    run_map = manifest.run_map
    if selected_ids:
        unknown = sorted(set(selected_ids) - set(run_map))
        if unknown:
            raise ExperimentManifestError(
                f"Unknown selected run ids: {', '.join(unknown)}."
            )
        disabled = sorted(run_id for run_id in selected_ids if not run_map[run_id].enabled)
        if disabled:
            raise ExperimentManifestError(
                f"Selected runs are disabled: {', '.join(disabled)}."
            )
        roots = list(dict.fromkeys(selected_ids))
    else:
        roots = [run.run_id for run in manifest.runs if run.enabled]

    ordered: list[ExperimentRun] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(run_id: str) -> None:
        if run_id in visited:
            return
        if run_id in visiting:
            raise ExperimentManifestError(f"Dependency cycle detected at run {run_id}.")
        run = run_map[run_id]
        if not run.enabled:
            raise ExperimentManifestError(
                f"Enabled run depends on disabled run {run_id}."
            )
        visiting.add(run_id)
        for dependency_id in run.after:
            visit(dependency_id)
        visiting.remove(run_id)
        visited.add(run_id)
        ordered.append(run)

    for root in roots:
        visit(root)
    return ordered


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _command_hash(command: Sequence[str]) -> str:
    serialized = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "runs": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentManifestError(f"Could not load runner state {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ExperimentManifestError(f"Unsupported runner state in {path}.")
    if not isinstance(payload.get("runs"), dict):
        raise ExperimentManifestError(f"Runner state {path} has invalid runs metadata.")
    return payload


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _execute(command: list[str], log_path: Path, project_root: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        header = f"$ {shlex.join(command)}\n"
        print(header, end="", flush=True)
        log_file.write(header)
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 130


def run_experiments(
    manifest: ExperimentManifest,
    *,
    project_root: Path,
    selected_ids: Sequence[str] | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
    force: bool = False,
    state_root: Path | None = None,
) -> int:
    """Execute a manifest and return a shell-compatible exit status."""
    ordered = resolve_execution_order(manifest, selected_ids)
    run_root = state_root or project_root / ".experiment-runs" / manifest.path.stem
    state_path = run_root / "state.json"
    log_root = run_root / "logs"
    state = _load_state(state_path)
    run_state: dict[str, Any] = state["runs"]
    completed_this_invocation: dict[str, bool] = {}
    failures = 0

    for run in ordered:
        command = [
            sys.executable,
            str(project_root / "main.py"),
            *manifest.default_args,
            *run.args,
        ]
        command_hash = _command_hash(command)
        previous = run_state.get(run.run_id, {})
        unchanged_success = (
            isinstance(previous, dict)
            and previous.get("status") == "succeeded"
            and previous.get("command_hash") == command_hash
        )
        dependency_failed = any(
            completed_this_invocation.get(dependency_id) is False
            for dependency_id in run.after
        )

        if dependency_failed:
            print(f"BLOCKED {run.run_id}: a dependency failed.")
            completed_this_invocation[run.run_id] = False
            failures += 1
            if not dry_run:
                run_state[run.run_id] = {
                    "status": "blocked",
                    "command": command,
                    "command_hash": command_hash,
                    "updated_at": _utc_now(),
                }
                _save_state(state_path, state)
            continue

        if unchanged_success and not force:
            print(f"SKIP {run.run_id}: already succeeded with the same command.")
            completed_this_invocation[run.run_id] = True
            continue

        print(f"{'WOULD RUN' if dry_run else 'RUN'} {run.run_id}: {shlex.join(command)}")
        if dry_run:
            completed_this_invocation[run.run_id] = True
            continue

        log_path = log_root / f"{run.run_id}.log"
        started_at = _utc_now()
        run_state[run.run_id] = {
            "status": "running",
            "command": command,
            "command_hash": command_hash,
            "log_path": str(log_path),
            "started_at": started_at,
            "updated_at": started_at,
        }
        _save_state(state_path, state)

        exit_code = _execute(command, log_path, project_root)
        status = "succeeded" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed"
        finished_at = _utc_now()
        run_state[run.run_id].update(
            {
                "status": status,
                "exit_code": exit_code,
                "finished_at": finished_at,
                "updated_at": finished_at,
            }
        )
        _save_state(state_path, state)
        completed_this_invocation[run.run_id] = exit_code == 0
        print(f"{status.upper()} {run.run_id} (exit_code={exit_code})")

        if exit_code != 0:
            failures += 1
            if exit_code == 130 or not continue_on_error:
                return exit_code or 1

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable TOML sequence of graphragX CLI experiments."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--run",
        dest="selected_ids",
        action="append",
        help="Run only this id and its dependencies. Repeat to select multiple ids.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun successful selected commands.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    try:
        manifest = load_manifest(args.manifest)
        return run_experiments(
            manifest,
            project_root=project_root,
            selected_ids=args.selected_ids,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            force=args.force,
        )
    except ExperimentManifestError as error:
        print(f"Experiment manifest error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
