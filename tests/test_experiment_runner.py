"""Tests for the lightweight TOML experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import main

from scripts.experiments.run_experiments import (
    ExperimentManifestError,
    load_manifest,
    resolve_execution_order,
    run_experiments,
)


def test_experiment_zero_contains_complete_retriever_matrix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(
        project_root / "experiments" / "experiment_0_gnn_architectures.toml"
    )
    parser = main.build_parser()
    required_option_flags = {
        "graphsage": {"--gnn-layers", "--gnn-hidden-dim", "--node-classifier", "--dropout"},
        "aa-graphsage": {
            "--gnn-layers",
            "--gnn-hidden-dim",
            "--node-classifier",
            "--dropout",
            "--use-edge-mlp",
            "--use-reverse-edges",
            "--question-aware-classifier",
            "--add-layer-normalization",
            "--edge-mlp-hidden-dim",
        },
        "rgcn": {"--gnn-layers", "--gnn-hidden-dim", "--dropout", "--num-bases"},
        "hgt": {
            "--gnn-layers",
            "--gnn-hidden-dim",
            "--dropout",
            "--attention-heads",
        },
        "rearev": {
            "--gnn-hidden-dim",
            "--dropout",
            "--num-instructions",
            "--reasoning-steps",
            "--adaptive-iterations",
        },
        "nbfnet": {"--gnn-layers", "--gnn-hidden-dim"},
    }

    observed: set[tuple[str, int]] = set()
    hgt_layers: set[int] = set()
    for run in manifest.runs:
        parsed = parser.parse_args([*manifest.default_args, *run.args])
        assert parsed.run_mode == "retriever-only"
        assert parsed.training_epochs == 10
        assert parsed.training_batch_size == 1
        assert parsed.answer_threshold == 0.7
        assert parsed.candidate_top_k == 10
        assert parsed.candidate_limit == 15
        assert required_option_flags[parsed.gnn_architecture].issubset(run.args)
        assert run.run_id.startswith("exp0_")
        assert parsed.training_run_name.startswith("exp0_")
        assert parsed.evaluation_run_name.startswith("exp0_")
        observed.add((parsed.gnn_architecture, parsed.random_seed))
        if parsed.gnn_architecture == "hgt":
            hgt_layers.add(parsed.gnn_layer_count)

    assert observed == {
        (architecture, seed)
        for architecture in (
            "graphsage",
            "aa-graphsage",
            "rgcn",
            "hgt",
            "rearev",
            "nbfnet",
        )
        for seed in (42, 1337, 2026)
    }
    assert hgt_layers == {3}


def test_experiment_one_contains_bounded_nbfnet_evidence_matrix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(
        project_root / "experiments" / "experiment_1_evidence_subgraphs.toml"
    )
    parser = main.build_parser()
    expected_retrievers = {
        "exp0_nbfnet_seed42_retriever",
        "exp0_nbfnet_seed1337_retriever",
        "exp0_nbfnet_seed2026_retriever",
    }
    lambdas_by_mode: dict[str, set[float]] = {
        "constant": set(),
        "semantic": set(),
    }
    counts_by_retriever: dict[str, int] = {
        retriever: 0 for retriever in expected_retrievers
    }
    shortest_path_count = 0

    assert len(manifest.runs) == 27
    for run in manifest.runs:
        parsed = parser.parse_args([*manifest.default_args, *run.args])
        assert parsed.run_mode == "evidence-only"
        assert parsed.retriever_run_name in expected_retrievers
        assert parsed.main_llm_model is None
        assert parsed.llm_provider is None
        assert "--inference-run-name" not in run.args
        assert run.run_id.startswith("exp1_")
        assert parsed.evidence_run_name == run.run_id
        counts_by_retriever[parsed.retriever_run_name] += 1
        if parsed.subgraph_algorithm == "shortest_path":
            shortest_path_count += 1
            assert parsed.pcst_edge_cost_strategy is None
            assert parsed.pcst_edge_cost is None
            continue
        assert parsed.subgraph_algorithm == "pcst"
        assert parsed.pcst_edge_cost_strategy in lambdas_by_mode
        lambdas_by_mode[parsed.pcst_edge_cost_strategy].add(
            parsed.pcst_edge_cost
        )

    assert shortest_path_count == 3
    assert counts_by_retriever == {
        retriever: 9 for retriever in expected_retrievers
    }
    assert lambdas_by_mode == {
        "constant": {0.01, 0.25, 0.5, 1.0},
        "semantic": {0.01, 0.25, 0.5, 1.0},
    }


def test_experiment_two_contains_complete_end_to_end_matrix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(
        project_root / "experiments" / "experiment_2_end_to_end.toml"
    )
    parser = main.build_parser()
    expected_retrievers = {
        "exp0_nbfnet_seed42_retriever",
        "exp0_nbfnet_seed1337_retriever",
        "exp0_nbfnet_seed2026_retriever",
    }
    expected_models = {"deepseek-v4-flash", "gpt-5.6-luna"}
    strategies_by_model_and_retriever: dict[
        tuple[str, str], set[tuple[str, str | None, float | None]]
    ] = {
        (model, retriever): set()
        for model in expected_models
        for retriever in expected_retrievers
    }

    assert len(manifest.runs) == 30
    for run in manifest.runs:
        parsed = parser.parse_args([*manifest.default_args, *run.args])
        assert parsed.run_mode == "inference-only"
        assert parsed.retriever_run_name in expected_retrievers
        assert parsed.main_llm_model in expected_models
        assert parsed.reasoning_effort == "none"
        if parsed.main_llm_model == "deepseek-v4-flash":
            assert parsed.llm_provider == "deepseek"
            assert parsed.llm_inference_batch_size == 500
            assert parsed.llm_inference_parallel_calls == 500
        else:
            assert parsed.llm_provider == "openai"
            assert parsed.llm_inference_batch_size == 20
            assert parsed.llm_inference_parallel_calls == 20
        assert parsed.generate_explanation is False
        assert run.run_id.startswith("exp2_")
        assert parsed.inference_run_name == run.run_id
        strategies_by_model_and_retriever[
            (parsed.main_llm_model, parsed.retriever_run_name)
        ].add(
            (
                parsed.subgraph_algorithm,
                parsed.pcst_edge_cost_strategy,
                parsed.pcst_edge_cost,
            )
        )

    expected_strategies = {
        ("shortest_path", None, None),
        ("pcst", "constant", 0.01),
        ("pcst", "constant", 1.0),
        ("pcst", "semantic", 0.01),
        ("pcst", "semantic", 1.0),
    }
    assert strategies_by_model_and_retriever == {
        (model, retriever): expected_strategies
        for model in expected_models
        for retriever in expected_retrievers
    }


def test_experiment_directory_has_only_three_official_manifests() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiment_files = {
        path.name for path in (project_root / "experiments").glob("experiment_*.toml")
    }
    assert experiment_files == {
        "experiment_0_gnn_architectures.toml",
        "experiment_1_evidence_subgraphs.toml",
        "experiment_2_end_to_end.toml",
    }


def test_probe_manifests_and_run_names_use_probe_prefix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    probe_paths = sorted((project_root / "experiments").glob("probe_*.toml"))
    assert {path.name for path in probe_paths} == {
        "probe_end_to_end_recovery.toml",
        "probe_pcst_low_cost.toml",
    }
    parser = main.build_parser()
    for path in probe_paths:
        manifest = load_manifest(path)
        for run in manifest.runs:
            parsed = parser.parse_args([*manifest.default_args, *run.args])
            assert run.run_id.startswith("probe_")
            output_name = parsed.evidence_run_name or parsed.inference_run_name
            assert output_name.startswith("probe_")


def _write_manifest(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _write_fake_main(project_root: Path) -> None:
    (project_root / "main.py").write_text(
        """\
import json
import pathlib
import sys

path = pathlib.Path("invocations.jsonl")
with path.open("a", encoding="utf-8") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
print("fake pipeline", *sys.argv[1:])
raise SystemExit(7 if "--fail" in sys.argv else 0)
""",
        encoding="utf-8",
    )


def test_manifest_order_includes_dependencies_and_defaults(tmp_path: Path) -> None:
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "experiments.toml",
            """
version = 1
[defaults]
args = ["--default", "--seed", "9"]
[[runs]]
id = "train"
args = ["--retriever-only"]
[[runs]]
id = "infer"
after = ["train"]
args = ["--inference-only"]
""",
        )
    )

    assert manifest.default_args == ("--default", "--seed", "9")
    assert [run.run_id for run in resolve_execution_order(manifest, ["infer"])] == [
        "train",
        "infer",
    ]


def test_manifest_rejects_cycles_and_unknown_dependencies(tmp_path: Path) -> None:
    with pytest.raises(ExperimentManifestError, match="unknown dependencies"):
        load_manifest(
            _write_manifest(
                tmp_path / "unknown.toml",
                'version = 1\n[[runs]]\nid = "a"\nafter = ["missing"]\nargs = []\n',
            )
        )

    with pytest.raises(ExperimentManifestError, match="Dependency cycle"):
        load_manifest(
            _write_manifest(
                tmp_path / "cycle.toml",
                """
version = 1
[[runs]]
id = "a"
after = ["b"]
args = []
[[runs]]
id = "b"
after = ["a"]
args = []
""",
            )
        )


def test_runner_executes_in_order_logs_and_skips_successful_resume(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "run.toml",
            """
version = 1
[defaults]
args = ["--seed", "42"]
[[runs]]
id = "first"
args = ["--one"]
[[runs]]
id = "second"
after = ["first"]
args = ["--two"]
""",
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        state_root=state_root,
    ) == 0
    assert run_experiments(
        manifest,
        project_root=project_root,
        state_root=state_root,
    ) == 0

    invocations = [
        json.loads(line)
        for line in (project_root / "invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations == [
        ["--seed", "42", "--one"],
        ["--seed", "42", "--two"],
    ]
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["runs"]["first"]["status"] == "succeeded"
    assert state["runs"]["second"]["status"] == "succeeded"
    assert "fake pipeline --seed 42 --one" in (
        state_root / "logs" / "first.log"
    ).read_text(encoding="utf-8")


def test_continue_on_error_runs_independent_work_and_blocks_dependents(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "failures.toml",
            """
version = 1
[[runs]]
id = "failure"
args = ["--fail"]
[[runs]]
id = "dependent"
after = ["failure"]
args = ["--dependent"]
[[runs]]
id = "independent"
args = ["--independent"]
""",
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        continue_on_error=True,
        state_root=state_root,
    ) == 1

    invocations = [
        json.loads(line)
        for line in (project_root / "invocations.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert invocations == [["--fail"], ["--independent"]]
    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    assert state["runs"]["failure"]["status"] == "failed"
    assert state["runs"]["dependent"]["status"] == "blocked"
    assert state["runs"]["independent"]["status"] == "succeeded"


def test_dry_run_does_not_create_state_or_launch_commands(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _write_fake_main(project_root)
    manifest = load_manifest(
        _write_manifest(
            tmp_path / "dry.toml",
            'version = 1\n[[runs]]\nid = "only"\nargs = ["--one"]\n',
        )
    )
    state_root = tmp_path / "state"

    assert run_experiments(
        manifest,
        project_root=project_root,
        dry_run=True,
        state_root=state_root,
    ) == 0
    assert not state_root.exists()
    assert not (project_root / "invocations.jsonl").exists()
