from pathlib import Path
from types import SimpleNamespace

import pytest

import pcwannier.cli as cli_module
from pcwannier.cli import main, parse_args
from pcwannier.maxwell import MaxwellProblem


def test_cli_orchestrates_calculation_and_interpolation_without_dataset(tmp_path, monkeypatch):
    config = _config(tmp_path)
    bundle = object()
    result = object()
    calls = {}

    monkeypatch.setattr(cli_module, "load_config", lambda path: config)
    monkeypatch.setattr(cli_module, "load_input", lambda loaded: bundle)

    def run_calculation(loaded_bundle, *, threads, backend):
        calls["run"] = (loaded_bundle, threads, backend)
        return result

    def write_outputs(actual_result, actual_config, out_dir):
        calls["outputs"] = (actual_result, actual_config, out_dir)

    def write_interpolation(actual_result, points, wannier, metric, *, out_dir):
        calls["interpolation"] = (actual_result, points, wannier, metric, out_dir)

    monkeypatch.setattr(cli_module, "run_calculation", run_calculation)
    monkeypatch.setattr(cli_module, "write_outputs", write_outputs)
    monkeypatch.setattr(cli_module, "write_interpolation_outputs", write_interpolation)

    out = tmp_path / "out"
    assert main(
        [
            "-i",
            "example-incar",
            "--out",
            str(out),
            "-t",
            "3",
            "--backend",
            "auto",
            "--interp",
            "points.txt",
            "--interp-wannier",
            "wannier.txt",
            "--interp-metric",
            "metric.txt",
        ]
    ) == 0

    assert calls["run"] == (bundle, 3, "auto")
    assert calls["outputs"] == (result, config, out)
    assert calls["interpolation"] == (
        result,
        "points.txt",
        "wannier.txt",
        "metric.txt",
        out,
    )
    log_text = (out / "log.txt").read_text(encoding="utf-8")
    assert "=========  PCWannier v" in log_text
    assert "total runtime:" in log_text
    assert "memory usage:" in log_text
    assert "pcwannier.compute.runner" not in log_text


def test_cli_cache_paths_and_base_mode_are_dataset_independent(tmp_path, monkeypatch):
    cache_config = _config(tmp_path)
    monkeypatch.setattr(cli_module, "load_config", lambda path: cache_config)
    monkeypatch.setattr(cli_module, "load_input", lambda config: object())
    monkeypatch.setattr(cli_module, "run_calculation", lambda bundle, **kwargs: object())
    monkeypatch.setattr(cli_module, "write_outputs", lambda result, config, out_dir: None)

    cache_out = tmp_path / "cache-out"
    assert main(["-i", "example-incar", "--out", str(cache_out), "--cache"]) == 0
    assert cache_config.use_cached_data == ["U", "V", "M", "S", "A", "D"]
    for attr in ("M_file", "A_file", "V_file", "U_file", "S_file", "D_file"):
        assert Path(getattr(cache_config, attr)).parent == cache_out

    constrained_config = _config(tmp_path)
    constrained_config.symmetry_constrained = True
    monkeypatch.setattr(cli_module, "load_config", lambda path: constrained_config)
    assert main(["-i", "example-incar", "--cache"]) == 0
    assert constrained_config.use_cached_data == ["V", "M", "S", "A", "D"]

    base_config = _config(tmp_path)
    mesh = object()
    calls = {}
    monkeypatch.setattr(cli_module, "load_config", lambda path: base_config)

    def fail_load_input(_config):
        raise AssertionError("--base should not load field or eigenvalue data")

    monkeypatch.setattr(cli_module, "load_input", fail_load_input)
    monkeypatch.setattr(cli_module, "load_comsol_mesh", lambda path: mesh)
    monkeypatch.setattr(
        cli_module,
        "write_base_figures",
        lambda config, actual_mesh, out_dir: calls.setdefault("base", (config, actual_mesh, out_dir)),
    )
    base_out = tmp_path / "base-out"
    assert main(["-i", "example-incar", "--out", str(base_out), "--base"]) == 0
    assert calls["base"] == (base_config, mesh, base_out)


def test_relative_out_is_resolved_once_for_cache_input_and_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    calls = {}
    monkeypatch.setattr(cli_module, "load_config", lambda path: config)
    monkeypatch.setattr(cli_module, "load_input", lambda actual_config: object())
    monkeypatch.setattr(cli_module, "run_calculation", lambda bundle, **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "write_outputs",
        lambda result, actual_config, out_dir: calls.setdefault("out_dir", out_dir),
    )

    assert main(["-i", "example-incar", "--out", "relative-out", "--cache"]) == 0

    expected = (tmp_path / "relative-out").resolve()
    assert calls["out_dir"] == expected
    for attr in ("M_file", "A_file", "V_file", "U_file", "S_file", "D_file"):
        cache_path = Path(getattr(config, attr))
        assert cache_path.is_absolute()
        assert cache_path.parent == expected


def test_interpolation_outputs_require_points_file(tmp_path):
    with pytest.raises(ValueError, match="--interp is required"):
        main(
            [
                "-i",
                "unused",
                "--out",
                str(tmp_path / "invalid-interpolation"),
                "--interp-wannier",
                "wannier.txt",
            ]
        )


def test_fatband_argument_removed():
    with pytest.raises(SystemExit):
        parse_args(["-i", "unused", "--fatband"])


def test_group_mode_prints_character_table_without_incar_or_log(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["--group", "c4v"]) == 0

    output = capsys.readouterr().out
    assert "Finite group: C4v" in output
    assert "Canonical elements: E, C4, C2, C4_inv" in output
    assert "Conjugacy classes:" in output
    assert "Character table:" in output
    assert "Available site_irrep names: A1, A2, B1, B2, E" in output
    assert not (tmp_path / "log.txt").exists()


def test_input_and_group_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["-i", "incar", "--group", "c4v"])


def _config(base_dir: Path):
    config = SimpleNamespace(
        name="synthetic",
        dataset_type="comsol",
        maxwell_problem=MaxwellProblem.for_components("Ez"),
        metric_file="eps.txt",
        kdim=2,
        k_points=[[0.0], [0.0]],
        band_calc_num=1,
        compute_backend="python",
        symmetry_context=None,
        symmetry_constrained=False,
        symmetry_output_basis="strict",
        use_cached_data=[],
        mesh_file="mesh.mphtxt",
        M_file="M.txt",
        A_file="A.txt",
        V_file="V.txt",
        U_file="U.txt",
        S_file="S.txt",
        D_file="D.txt",
        base_dir=base_dir,
    )
    config.input_path = lambda value: base_dir / str(value)
    return config
