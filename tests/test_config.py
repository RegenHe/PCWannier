import logging

import numpy as np
import pytest

from pcwannier import (
    EnergyWindow,
    FieldComponents,
    FieldKind,
    MaterialKind,
    PrimaryField,
    load_config,
)
from pcwannier.config import IncarConfig, evaluate_math_expression, preprocess_config


def test_load_incar_defaults_and_preprocess_without_external_data(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar(), encoding="utf-8")

    cfg = load_config(incar)

    assert cfg.dataset_type == "comsol"
    assert cfg.hermitian is True
    assert cfg.kdim == 2
    assert [len(axis) for axis in cfg.k_points] == [2, 2]
    assert np.array_equal(cfg.band_window, np.arange(0, 3))
    assert cfg.band_calc_num == 3
    assert len(cfg.composition_of_b) == 4
    assert cfg.b_vectors.shape == (4, 2)
    assert cfg.wb.shape == (4,)
    assert len(cfg.projections) == 1
    assert len(cfg.projections[0]["states"]) == 3
    assert cfg.compute_backend == "python"
    assert cfg.integration_mode == "nodal"
    assert cfg.P_file == "./P.txt"
    assert cfg.wannier_subspace == "T"
    assert cfg.invert_longitudinal_energies is False
    assert cfg.projector_preserving_band_interpolation is False
    assert cfg.symmetry_report_file == "./sym.txt"
    assert cfg.ebr_subspace_dimension is None
    assert cfg.ebr_subspace_fixed_bands is None
    assert cfg.ebr_max_auxiliary_bands == 6
    assert cfg.ebr_max_states == 1_000_000
    assert cfg.ebr_report_file == "./ebr.txt"
    assert cfg.ebr_data_file == "./ebr.json"
    assert cfg.wannier_file is False
    assert cfg.field_components == FieldComponents.EZ.value
    assert cfg.maxwell_problem.primary_field == PrimaryField.ELECTRIC
    assert cfg.maxwell_problem.metric_material == MaterialKind.EPSILON
    assert cfg.maxwell_problem.curl_material == MaterialKind.MU
    assert cfg.symmetry_constrained is True
    assert cfg.output_basis == "fem"
    assert cfg.representation_character_tolerance == pytest.approx(1.0e-2)
    assert cfg.symmetry_context is not None
    assert cfg.symmetry_context.model.symmetry_gauge.enabled
    assert cfg.symmetry_context.model.bloch_convention.sign == -1
    assert cfg.symmetry_context.model.bloch_convention.name == "comsol"
    assert cfg.symmetry_context.model.boundary_tolerance == pytest.approx(1.0e-6)


def test_ebr_subspace_dimension_two_and_fixed_band_slice(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        _minimal_symmetry_incar()
        + "\nebr_subspace_dimension = 2\n"
        + "ebr_subspace_fixed_bands = 0:2\n",
        encoding="utf-8",
    )

    cfg = load_config(incar)

    assert cfg.ebr_subspace_dimension == 2
    assert np.array_equal(cfg.ebr_subspace_fixed_bands, [0, 1])


def test_ebr_fixed_bands_require_subspace_dimension():
    cfg = IncarConfig(
        lattice_const=1.0,
        real_lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
        ebr_subspace_fixed_bands=np.asarray([0, 1]),
    )

    with pytest.raises(ValueError, match="requires ebr_subspace_dimension"):
        preprocess_config(cfg)


def test_bloch_symmetry_config_mode_does_not_require_wannier_inputs(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = 0:1:1, 0:1:1",
                "band_window = 0:3",
                "dataset_file = Ez.txt",
                "metric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "symmetry_file = p4mm",
                "symmetry_constrained = true",
                "wannier_targets",
                "ignored_A1; 0, 0; A1",
                "end",
                "representation_analysis",
                "Gamma; 0, 0; 0:3; ignored_A1",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(incar, mode="bloch_symmetry")

    assert cfg.projections is None
    assert cfg.composition_of_b is None
    assert cfg.extension is None
    assert cfg.symmetry_context is not None
    assert cfg.symmetry_context.model.targets == ()
    assert cfg.symmetry_context.model.symmetry_gauge is None
    point = cfg.symmetry_context.model.representation_analysis.points[0]
    assert point.target_names is None


def test_projector_preserving_band_interpolation_switch_parses_boolean(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        _minimal_symmetry_incar()
        + "\nprojector_preserving_band_interpolation = false\n",
        encoding="utf-8",
    )

    cfg = load_config(incar)

    assert cfg.projector_preserving_band_interpolation is False


def test_longitudinal_settings_are_ignored_for_transverse_subspace(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        _minimal_symmetry_incar()
        + "\ninvert_longitudinal_energies = true\n"
        + "longitudinal_band_window = 0:2\n"
        + "longitudinal_field_file = missing-HL.h5\n",
        encoding="utf-8",
    )

    config = load_config(incar)

    assert config.wannier_subspace == "T"
    assert config.invert_longitudinal_energies is False
    assert config.longitudinal_band_window is None
    assert config.longitudinal_field_file is False


def test_energy_window_parser(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 1",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = 0:1:1, 0:1:1",
                "composition_of_b = 1 0, 0 1",
                "band_window = 0.1, 0.9",
                "dataset_file = ./Ez.txt",
                "metric_file = ./eps.txt",
                "mesh_file = ./mesh.mphtxt",
                "E_file = ./E.txt",
                "compute_backend = auto",
                "disentangle_max_iter = 25",
                "disentangle_err_diff = 1e-7",
                "disentangle_projector_tolerance = 2e-7",
                "disentangle_mixing = 0.75",
                "output_basis = STRICT",
                "extension = 1, 1",
                "projections",
                "a; [0, 0]; 0; [1, 0, 5]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(incar)

    assert isinstance(cfg.band_window, EnergyWindow)
    assert cfg.band_window.emin == 0.1
    assert cfg.band_window.emax == 0.9
    assert cfg.compute_backend == "auto"
    assert cfg.symmetry_constrained is False
    assert cfg.disentangle_max_iter == 25
    assert cfg.disentangle_err_diff == pytest.approx(1e-7)
    assert cfg.disentangle_projector_tolerance == pytest.approx(2e-7)
    assert cfg.disentangle_mixing == pytest.approx(0.75)
    assert cfg.output_basis == "strict"


def test_invalid_output_basis_is_rejected():
    cfg = IncarConfig(
        lattice_const=1.0,
        real_lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
        output_basis="mixed",
    )

    with pytest.raises(ValueError, match="output_basis"):
        preprocess_config(cfg)


def test_math_expression_parser_is_limited():
    assert np.isclose(evaluate_math_expression("sqrt(4) + pi / pi"), 3.0)
    with pytest.raises(ValueError):
        evaluate_math_expression("__import__('os').system('echo nope')")


def test_symmetry_constrained_requires_symmetry_file(tmp_path):
    source = _minimal_symmetry_incar().replace("symmetry_file = p4mm.yaml", "symmetry_file = false")
    incar = tmp_path / "incar"
    incar.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="requires symmetry_file"):
        load_config(incar)


def test_rank_deficient_b_vectors_are_rejected(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        "\n".join(
            [
                "lattice_const = 1",
                "real_lattice_vectors = 1 0, 0 2",
                "reciprocal_lattice_vectors = 0 0, 0 0",
                "k_points = 0:0.5:1, 0:0.5:1",
                "composition_of_b = 1 0",
                "band_window = 0:1",
                "dataset_file = Ez.txt",
                "metric_file = eps.txt",
                "mesh_file = mesh.mphtxt",
                "E_file = E.txt",
                "extension = 1, 1",
                "projections",
                "a; [0, 0]; 0; [1, 0, 1]",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Add independent neighbor directions"):
        load_config(incar)


def test_removed_w_center_input_is_warned_and_ignored(tmp_path, caplog):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar() + "\nw_center = 0, 0\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_config(incar)

    assert "Unknown incar field 'w_center' is ignored" in caplog.text
    assert config.name == "Wannier"
    assert config.kdim == 2


@pytest.mark.parametrize(
    "suffix",
    [
        "unknown_option = 1",
        "dielectric_file = eps.txt",
        "representation_field_kind = scalar",
        "hopping_state = 0:3, 0:3",
    ],
)
def test_incar_warns_and_ignores_unknown_fields(tmp_path, suffix, caplog):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar() + "\n" + suffix + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        config = load_config(incar)

    assert "Unknown incar field" in caplog.text
    assert config.name == "Wannier"
    assert config.kdim == 2


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        ("lattice_const = 2", "Duplicate incar field"),
        ("this is not an assignment", "Malformed incar line"),
    ],
)
def test_incar_rejects_unknown_duplicate_and_malformed_fields(tmp_path, suffix, message):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar() + "\n" + suffix + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(incar)


@pytest.mark.parametrize("value", ["yes", "0", "tru", "enabled"])
def test_boolean_fields_accept_only_true_or_false(tmp_path, value):
    incar = tmp_path / "incar"
    source = _minimal_symmetry_incar().replace(
        "symmetry_constrained = true", f"symmetry_constrained = {value}"
    )
    incar.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        load_config(incar)


@pytest.mark.parametrize(
    ("components", "primary", "metric", "curl", "field_kind"),
    [
        (
            "eZ",
            PrimaryField.ELECTRIC,
            MaterialKind.EPSILON,
            MaterialKind.MU,
            FieldKind.ELECTRIC_Z,
        ),
        (
            "hZ",
            PrimaryField.MAGNETIC,
            MaterialKind.MU,
            MaterialKind.EPSILON,
            FieldKind.MAGNETIC_AXIAL_Z,
        ),
    ],
)
def test_field_components_select_maxwell_problem(
    tmp_path, components, primary, metric, curl, field_kind
):
    incar = tmp_path / "incar"
    source = _minimal_symmetry_incar().replace(
        "field_components = Ez", f"field_components = {components}"
    )
    source += "\nrepresentation_analysis\nGamma; 0.0, 0.0; 0:3\nend\n"
    incar.write_text(source, encoding="utf-8")

    cfg = load_config(incar)

    assert cfg.maxwell_problem.primary_field == primary
    assert cfg.maxwell_problem.metric_material == metric
    assert cfg.maxwell_problem.curl_material == curl
    assert (
        cfg.symmetry_context.model.representation_analysis.field_kind == field_kind
    )


def test_full_vector_requires_primary_field_and_analysis_scope(tmp_path):
    incar = tmp_path / "incar"
    source = _minimal_symmetry_incar()
    incar.write_text(
        source.replace("field_components = Ez", "field_components = full_vector"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires primary_field"):
        load_config(incar)

    incar.write_text(
        source.replace(
            "field_components = Ez",
            "field_components = full_vector\nprimary_field = magnetic",
        ),
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="full_vector"):
        load_config(incar)

    incar.write_text(
        source.replace("field_components = Ez", "field_components = Ex"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="field_components must be one of"):
        load_config(incar)


def test_metric_file_is_required(tmp_path):
    incar = tmp_path / "incar"
    incar.write_text(
        _minimal_symmetry_incar().replace("metric_file = eps.txt\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metric_file"):
        load_config(incar)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("real_lattice_vectors = 1 0, 0 1", "real_lattice_vectors = 1 0, 2 0", "invertible"),
        ("k_points = -0.5:0.5:0.5, -0.5:0.5:0.5", "k_points = 0:1:2, -0.5:0.5:0.5", "periodic duplicate"),
        ("band_window = 0:3", "band_window = false", "band_window must not be false"),
        ("extension = 1, 1", "extension = 1, 0", "positive integers"),
    ],
)
def test_central_config_validation_rejects_invalid_core_geometry(tmp_path, old, new, message):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar().replace(old, new), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(incar)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("max_iter = -1", "max_iter must be non-negative"),
        ("epsilon = 0", "epsilon must be finite and positive"),
        ("projection_rank_tolerance = 1", "projection_rank_tolerance"),
        ("representation_character_tolerance = 0", "representation_character_tolerance"),
        ("neighbor = 0 0", r"neighbor\[0\] must be non-zero"),
        ("inner_window = 2:4", "frozen inner_window bands"),
        ("k_path\nX; 0.5; 0\nend", "k_path point 0"),
    ],
)
def test_central_config_validation_rejects_invalid_algorithm_inputs(tmp_path, extra, message):
    incar = tmp_path / "incar"
    incar.write_text(_minimal_symmetry_incar() + "\n" + extra + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(incar)


def _minimal_symmetry_incar() -> str:
    return "\n".join(
        [
            "lattice_const = 1",
            "real_lattice_vectors = 1 0, 0 1",
            "k_points = -0.5:0.5:0.5, -0.5:0.5:0.5",
            "composition_of_b = 1 0, 0 1",
            "band_window = 0:3",
            "field_components = Ez",
            "dataset_file = Ez.txt",
            "metric_file = eps.txt",
            "mesh_file = mesh.mphtxt",
            "E_file = E.txt",
            "extension = 1, 1",
            "symmetry_file = p4mm.yaml",
            "symmetry_constrained = true",
            "wannier_targets",
            "center_s_A1; 0.0, 0.0; A1",
            "center_p_E; 0.0, 0.0; E",
            "end",
            "projections",
            "a; [0, 0]; 0; [1, 0, 1]; [2, 1, 1]; [2, -1, 1]",
            "end",
        ]
    )
