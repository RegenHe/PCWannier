import pcwannier
import pcwannier.compute as compute
import pcwannier.sources as sources
import pcwannier.symmetry as symmetry


def test_stable_top_level_entry_points_are_exported():
    assert {
        "load_config",
        "load_input",
        "run_calculation",
        "run_bloch_symmetry_preanalysis",
        "write_outputs",
        "write_bloch_symmetry_outputs",
    } <= set(pcwannier.__all__)


def test_compute_facade_only_exports_entry_points_and_integration_protocol():
    assert set(compute.__all__) == {
        "IntegrationMode",
        "MetricInnerProductProtocol",
        "create_metric_inner_product",
        "integrate_components",
        "integrate_scalar",
        "periodic_grid_coordinates",
        "run_calculation",
        "run_bloch_symmetry_preanalysis",
    }


def test_sources_facade_only_exports_registry_and_adapter_api():
    assert set(sources.__all__) == {
        "SourceAdapter",
        "load_input",
        "load_mesh",
        "resolve_source",
    }


def test_symmetry_facade_excludes_optimization_implementation_types():
    exported = set(symmetry.__all__)
    assert {
        "SpaceGroup",
        "SpaceGroupOperation",
        "SpaceGroupDefinition",
        "FiniteGroupDefinition",
        "SymmetryContext",
        "SymmetryModel",
        "analyze_bloch_symmetry",
        "run_bloch_symmetry_analysis",
        "run_symmetry_analysis",
        "load_space_group",
        "load_finite_group",
        "load_symmetry",
    } <= exported
    assert {
        "StateBlochSymmetryProvider",
        "SymmetryGaugeResult",
        "SymmetryDisentanglementResult",
        "SymmetryLocalizationResult",
        "TwistedRepresentation",
    }.isdisjoint(exported)
