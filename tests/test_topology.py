from types import SimpleNamespace

import numpy as np

from pcwannier.compute.tba import TBAModel
from pcwannier.compute.topology import Topology2D, calculate_topology
from pcwannier.data import BandResult, TopologyResult
from pcwannier.outputs import write_topology_figures


def _pauli_hamiltonian(kx, ky, mass):
    dx = np.sin(kx)
    dy = np.sin(ky)
    dz = mass + np.cos(kx) + np.cos(ky)
    return np.array([[dz, dx - 1j * dy], [dx + 1j * dy, -dz]], dtype=np.complex128)


def _qwz_occupied(mass, count=24):
    eigvecs = np.empty((count, count, 2, 1), dtype=np.complex128)
    for i in range(count):
        for j in range(count):
            _, vectors = np.linalg.eigh(_pauli_hamiltonian(2 * np.pi * i / count, 2 * np.pi * j / count, mass))
            eigvecs[i, j, :, 0] = vectors[:, 0]
    return eigvecs


def _bhz_occupied(mass, count=24):
    eigvecs = np.empty((count, count, 4, 2), dtype=np.complex128)
    for i in range(count):
        for j in range(count):
            kx = 2 * np.pi * i / count
            ky = 2 * np.pi * j / count
            hamiltonian = np.zeros((4, 4), dtype=np.complex128)
            hamiltonian[:2, :2] = _pauli_hamiltonian(kx, ky, mass)
            hamiltonian[2:, 2:] = np.conj(_pauli_hamiltonian(-kx, -ky, mass))
            _, vectors = np.linalg.eigh(hamiltonian)
            eigvecs[i, j] = vectors[:, :2]
    return eigvecs


def test_qwz_chern_number_and_gauge_invariance():
    occupied = _qwz_occupied(-1.0)
    _, chern = Topology2D().Chern_number(occupied)
    trivial = _qwz_occupied(3.0)
    _, trivial_chern = Topology2D().Chern_number(trivial)

    rng = np.random.default_rng(123)
    phase = np.exp(2j * np.pi * rng.random(occupied.shape[:2]))
    gauged = occupied * phase[:, :, None, None]
    _, gauged_chern = Topology2D().Chern_number(gauged)

    assert np.isclose(abs(chern), 1.0, atol=5e-3)
    assert np.isclose(trivial_chern, 0.0, atol=5e-3)
    assert np.isclose(gauged_chern, chern, atol=1e-10)


def test_bhz_wilson_loop_distinguishes_z2_phases():
    occupied = _bhz_occupied(-1.0)
    centers, _, topological = Topology2D().hybrid_Wilson_loop(occupied, direction=0)
    _, _, trivial = Topology2D().hybrid_Wilson_loop(_bhz_occupied(3.0), direction=0)

    rng = np.random.default_rng(321)
    gauged = np.empty_like(occupied)
    for idx in np.ndindex(occupied.shape[:2]):
        raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        gauge, _ = np.linalg.qr(raw)
        gauged[idx] = occupied[idx] @ gauge
    gauged_centers, _, gauged_z2 = Topology2D().hybrid_Wilson_loop(gauged, direction=0)

    assert topological == 1
    assert trivial == 0
    assert gauged_z2 == topological
    spectrum = np.exp(2j * np.pi * centers)
    gauged_spectrum = np.exp(2j * np.pi * gauged_centers)
    assert np.allclose(np.sum(gauged_spectrum, axis=1), np.sum(spectrum, axis=1), atol=1e-10)
    assert np.allclose(np.prod(gauged_spectrum, axis=1), np.prod(spectrum, axis=1), atol=1e-10)


def test_z2_requires_an_even_transverse_mesh():
    centers = np.zeros((9, 2), dtype=float)

    assert Topology2D._z2_from_tracked_centers(centers) is None


def test_chern_result_records_and_logs_one_based_band_range(caplog):
    occupied = _qwz_occupied(-1.0)
    band = BandResult(
        k_path=np.empty((0, 2)),
        k_axis=np.empty(0),
        high_sym_points=[],
        energies=np.empty((0, 1)),
        bz_eigvecs=occupied,
        groups=[[0]],
    )
    config = SimpleNamespace(kdim=2, hermitian=True, hybrid_Wilson_loop=False, Chern_number=True)

    with caplog.at_level("INFO"):
        result = calculate_topology(band, config)

    assert result.chern_bands["0"] == (1,)
    assert result.chern_bands["all"] == (1,)
    assert "bands=1" in caplog.text


def test_chern_figure_title_contains_band_range(tmp_path, monkeypatch):
    topology = TopologyResult(
        chern={"0": (np.zeros((2, 2)), 1.0)},
        chern_bands={"0": (2, 3)},
    )
    titles = []

    def capture_title(self, label, *args, **kwargs):
        titles.append(label)

    monkeypatch.setattr("matplotlib.axes.Axes.set_title", capture_title)
    write_topology_figures(tmp_path, topology)

    assert any("bands 2-3" in title for title in titles)


def test_dos_uses_normalized_brillouin_zone_mesh_and_pdos_sum_rule():
    config = SimpleNamespace(
        DOS_Brillouin_mesh=[8, 6],
        DOS_eps=0.05,
        DOS_num=2001,
        DOS=2,
        band_calc_num=2,
        hermitian=True,
        kdim=2,
        reciprocal_lattice_vectors=np.eye(2),
        lattice_const=1.0,
    )
    tba = object.__new__(TBAModel)
    tba.config = config

    def h_of_k(k_cart):
        base = np.diag([-1.0, 2.0]).astype(np.complex128)
        return np.broadcast_to(base, (k_cart.shape[0], 2, 2)).copy()

    energy, pdos = tba._calculate_dos(h_of_k, 2)
    config.DOS = 1
    total_energy, total = tba._calculate_dos(h_of_k, 2)

    assert np.array_equal(total_energy, energy)
    assert np.allclose(np.sum(pdos, axis=0), total[0], rtol=1e-13, atol=1e-13)
    assert np.isclose(np.trapezoid(total[0], energy), 2.0, atol=0.08)
