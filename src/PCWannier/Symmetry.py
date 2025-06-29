from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np
import spglib
import spgrep
import spgrep.representation

from .Log import Logger
from .Timer import Timer, timer
from .IO import IO

from .GlobalData import global_data
from .Utils import FieldData, WannierTools, StateCollection, Mesh

@dataclass(slots=True)
class OrbitSite:
    qj: np.ndarray
    Rj: np.ndarray
    vj: np.ndarray

class Symmetry:
    def __init__(
        self,
        lattice2d: np.ndarray,
        positions2d: np.ndarray,
        numbers: Sequence[int],
        *,
        vacuum: float = 20.0,
        aperiodic_dir: int = 2,
        symprec: float = 1.0e-5,
    ) -> None:
        if lattice2d.shape != (2, 2):
            raise ValueError("lattice2d must be 2x2 for a 2-D system.")

        self.lattice3d = np.zeros((3, 3), dtype=float)
        self.lattice3d[:2, :2] = lattice2d
        self.lattice3d[2, 2] = vacuum

        self.positions3d = np.c_[positions2d, np.full(len(positions2d), 0.5)]
        self.numbers = np.asarray(numbers, dtype=int)
        self.aperiodic_dir, self.symprec = aperiodic_dir, symprec

        self.lds: spglib.SpglibDataset = spglib.get_layergroup(
            (self.lattice3d, self.positions3d, self.numbers),
            aperiodic_dir=self.aperiodic_dir,
            symprec=self.symprec,
        )
        self.rotations, self.translations = self.filter_inplane_ops(
            self.lds.rotations, self.lds.translations
        )
        print(f"group: {self.lds.international} ({self.lds.number})")

    def get_site_group(self, q_idx: int, *, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        q = self._q_from_idx(q_idx)
        rot, trans = [], []
        for R, v in zip(self.rotations, self.translations):
            if np.all(np.abs((R @ q + v - q) % 1.0) < tol):
                rot.append(R); trans.append(v)
        return np.asarray(rot), np.asarray(trans)
    
    def build_orbit(self, q_idx: int, *, tol: float = 1e-6) -> List[OrbitSite]:
        q = self._q_from_idx(q_idx)
        orbit: List[OrbitSite] = []
        for R, v in zip(self.rotations, self.translations):
            qj = (R @ q + v) % 1.0
            if not any(np.all(np.abs(qj - s.qj) < tol) for s in orbit):
                orbit.append(OrbitSite(qj, R, v))
        return orbit
    
    def build_mapping(self, orbit: List[OrbitSite], *, tol: float = 1e-6) -> Dict[Tuple[int, int], Tuple[int, np.ndarray]]:
        mapping: Dict[Tuple[int, int], Tuple[int, np.ndarray]] = {}
        for g_idx, (R, v) in enumerate(zip(self.rotations, self.translations)):
            for j, site in enumerate(orbit):
                q_img = (R @ site.qj + v) % 1.0
                for jp, op in enumerate(orbit):
                    if np.all(np.abs(q_img - op.qj) < tol):
                        mapping[(g_idx, j)] = (jp, self.fractional_wrap(q_img - op.qj))
                        break
        return mapping
    
    def get_irrep_maps(self, site_rot: np.ndarray) -> List[Dict[Tuple[int, ...], np.ndarray]]:
        irreps = spgrep.get_crystallographic_pointgroup_irreps_from_symmetry(site_rot)
        return [{tuple(R.flatten()): d_mats[i] for i, R in enumerate(site_rot)} for d_mats in irreps]

    def pointgroup_chars_table(self, irrep_maps_list: List[Dict[Tuple[int, ...], np.ndarray]], rotations: np.ndarray) -> np.ndarray:
        def rot_key(R):
            return tuple(int(x) for x in R.flatten())

        rot_keys = [rot_key(R) for R in rotations]
        Ng = len(rot_keys)
        N_beta = len(irrep_maps_list)
        chi = np.empty((N_beta, Ng), dtype=complex)
        for i_beta, irrep_map in enumerate(irrep_maps_list):
            mats = [irrep_map[key] for key in rot_keys]
            Gamma = np.stack(mats)
            chi[i_beta] = spgrep.representation.get_character(Gamma)

        return chi

    def representation_characters(self, nm_list: List[Tuple[int, int]], rotations: np.ndarray) -> np.ndarray:
        Ng, dim = rotations.shape[0], len(nm_list)

        pair = {}
        for idx, (_, m) in enumerate(nm_list):
            pair.setdefault(abs(m), []).append((idx, m))

        Gamma = np.zeros((Ng, dim, dim), dtype=complex)
        for k, R in enumerate(rotations):
            alpha = np.arctan2(R[1, 0], R[0, 0])
            det   = int(round(np.linalg.det(R)))
            if det == 1:
                for i, (_, m) in enumerate(nm_list):
                    Gamma[k, i, i] = np.exp(1j * m * alpha)
                continue

            flip_x = R[0, 0] < 0
            flip_y = R[1, 1] < 0
            if 0 in pair:
                idx0, _ = pair[0][0]
                Gamma[k, idx0, idx0] = 1.0

            for m_abs, items in pair.items():
                if m_abs == 0:
                    continue
                if len(items) == 2:
                    (i_pos, _), (i_neg, _) = sorted(items, key=lambda t: t[1])
                    Gamma[k, i_pos, i_neg] = 1.0
                    Gamma[k, i_neg, i_pos] = 1.0
                    continue

                i, m = items[0]
                if m_abs % 2 == 1:
                    sign = -1 if ((m > 0 and flip_x) or (m < 0 and flip_y)) else 1
                else:
                    sign = -1 if (m < 0 and (flip_x ^ flip_y)) else 1
                Gamma[k, i, i] = sign

        return spgrep.representation.get_character(Gamma)

    def irreps_multiplicity(self, chars: np.ndarray, chi_irreps: np.ndarray, *, irrep_labels: Sequence[Any] | None = None, atol=1e-6) -> Dict[Any, int]:
        gsize = chars.size
        proj = (chi_irreps.conj() @ chars) / gsize
        if not np.allclose(proj.imag, 0.0, atol=atol):
            raise ValueError("Imaginary parts too large: check character tables.")
        proj = proj.real

        n_vec = np.rint(proj).astype(int)
        if irrep_labels is None:
            irrep_labels = range(len(n_vec))
        mult = {label: int(n) for label, n in zip(irrep_labels, n_vec) if n > 0}
        return mult
    
    def build_D_matrices(
        self, 
        orbit: List[OrbitSite], 
        mapping: Dict[Tuple[int, int], Tuple[int, np.ndarray]], 
        irrep_map: Dict[Tuple[int, ...], np.ndarray], 
        kvec: np.ndarray
    ) -> List[np.ndarray]:
        if kvec.shape == (2,):
            kvec = np.pad(kvec, (0, 1), mode='constant')
        elif kvec.shape != (3,):
            raise ValueError("kvec must be length-2 or length-3.")
        n_beta = next(iter(irrep_map.values())).shape[0]
        Nw = n_beta * len(orbit)
        Dall: List[np.ndarray] = []
        for g_idx, R in enumerate(self.rotations):
            D = np.zeros((Nw, Nw), dtype=complex)
            for j in range(len(orbit)):
                jp, T = mapping[(g_idx, j)]
                S = orbit[jp].Rj.T @ R @ orbit[j].Rj
                phase = np.exp(-1j*2*np.pi * (R @ kvec) @ T)
                D[jp*n_beta:(jp+1)*n_beta, j*n_beta:(j+1)*n_beta] = phase * irrep_map[tuple(S.flatten())]
            Dall.append(D)
        return Dall
    
    def build_total_D_matrices(
        self, block_data: List[Tuple[List[OrbitSite], 
        Dict[Tuple[int,int], Tuple[int,np.ndarray]], Dict[Tuple[int,...], np.ndarray]]], 
        kvec: np.ndarray
    ) -> List[np.ndarray]:
        block_dims = []
        for orbit, _, ir_map in block_data:
            n_beta = next(iter(ir_map.values())).shape[0]
            block_dims.append(n_beta * len(orbit))
        Nw = sum(block_dims)
        Ng = len(self.rotations)
        D_all = [np.zeros((Nw, Nw), dtype=complex) for _ in range(Ng)]

        offset = 0
        for (orbit, mapping, ir_map), dim in zip(block_data, block_dims):
            small_Ds = self.build_D_matrices(orbit, mapping, ir_map, kvec)
            for g_idx, D_small in enumerate(small_Ds):
                rows = slice(offset, offset + dim)
                D_all[g_idx][rows, rows] = D_small
            offset += dim
        return D_all
    
    def build_block(self, q_idx: int, lm_list):
        orbit = self.build_orbit(q_idx)
        site_rot, _ = self.get_site_group(q_idx)
        irrep_maps = self.get_irrep_maps(site_rot)
        chars = self.representation_characters(lm_list, site_rot)
        chi_irreps = self.pointgroup_chars_table(irrep_maps, site_rot)
        mult = self.irreps_multiplicity(chars, chi_irreps)
        mapping = self.build_mapping(orbit)

        print(f"Building block for q_idx={q_idx}, lm_list={lm_list}, characters={chars}, multiplicity={mult}")

        blocks = []
        for beta_idx, n_beta in mult.items():
            for copy in range(n_beta):
                blocks.append((orbit, mapping, irrep_maps[beta_idx]))
        return blocks

    def auto_build(self, proj: Dict[int, List[Tuple[int, int]]]):
        data = []
        for idx, lm_list in proj.items():
            data.extend(self.build_block(idx, lm_list))
        return data
    

    def get_k_group(
        self,
        kvec: np.ndarray,
        *,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        if kvec.shape == (2,):
            kf = np.pad(kvec, (0, 1), constant_values=0.0)
        elif kvec.shape == (3,):
            kf = kvec.copy()
        else:
            raise ValueError("kvec must be length-2 or length-3 in fractional coords.")

        keep_R, keep_v = [], []
        for R, v in zip(self.rotations, self.translations):
            delta_k = self.fractional_wrap(R @ kf - kf)
            if np.all(np.abs(delta_k) < tol):
                keep_R.append(R)
                keep_v.append(v)

        return np.asarray(keep_R), np.asarray(keep_v)


    
    def _q_from_idx(self, q_idx: int) -> np.ndarray:
        if not (0 <= q_idx < len(self.positions3d)):
            raise IndexError("q_idx out of range")
        return self.positions3d[q_idx]


    @staticmethod
    def filter_inplane_ops(
            rotations: np.ndarray,
            translations: np.ndarray,
            *,
            tol: float = 1e-6,
        ) -> Tuple[np.ndarray, np.ndarray]:
        keep: list[int] = []
        for i, (R, v) in enumerate(zip(rotations, translations)):
            if not np.allclose(R[2, :2], 0, atol=tol):
                continue
            if not np.allclose(R[:2, 2], 0, atol=tol):
                continue
            if not np.isclose(R[2, 2], 1, atol=tol):
                continue
            if not np.isclose(v[2] % 1.0, 0.0, atol=tol):
                continue
            keep.append(i)
        return rotations[keep], translations[keep]
    
    @staticmethod
    def _rotation_angle(R: np.ndarray) -> float:
        return np.arctan2(R[1, 0], R[0, 0])
    
    @staticmethod
    def fractional_wrap(vec: np.ndarray) -> np.ndarray:
        return (vec + 0.5) % 1.0 - 0.5
    

class Orthogonalizer:
    def __init__(self, tol_rel: float = 1e-6):
        self.tol_rel = tol_rel

    def group_eigenmodes(self, eigenvals: np.ndarray) -> List[List[int]]:
        idx_sorted = np.argsort(eigenvals)
        groups: List[List[int]] = []
        current = [idx_sorted[0]]
        base_val = eigenvals[idx_sorted[0]]

        for idx in idx_sorted[1:]:
            val = eigenvals[idx]
            if abs(val - base_val) <= self.tol_rel*abs(base_val):
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
                base_val = val
        groups.append(current)
        return groups
    
    def overlap_matrix(group: List[int], wavefuncs: List[np.ndarray], mesh: Mesh, espilon) -> np.ndarray:
        S = np.empty((len(group), len(group)), dtype=complex)
        for i, ia in enumerate(group):
            for j, ib in enumerate(group[i:], i):
                fd = FieldData("S", mesh, wavefuncs[ia].conj() * espilon * wavefuncs[ib])
                val = WannierTools.integrate_over_mesh(fd)
                S[i, j] = val
                S[j, i] = val.conjugate()
        return S
    
    def lowdin_orthogonalize_group(self, group: List[int], S: np.ndarray,atol: float = 1e-10) -> Tuple[List["FieldData"], np.ndarray]:
        w, V = np.linalg.eigh(S)
        keep = w > self.tol_rel * np.max(np.abs(w))
        if not np.all(keep):
            Logger.warning(f"Degenerate group ill-conditioned: {np.sum(~keep)} eigenvalues <= {atol}")
        
        inv_sqrt = np.diag(1.0 / np.sqrt(w[keep]))
        T = V[:, keep] @ inv_sqrt @ V[:, keep].conj().T
        return T
    
    @timer("build orthogonalization matrix - ")
    def build_orthogonalization_matrix(self, state_collection: StateCollection) -> np.ndarray:
        if 'O' in global_data.incar.use_cached_data:
            Logger.info(f"using cache data - O")
            self.transform = IO.load_cell_matrix(global_data.incar.O_file, shape=(len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))
            return self.transform

        self.transform = [[None for _ in range(len(state_collection.field[0]))] for _ in range(len(state_collection.field))]
        for i in range(len(state_collection.field)):
            for j in range(len(state_collection.field[i])):
                group = self.group_eigenmodes(state_collection.E[i][j])

                T = np.eye(len(state_collection.E[i][j]), dtype=complex)
                for g in group:
                    if len(g) == 1:
                        continue

                    S = self.overlap_matrix(g, state_collection.field[i][j], state_collection.mesh, state_collection.epsilon)
                    T_block = self.lowdin_orthogonalize_group(g, S)
                    idx = np.ix_(g, g)
                    T[idx] = T_block
                self.transform[i][j] = T
        return np.array(self.transform)
    
    def save_as(self, filename):
        IO.save_to_txt(filename, self.transform, (len(global_data.incar.k_points[0]), len(global_data.incar.k_points[1])))



if __name__ == "__main__":
    lattice2d = np.array([[1.0, 0.0], [0.0, 1.0]])
    positions2d = np.array([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0]])
    numbers = [0, 1, 1]
    
    sym = Symmetry(lattice2d, positions2d, numbers)
    print(len(sym.rotations))
    print(sym.get_site_group(0))
    print(sym.get_site_group(1))

    print(sym.build_orbit(0))
    print(sym.build_orbit(1))

    q_idx = 0

    kvec = np.array([0.0, 0.0])
    rot_q, _ = sym.get_site_group(q_idx)
    irr_maps = sym.get_irrep_maps(rot_q)

    print('-' * 20)
    table = sym.pointgroup_chars_table(irr_maps, rot_q)
    print(table)
    print('-' * 20)
    chars = sym.representation_characters([(1, 0), (2, 1), (2, -1), (3, 2), (3, -2)], rot_q)
    print(chars)

    decomposed = sym.irreps_multiplicity(chars, table)
    print(decomposed)

    print('-' * 20)
    proj = {
        0: [(1, 0), (2, 1), (2, -1)],
        1: [(1, 0)],
    }
    blocks = sym.auto_build(proj)

    D_matrices = sym.build_total_D_matrices(blocks, kvec)
    print('-' * 20)
    print(D_matrices)

    G = np.array([0.0, 0.0])
    M = np.array([0.5, 0.0])
    K = np.array([1/3, 1/3])

    for label, k in [("Γ", G), ("M", M), ("K", K)]:
        Rk, vk = sym.get_k_group(k)
        print(f"{label}: |G_k| = {len(Rk)}, Rk = {Rk}, vk = {vk}")
