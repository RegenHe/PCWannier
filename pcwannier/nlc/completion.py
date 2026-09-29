"""Symmetric, frozen-T Neighbor-based Longitudinal Completion (NLC).

The algorithm works entirely on run-scoped periodic fields. Only L is changed;
Gamma constants are handled jointly. Candidate rank deficiency is resolved with
symmetric gradient trials, and simultaneous updates require global descent.
"""
from __future__ import annotations

from itertools import product
import logging

import numpy as np
import scipy.linalg

from ..conventions import BlochFieldRepresentation, SpatialDiscretization
from ..data import BandChannelReference, InputBundle
from ..logging_utils import should_log_progress
from ..maxwell import FieldComponents, PrimaryField
from ..symmetry.analysis import decompose_little_group_characters
from ..symmetry.field_action import cartesian_field_matrix
from ..symmetry.stars import build_symmetry_stars
from .models import NLCCompletionArtifacts, NLCCompletionResult, NLCSettings

LOGGER = logging.getLogger(__name__)


def hermitian(a):
    """Remove roundoff before Hermitian eigendecompositions."""
    return (a + a.conj().T) / 2


class NeighborLongitudinalCompletion:
    """Run-scoped Fourier geometry, little-group blocks and L projectors.

    T coefficients use an orthonormal Fourier convention (including sqrt(V)).
    L is stored as scalar coefficients times q-hat, so its scalar space-group
    action is pseudoscalar for axial H. Only Gamma's missing constant is kept
    outside this q != 0 representation.
    """

    def __init__(self, state, settings: NLCSettings):
        self.state = state
        self.settings = settings
        self.cfg = state.config
        self.context = state.symmetry
        self.ops = self.context.model.group.operations
        self.lattice = np.asarray(self.cfg.real_lattice_vectors, dtype=float)
        self.reciprocal = np.linalg.inv(self.lattice).T
        self.volume = float(state.mesh.cell_volume)
        self.sign = int(state.bloch_sign)
        self.origin = np.asarray(state.mesh.fractional_vertices[0], dtype=float)
        self.kshape = tuple(state.k_shape)
        self.k = np.stack(np.meshgrid(*self.cfg.k_points, indexing='ij'), axis=-1).reshape(-1, 3)
        self.count = len(self.k)
        self.nt = len(state.E_idx[(0, 0, 0)])
        self.nl = self.cfg.band_calc_num - self.nt
        self.cutoff = settings.cutoff
        self.stars = build_symmetry_stars(self.context)
        self.shape = tuple(state.mesh.shape)
        self.points = int(np.prod(self.shape))
        self.full_g = np.rint(np.stack(np.meshgrid(*[np.fft.fftfreq(n) * n for n in self.shape], indexing='ij'), axis=-1).reshape(-1, 3)).astype(int)
        if len(np.unique(self.full_g, axis=0)) != self.points:
            raise ValueError('NLC Fourier grid has duplicate frequency indices.')
        # A bounded Cartesian sphere at each k is exactly rotation covariant.
        bound = int(np.ceil(self.cutoff * max(np.linalg.norm(self.lattice, axis=1)) + 2.))
        if 2 * bound + 1 >= min(self.shape):
            raise ValueError('NLC cutoff plus BZ-seam margin exceeds the spatial Fourier grid; reduce nlc_cutoff or refine the spatial grid.')
        self.g = np.asarray(list(product(range(-bound, bound + 1), repeat=3)), dtype=int)
        self.g_lookup = {tuple(g): i for i, g in enumerate(self.g)}
        self.ng = len(self.g)
        self.compact_full = np.ravel_multi_index((self.g % np.asarray(self.shape)).T, self.shape)
        q = (self.g[None] + self.sign*self.k[:, None]) @ self.reciprocal
        norms = np.linalg.norm(q, axis=-1)
        self.qhat = np.divide(q, norms[..., None], out=np.zeros_like(q), where=norms[..., None] > 1e-12)
        self.active = [np.flatnonzero((row > 1e-12) & (row <= self.cutoff + 1e-10)) for row in norms]
        self.zero_g = self.g_lookup[(0, 0, 0)]
        self.gamma = int(np.argmin(np.linalg.norm(self.k, axis=1)))
        if np.linalg.norm(self.k[self.gamma]) >= 1e-10:
            raise ValueError('NLC requires a mesh containing Gamma at k=(0,0,0).')
        for i, active in enumerate(self.active):
            if len(active) < self.nl - int(i == self.gamma):
                raise ValueError(f'NLC cutoff supplies too few longitudinal Fourier directions at k={self.k[i]}; increase nlc_cutoff.')

        self.t = np.empty((self.count, self.nt, self.points, 3), complex)
        self.energies = np.empty((self.count, self.nt))
        normalization_error = 0.
        for i, index in enumerate(np.ndindex(self.kshape)):
            values = np.asarray(state.get_internal_block(*index), dtype=complex)
            fourier = np.fft.fftn(values.reshape((self.nt,)+self.shape+(3,)), axes=(1, 2, 3)).reshape(self.nt, -1, 3)/self.points
            fourier *= np.exp(-2j*np.pi*(self.full_g@self.origin))[None, :, None]*np.sqrt(self.volume)
            gram = np.einsum('ngc,mgc->nm', fourier.conj(), fourier)
            normalization_error = max(normalization_error, float(np.linalg.norm(gram-np.eye(self.nt), ord=2)))
            eigenvalues, eigenvectors = np.linalg.eigh(hermitian(gram))
            if eigenvalues.min() <= 1e-10 or not np.isfinite(eigenvalues).all():
                raise ValueError(f'NLC physical T frame is rank deficient at k={index}.')
            correction = eigenvectors@np.diag(eigenvalues**-.5)@eigenvectors.conj().T
            self.t[i] = np.einsum('ngc,nm->mgc', fourier, correction)
            self.energies[i] = np.asarray(state.E[index], dtype=float)
        if normalization_error > 1e-8:
            raise ValueError(f'NLC requires internally orthonormal T fields; Gram error={normalization_error:.6g}. '
                             'Apply strict orthogonalization before calling the completion API.')
        self.tc = self.t[:, :, self.compact_full].transpose(0, 2, 3, 1)
        # The fixed Gamma space is C^3 plus the positive-frequency T modes.
        self.gamma_zero = np.flatnonzero(np.abs(self.energies[self.gamma]) <= self.cfg.gamma_zero_mode_tolerance)
        if len(self.gamma_zero) != 2:
            raise ValueError('NLC requires two selected constant transverse zero modes at Gamma.')
        mean = self.t[self.gamma, self.gamma_zero, 0, :].T
        u, s, _ = np.linalg.svd(mean, full_matrices=True)
        nonconstant = self.t[self.gamma, self.gamma_zero].copy()
        nonconstant[:, 0] = 0
        if np.max(np.abs(s-1)) > 1e-6 or np.linalg.norm(nonconstant) > 1e-6:
            raise ValueError('NLC Gamma transverse zero modes must span two independent constant fields.')
        self.gamma_harmonic = u[:, 2]
        self.gamma_positive_positions = np.flatnonzero(np.abs(self.energies[self.gamma]) > self.cfg.gamma_zero_mode_tolerance)
        self.gamma_positive = self.t[self.gamma, self.gamma_positive_positions]
        self.rotations = [cartesian_field_matrix(op, self.lattice, state.maxwell.symmetry_field_kind) for op in self.ops]

        directions = np.asarray(self.cfg.composition_of_b, dtype=float)
        weights = np.asarray(self.cfg.wb, dtype=float)
        retained = np.abs(weights) > 1e-12*max(float(np.max(np.abs(weights))), 1.)
        if np.any(weights[retained] <= 0):
            raise ValueError('NLC requires positive finite-difference neighbor weights.')
        if not np.allclose(directions, np.rint(directions), atol=1e-10, rtol=0):
            raise ValueError('NLC composition_of_b must use integer mesh steps.')
        self.directions = np.rint(directions[retained]).astype(int)
        self.weights = weights[retained]
        lookup = {tuple(direction): weight for direction, weight in zip(self.directions, self.weights)}
        for op in self.ops:
            for direction, weight in zip(self.directions, self.weights):
                rotated = (direction/np.asarray(self.kshape))@np.linalg.inv(op.rotation)*np.asarray(self.kshape)
                image = tuple(np.rint(rotated).astype(int))
                if not np.allclose(rotated, image, atol=1e-9, rtol=0) or image not in lookup or not np.isclose(weight, lookup[image], atol=1e-10, rtol=1e-8):
                    raise ValueError('NLC neighbor directions and weights must be closed under the space group; update composition_of_b.')
        self.neighbors = []
        for i, k in enumerate(self.k):
            index = np.asarray(np.unravel_index(i, self.kshape))
            neighbors = []
            for direction in self.directions:
                raw = index + direction
                wrapped = raw % np.asarray(self.kshape)
                j = int(np.ravel_multi_index(tuple(wrapped), self.kshape))
                seam = np.rint(k + direction / np.asarray(self.kshape) - self.k[j]).astype(int)
                compact_g = self.g + self.sign*seam
                source_compact = np.asarray([self.g_lookup.get(tuple(g), -1) for g in compact_g])
                source_full = np.ravel_multi_index((compact_g % np.asarray(self.shape)).T, self.shape)
                neighbors.append((j, seam, source_compact, source_full))
            self.neighbors.append(neighbors)
        self.star_data = []
        self.representation_reports = []
        self.prepare_representations()
        self.tt_score = 0.
        for i in range(self.count):
            for b, (j, seam, _, _) in enumerate(self.neighbors[i]):
                source_g = self.full_g + self.sign*seam
                source_full = np.ravel_multi_index((source_g % np.asarray(self.shape)).T, self.shape)
                overlap = np.einsum('ngc,mgc->nm', self.t[i].conj(), self.t[j, :, source_full, :].transpose(1, 0, 2))
                self.tt_score += self.weights[b]*np.linalg.norm(overlap, ord='fro')**2
        self.metadata = {'source': str(self.cfg.name), 'T_dimension': self.nt, 'L_dimension': self.nl,
                         'T_input_normalization_error': normalization_error, 'L_cartesian_cutoff_in_reciprocal_units': self.cutoff,
                         'neighbor_directions': self.directions.tolist(), 'neighbor_weights': self.weights.tolist(),
                         'Gamma_constant_pin': False, 'source_T_subspace_unchanged': True,
                         'representations': self.representation_reports}
        self.filtered_candidates = None
        if settings.filter_candidates:
            self.prepare_filtered_candidates()
        self.prepare_iteration_geometry()

    def prepare_iteration_geometry(self):
        """Cache fixed T projections and sparse L/BZ geometry for every link.

        Each longitudinal Fourier direction has one scalar coefficient. Work
        only on its nonzero support instead of allocating full vector fields
        and reprojecting frozen T during every optimization iteration.
        Gamma's missing harmonic uses its constant vector as this direction.
        """
        self.supports = [np.append(active, self.zero_g) if i == self.gamma else active
                         for i, active in enumerate(self.active)]
        directions = [self.qhat[i, support].astype(complex) for i, support in enumerate(self.supports)]
        directions[self.gamma][-1] = self.gamma_harmonic
        self.iteration_geometry = []
        self._initial_covariances = {}
        for i, links in enumerate(self.neighbors):
            left = self.supports[i]
            left_lookup = {tuple(self.g[g]): row for row, g in enumerate(left)}
            cached = []
            for j, seam, _, _ in links:
                right = self.supports[j]
                right_at_left = self.g[right] - self.sign*seam
                left_at_right = self.g[left] + self.sign*seam
                source_full = np.ravel_multi_index((left_at_right % np.asarray(self.shape)).T, self.shape)
                destination_full = np.ravel_multi_index((right_at_left % np.asarray(self.shape)).T, self.shape)
                lt = np.einsum('gc,ngc->gn', directions[i].conj(), self.t[j][:, source_full])
                tl = np.einsum('gc,ngc->gn', directions[j].conj(), self.t[i][:, destination_full])
                left_rows, right_rows = [], []
                for row, g in enumerate(right_at_left):
                    match = left_lookup.get(tuple(g))
                    if match is not None:
                        left_rows.append(match)
                        right_rows.append(row)
                left_rows = np.asarray(left_rows, dtype=int)
                right_rows = np.asarray(right_rows, dtype=int)
                factors = np.einsum('gc,gc->g', directions[i][left_rows].conj(), directions[j][right_rows])
                cached.append((j, lt, tl, left_rows, right_rows, factors))
            self.iteration_geometry.append(cached)

    def supported_coefficients(self, ell):
        coefficients = [ell[i, support].copy() for i, support in enumerate(self.supports)]
        coefficients[self.gamma][-1, 0] = 1.
        return coefficients

    def prepare_filtered_candidates(self):
        """Remove forbidden little-group content from candidates, keeping frozen T unchanged."""
        neighbor_steps = (self.directions / np.asarray(self.kshape)) @ self.reciprocal
        cutoff = self.cutoff + np.max(np.linalg.norm(neighbor_steps, axis=1)) + .05
        self.filtered_candidates = []
        removed = []
        for i in range(self.count):
            current_index = tuple(np.unravel_index(i, self.kshape))
            indices = tuple(
                j for j in range(len(self.ops))
                if self.context.k_mappings[j][i].target_k_index == current_index
            )
            resolved = self.context.model.group_definition.resolve_little_group(
                indices, self.k[i], bloch_convention=self.context.model.bloch_convention
            )
            q = (self.g + self.sign*self.k[i]) @ self.reciprocal
            active = np.flatnonzero(np.linalg.norm(q, axis=1) <= cutoff+1e-10)
            source = self.tc[i, active].copy()
            chars = {}
            if i == self.gamma:
                source = np.concatenate(
                    (source[:, :, self.gamma_positive_positions], np.zeros((len(active), 3, 3), complex)),
                    axis=2,
                )
                source[np.flatnonzero(active == self.zero_g)[0], :, -3:] = np.eye(3)
            for name, op_index in zip(resolved.table.operation_names, indices):
                transformed = self.act_full_t(i, op_index)
                if i == self.gamma:
                    value = np.trace(self.rotations[op_index]) + np.einsum(
                        'ngc,ngc->', self.gamma_positive.conj(), transformed[self.gamma_positive_positions]
                    )
                else:
                    value = np.einsum('ngc,ngc->', self.t[i].conj(), transformed)
                chars[name] = complex(value)
            decomposition = decompose_little_group_characters(resolved, chars)
            lookup = {self.g[g].tobytes(): local for local, g in enumerate(active)}
            projected = np.zeros_like(source)
            for irrep in resolved.require_irreps():
                if decomposition.multiplicities[irrep.name] <= 0:
                    continue
                for character, op_index in zip(irrep.characters, indices):
                    op = self.ops[op_index]
                    transformed_q = (self.g[active] + self.sign*self.k[i]) @ np.linalg.inv(op.rotation)
                    target_g = np.rint(transformed_q - self.sign*self.k[i]).astype(int)
                    destination = np.array([lookup[g.tobytes()] for g in target_g])
                    phase = np.exp(-2j*np.pi*(transformed_q @ op.translation))
                    transformed = np.einsum('ab,gbn->gan', self.rotations[op_index], source)
                    projected[destination] += (
                        irrep.dimension / len(indices) * character.conjugate()
                        * transformed * phase[:, None, None]
                    )
            full = np.zeros((self.ng, 3, source.shape[-1]), complex)
            full[active] = projected
            self.filtered_candidates.append(full)
            removed.append(float(np.linalg.norm(source-projected)**2))
        self.metadata['candidate_only_little_group_filter'] = {
            'enabled': True,
            'maximum_removed_candidate_weight': max(removed),
            'scope': 'Only seed/update candidates are filtered. Actual frozen T and the measured objective are unchanged.',
        }

    def transverse_candidate(self, item):
        j, _, compact, full = item
        if self.filtered_candidates is None:
            return self.t[j, :, full, :].transpose(0, 2, 1)
        result = np.zeros((self.ng, 3, self.filtered_candidates[j].shape[-1]), complex)
        valid = compact >= 0
        result[valid] = self.filtered_candidates[j][compact[valid]]
        return result

    def scalar_action(self, source, target, op_index):
        """Axial longitudinal action, including translation and reciprocal fold."""
        op = self.ops[op_index]
        active = self.active[source]
        transformed_q = (self.g[active] + self.sign*self.k[source]) @ np.linalg.inv(op.rotation)
        target_g = np.rint(transformed_q - self.sign*self.k[target]).astype(int)
        positions = {self.g[index].tobytes(): local for local, index in enumerate(self.active[target])}
        destination = np.asarray([positions[g.tobytes()] for g in target_g])
        phase = np.linalg.det(op.rotation) * np.exp(-2j*np.pi * (transformed_q @ op.translation))
        return destination, phase

    def act_full_t(self, i, op_index):
        op = self.ops[op_index]
        mapping = self.context.k_mappings[op_index][i]
        target = int(np.ravel_multi_index(mapping.target_k_index, self.kshape))
        transformed_q = (self.full_g + self.sign*self.k[i]) @ np.linalg.inv(op.rotation)
        gtarget = np.rint(transformed_q - self.sign*self.k[target]).astype(int)
        positions = np.ravel_multi_index((gtarget % np.asarray(self.shape)).T, self.shape)
        if len(np.unique(positions)) != self.points:
            raise ValueError('NLC spatial FFT grid is not closed under the configured rotation; use a symmetry-compatible grid.')
        phase = np.exp(-2j*np.pi * (transformed_q @ op.translation))
        output = np.empty_like(self.t[i])
        output[:, positions, :] = np.einsum('ab,ngb->nga', self.rotations[op_index], self.t[i]) * phase[None, :, None]
        return output

    def prepare_representations(self):
        for star in self.stars.stars:
            i = star.representative_flat_index
            paths = star.representative_member.paths
            indices = tuple(path.operation_index for path in paths)
            resolved = self.context.model.group_definition.resolve_little_group(indices, self.k[i], bloch_convention=self.context.model.bloch_convention)
            characters_t = {}
            characters_target = {}
            actions = []
            for name, op_index in zip(resolved.table.operation_names, indices):
                transformed = self.act_full_t(i, op_index)
                if i == self.gamma:
                    value = np.trace(self.rotations[op_index]) + np.einsum('ngc,ngc->', self.gamma_positive.conj(), transformed[self.gamma_positive_positions])
                else:
                    value = np.einsum('ngc,ngc->', self.t[i].conj(), transformed)
                characters_t[name] = complex(value)
                characters_target[name] = complex(np.trace(self.context.target_matrix(op_index, self.k[i])))
                destination, phase = self.scalar_action(i, i, op_index)
                actions.append((destination, phase))
            decomposition_t = decompose_little_group_characters(resolved, characters_t)
            decomposition_target = decompose_little_group_characters(resolved, characters_target)
            if max(decomposition_t.max_residual, decomposition_target.max_residual) > self.cfg.representation_character_tolerance:
                raise ValueError(f'NLC cannot resolve integer little-group multiplicities at k={self.k[i]}; '
                                 f'T residual={decomposition_t.max_residual:.6g}, '
                                 f'target residual={decomposition_target.max_residual:.6g}.')
            needed = {irrep.name: decomposition_target.multiplicities[irrep.name] - decomposition_t.multiplicities[irrep.name] for irrep in resolved.require_irreps()}
            if any(count < 0 for count in needed.values()):
                raise RuntimeError(f'Target cannot contain frozen T at {self.k[i]}: {needed}')
            expected = self.nl - int(i == self.gamma)
            if sum(irrep.dimension * needed[irrep.name] for irrep in resolved.require_irreps()) != expected:
                raise ValueError(f'NLC target/T representation dimensions disagree at k={self.k[i]}.')
            active = self.active[i]
            d = len(active)
            blocks = []
            for irrep in resolved.require_irreps():
                count = needed[irrep.name] * irrep.dimension
                if not count:
                    continue
                central = np.zeros((d, d), complex)
                for character, (destination, phase) in zip(irrep.characters, actions):
                    central[destination, np.arange(d)] += irrep.dimension / len(actions) * character.conjugate() * phase
                central = hermitian(central)
                if np.linalg.norm(central @ central - central, ord=2) >= 1e-8:
                    raise ValueError(f'NLC irrep projector is inconsistent at k={self.k[i]} / {irrep.name}.')
                ev, vectors = np.linalg.eigh(central)
                isotypic = vectors[:, ev > .5]
                if isotypic.shape[1] < count:
                    raise RuntimeError(f'L Fourier space lacks {irrep.name} at {self.k[i]}')
                blocks.append((irrep.name, irrep.dimension, count, isotypic))
            self.star_data.append((star, actions, blocks))
            self.representation_reports.append({'k': self.k[i].tolist(), 'T_or_Gamma_fixed_irreps': decomposition_t.multiplicities,
                 'target_irreps': decomposition_target.multiplicities, 'variable_L_irreps': needed,
                 'T_character_rounding_error': decomposition_t.max_residual,
                 'Gamma_low_L_pinned': False, 'L_basis_dimension': d})

    def l_vectors(self, i, ell):
        values = self.qhat[i, :, :, None] * ell[i, :, None, :]
        if i == self.gamma:
            values[self.zero_g, :, 0] = self.gamma_harmonic
        return values

    def neighbor_vectors(self, i, item, ell):
        j, seam, compact, full = item
        transverse = self.t[j, :, full, :].transpose(0, 2, 1)
        longitudinal = np.zeros((self.ng, 3, self.nl), complex)
        valid = compact >= 0
        longitudinal[valid] = self.l_vectors(j, ell)[compact[valid]]
        return transverse, longitudinal

    def covariance(self, i, ell, *, coefficients=None):
        z = self.initialize_covariance(i).copy()
        if coefficients is None:
            coefficients = self.supported_coefficients(ell)
        for b, item in enumerate(self.iteration_geometry[i]):
            j, _, _, left_rows, right_rows, factors = item
            retained = left_rows < len(self.active[i])
            longitudinal = np.zeros((len(self.active[i]), self.nl), complex)
            longitudinal[left_rows[retained]] = factors[retained, None]*coefficients[j][right_rows[retained]]
            if self.filtered_candidates is not None and j == self.gamma:
                # The filtered Gamma source already contains all three fixed
                # constants. Its missing supplied-T constant is L column zero,
                # so append only the variable positive-frequency Gamma L modes.
                longitudinal = longitudinal[:, 1:]
            z += self.weights[b]*(longitudinal @ longitudinal.conj().T)
        return z

    def initialize_covariance(self, i):
        if i in self._initial_covariances:
            return self._initial_covariances[i]
        candidates = []
        for b, item in enumerate(self.neighbors[i]):
            transverse = self.transverse_candidate(item)
            candidates.append(np.einsum('gc,gcn->gn', self.qhat[i, self.active[i]], transverse[self.active[i]])*np.sqrt(self.weights[b]))
        r = np.concatenate(candidates, axis=1)
        z = r @ r.conj().T
        self._initial_covariances[i] = z
        return z

    def select(self, data, z, old=None, mixing=1., initial=False):
        star, actions, blocks = data
        i = star.representative_flat_index
        active = self.active[i]
        averaged = np.zeros_like(z)
        for destination, phase in actions:
            averaged[np.ix_(destination, destination)] += z * phase[:, None] * phase.conj()[None, :]
        z = hermitian(averaged / len(actions))
        covariance_scale = max(float(scipy.linalg.eigh(z, subset_by_index=(len(z)-1, len(z)-1),
            eigvals_only=True, check_finite=False)[0]), 1e-12)
        if old is not None:
            z = mixing * z + (1-mixing) * (old @ old.conj().T) * covariance_scale
        columns = []
        diagnostics = []
        for label, dimension, count, isotypic in blocks:
            reduced = hermitian(isotypic.conj().T @ z @ isotypic)
            values, vectors = np.linalg.eigh(reduced)
            selected = values[-count:]
            # Compare to the whole candidate covariance, not a weak irrep block.
            # A nearly zero block made from discretization noise is not a seed.
            rank = int(np.sum(values > max(1e-12, covariance_scale*self.settings.rank_tolerance)))
            fallback = rank < count
            if fallback:
                if not initial:
                    raise RuntimeError(f'Update lost required L candidate rank at {self.k[i]} / {label}')
                # Complete Cartesian plane-gradient trial star, with a radial envelope.
                q2 = np.sum(((self.g[active]+self.sign*self.k[i]) @ self.reciprocal)**2, axis=1)
                regularizer = isotypic.conj().T @ (np.exp(-q2/4.)[:, None] * isotypic)
                reduced += max(covariance_scale, 1e-6) * 1e-5 * regularizer
                values, vectors = np.linalg.eigh(hermitian(reduced))
            chosen = isotypic @ vectors[:, -count:]
            mapped_frames = [chosen[np.argsort(destination)] * phase[np.argsort(destination), None]
                             for destination, phase in actions]
            residual = max(np.linalg.norm(mapped-chosen@(chosen.conj().T@mapped), ord=2)
                           for mapped in mapped_frames)
            # Degeneracies must be retained as complete representations.
            if residual > 1e-7:
                raise RuntimeError(f'Irrep eigenvalue cutoff broke symmetry: {label}, residual={residual}')
            columns.append(chosen)
            diagnostics.append({'irrep': label, 'selected_dimension': count, 'T_candidate_rank': rank,
                                'gradient_trial_fallback': fallback, 'minimum_T_candidate_eigenvalue': float(selected.min()),
                                'selected_eigenvalue_relative_to_full_covariance': float(selected.min()/covariance_scale),
                                'selection_gap': float(values[-count]-values[-count-1]) if len(values)>count else None})
        result = np.column_stack(columns) if columns else np.empty((len(active), 0), complex)
        if np.linalg.norm(result.conj().T @ result - np.eye(result.shape[1])) >= 1e-8:
            raise RuntimeError(f'NLC selected a nonorthogonal L frame at k={self.k[i]}.')
        return result, diagnostics

    def propagate(self, representatives):
        ell = np.zeros((self.count, self.ng, self.nl), complex)
        for data, frame in zip(self.star_data, representatives):
            star = data[0]
            i = star.representative_flat_index
            offset = int(i == self.gamma)
            for member in star.members:
                j = member.flat_index
                destination, phase = self.scalar_action(i, j, member.canonical_path.operation_index)
                mapped = np.zeros((len(self.active[j]), frame.shape[1]), complex)
                mapped[destination] = frame * phase[:, None]
                ell[j, self.active[j], offset:] = mapped
        return ell

    def cost(self, ell):
        """Directed joint-projector distance per k, equal to 2*Omega_I.

        Frozen TT overlaps are cached; the varying terms are TL, LT and LL.
        Using the actual T here ensures filtering never changes the objective.
        """
        score = self.tt_score
        coefficients = self.supported_coefficients(ell)
        for i in range(self.count):
            left = coefficients[i]
            for b, (j, lt_projection, tl_projection, left_rows, right_rows, factors) in enumerate(self.iteration_geometry[i]):
                right = coefficients[j]
                lt = left.conj().T @ lt_projection
                tl = right.conj().T @ tl_projection
                ll = left[left_rows].conj().T @ (factors[:, None]*right[right_rows])
                score += self.weights[b]*(np.linalg.norm(lt)**2 + np.linalg.norm(tl)**2 + np.linalg.norm(ll)**2)
        return float(2*(self.count*np.sum(self.weights)*(self.nt+self.nl)-score)/self.count)

    def optimize(self):
        """Update L on star representatives and accept only global descent."""
        representatives = []
        initial_reports = []
        for data in self.star_data:
            i = data[0].representative_flat_index
            frame, report = self.select(data, self.initialize_covariance(i), initial=True)
            representatives.append(frame)
            initial_reports.append({'k': self.k[i].tolist(), 'blocks': report})
        ell = self.propagate(representatives)
        value = self.cost(ell)
        history = [{'iteration': 0, 'cost': value}]
        LOGGER.info('NLC initial joint projector cost: %.12g', value)
        converged = False
        for iteration in range(1, self.settings.max_iterations+1):
            coefficients = self.supported_coefficients(ell)
            covariances = [self.covariance(data[0].representative_flat_index, ell, coefficients=coefficients)
                          for data in self.star_data]
            mixing = self.settings.mixing
            accepted = False
            for _ in range(9):
                proposed = [self.select(data, z, old=old, mixing=mixing)[0] for data, z, old in zip(self.star_data, covariances, representatives)]
                candidate = self.propagate(proposed)
                new_value = self.cost(candidate)
                if new_value <= value + 1e-10:
                    accepted = True
                    break
                mixing /= 2
            if not accepted:
                history.append({'iteration': iteration, 'cost': value, 'status': 'line search stalled'})
                break
            # For orthonormal frames, ||P_a-P_b||_F^2 = 2r-2||a^H b||_F^2.
            change = max(float(np.sqrt(max(0., 2*a.shape[1]-2*np.linalg.norm(a.conj().T@b)**2)))
                         for a,b in zip(representatives, proposed))
            decrease = value - new_value
            ell = candidate
            representatives = proposed
            value = new_value
            history.append({'iteration': iteration, 'cost': value, 'projector_change': change, 'mixing': mixing})
            finished = decrease <= self.settings.cost_tolerance and change <= self.settings.projector_tolerance
            log = LOGGER.info if should_log_progress(
                iteration, total=self.settings.max_iterations, finished=finished
            ) else LOGGER.debug
            log('NLC iter %d cost=%.12g projector_change=%.6g mixing=%.6g', iteration, value, change, mixing)
            if finished:
                converged = True
                break
        if not all(history[i]['cost'] <= history[i-1]['cost'] + 1e-9 for i in range(1, len(history))):
            raise RuntimeError('NLC accepted an increase in joint projector cost.')
        return ell, {'converged': converged, 'history': history, 'initial_candidate_ranks': initial_reports}

    def validate(self, ell):
        norm_error = 0.
        cross_error = 0.
        curl_error = 0.
        covariance_error = 0.
        for i in range(self.count):
            vectors = self.l_vectors(i, ell)
            gram = np.einsum('gcn,gcm->nm', vectors.conj(), vectors)
            norm_error = max(norm_error, float(np.linalg.norm(gram-np.eye(self.nl), ord=2)))
            cross = np.einsum('gcn,gcm->nm', self.tc[i].conj(), vectors)
            cross_error = max(cross_error, float(np.linalg.norm(cross, ord=2)))
            q = (self.g+self.sign*self.k[i]) @ self.reciprocal
            curl = np.cross(q[:, :, None], vectors, axisa=1, axisb=1, axisc=1)
            curl_error = max(curl_error, float(np.linalg.norm(curl)))
            if i == self.gamma:
                continue
            source = ell[i, self.active[i]]
            for op_index in range(len(self.ops)):
                mapping = self.context.k_mappings[op_index][i]
                j = int(np.ravel_multi_index(mapping.target_k_index, self.kshape))
                destination, phase = self.scalar_action(i, j, op_index)
                mapped = np.zeros_like(source)
                mapped[destination] = source*phase[:, None]
                target = ell[j, self.active[j]]
                # Avoid constructing a large projector: leakage has the same principal angles.
                overlap = target.conj().T @ mapped
                covariance_error = max(covariance_error, float(np.linalg.norm(mapped-target@overlap,ord=2)))
        if norm_error >= 1e-8 or cross_error >= 1e-8 or curl_error >= 1e-10 or covariance_error >= 1e-7:
            raise ValueError('NLC frame validation failed: '
                             f'L Gram={norm_error:.6g}, T/L overlap={cross_error:.6g}, '
                             f'curl={curl_error:.6g}, space-group leakage={covariance_error:.6g}. '
                             'The input T fields must be transverse in the same Fourier convention.')
        return {'L_orthonormality_error': norm_error, 'T_L_overlap_error': cross_error,
                'L_curl_error_in_2pi_units': curl_error, 'L_symmetry_leakage_excluding_Gamma': covariance_error,
                'Gamma_three_constants_exactly_contained': True}


def _validate_input(state) -> None:
    from ..compute.kspace import is_complete_uniform_k_mesh

    if state.mesh.discretization is not SpatialDiscretization.PERIODIC_FOURIER_COLLOCATION or state.mesh.dimension != 3:
        raise ValueError('NLC requires a three-dimensional periodic Fourier grid.')
    if state.maxwell.field_components is not FieldComponents.FULL_VECTOR or state.maxwell.primary_field is not PrimaryField.MAGNETIC:
        raise ValueError('NLC requires full-vector magnetic fields.')
    if not np.allclose(state.metric_material, 1., atol=1e-10, rtol=0):
        raise ValueError('NLC currently requires nonmagnetic media (mu=1).')
    if not state.stores_periodic_bloch_parts:
        raise ValueError('NLC input must contain periodic Bloch parts in the internal orthonormal basis.')
    if state.symmetry is None or state.symmetry.model.group_definition is None:
        raise ValueError('NLC requires a resolved space group and target Wannier representation.')
    if any(op.antiunitary for op in state.symmetry.model.group.operations):
        raise ValueError('NLC currently supports unitary space-group operations only.')
    if not is_complete_uniform_k_mesh(state.config.k_points) or min(state.k_shape) < 2:
        raise ValueError('NLC requires a complete uniform three-dimensional k mesh with at least two points per axis.')
    counts = {len(state.E_idx[index]) for index in np.ndindex(state.k_shape)}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError('NLC requires a fixed number of at least two physical T bands at every k.')
    if state.config.band_calc_num <= min(counts):
        raise ValueError('NLC requires N_W > N_T; the target representation must provide auxiliary L dimensions.')
    if sum(target.wannier_dimension for target in state.symmetry.model.targets) != state.config.band_calc_num:
        raise ValueError('NLC target representation dimension must equal N_W.')
    for index in np.ndindex(state.k_shape):
        ids = np.asarray(state.E_idx[index], dtype=int)
        if any(state.band_channels.get(int(band), BandChannelReference('H', int(band))).channel != 'H' for band in ids):
            raise ValueError('NLC input must contain physical T bands only.')
        block = np.asarray(state.get_internal_block(*index))
        if block.shape != (len(ids), state.mesh.point_count, 3) or not np.isfinite(block).all():
            raise ValueError(f'NLC input field has an invalid shape or non-finite values at k={index}.')


def _build_augmented_bundle(problem, ell) -> InputBundle:
    """Keep the physical Hamiltonian and append negative Laplacian Ritz modes."""
    from ..compute.augmentation import resolve_channel_analysis_context

    state = problem.state
    shape = state.k_shape
    nt, nl = problem.nt, problem.nl
    fields = np.empty(shape, dtype=object)
    energies = np.empty(shape, dtype=object)
    band_indices = np.empty(shape, dtype=object)
    inner_indices = np.empty(shape, dtype=object)
    zero_modes = np.empty(shape, dtype=object)
    longitudinal_zero = np.empty(shape, dtype=object)
    hamiltonians = np.empty(shape, dtype=object)
    auxiliary_energies = np.empty(shape + (nl,), dtype=float)
    l_offset = int(np.asarray(state.energy_matrix).shape[-1])
    channels = dict(state.band_channels)
    scale = (2*np.pi / float(state.config.lattice_const))**2 / problem.settings.eta
    phase = np.exp(2j*np.pi*(problem.g @ problem.origin)) / np.sqrt(problem.volume)
    transform = state.get_transform(False)
    for i, index in enumerate(np.ndindex(shape)):
        longitudinal = problem.l_vectors(i, ell)
        q2 = np.sum(((problem.g + problem.sign*problem.k[i]) @ problem.reciprocal)**2, axis=1)
        ritz = hermitian(np.einsum('gcn,g,gcm->nm', longitudinal.conj(), scale*q2, longitudinal))
        values, vectors = np.linalg.eigh(ritz)
        values = np.maximum(values, 0.)
        longitudinal = np.einsum('gcn,nm->gcm', longitudinal, vectors)
        full = np.zeros((nl, problem.points, 3), dtype=complex)
        full[:, problem.compact_full, :] = longitudinal.transpose(2, 0, 1)*phase[None, :, None]
        real = np.fft.ifftn(full.reshape((nl,) + problem.shape + (3,)), axes=(1, 2, 3))*problem.points
        fields[index] = np.ascontiguousarray(np.concatenate((state.get_internal_block(*index), real.reshape(nl, -1, 3))))
        auxiliary_energies[index] = -values
        energies[index] = np.concatenate((np.asarray(state.E[index], dtype=float), -values))
        physical_ids = np.asarray(state.E_idx[index], dtype=int)
        band_indices[index] = np.concatenate((physical_ids, l_offset + np.arange(nl))).tolist()
        inner_indices[index] = physical_ids.tolist()
        zero_modes[index] = np.abs(energies[index]) <= state.config.gamma_zero_mode_tolerance
        longitudinal_zero[index] = np.flatnonzero(values <= state.config.gamma_zero_mode_tolerance).tolist()
        physical_h = transform[index].conj().T @ state.base_hamiltonian_at(index) @ transform[index]
        hamiltonians[index] = scipy.linalg.block_diag(hermitian(physical_h), -np.diag(values))
        for band in physical_ids:
            channels.setdefault(int(band), BandChannelReference('H', int(band)))
    channels.update({l_offset + position: BandChannelReference('L', position) for position in range(nl)})
    return InputBundle(
        config=state.config, maxwell=state.maxwell, bloch_convention=state.bloch_convention,
        mesh=state.mesh, fields=fields, metric_material=state.metric_material,
        energies=energies, band_indices=band_indices, inner_band_indices=inner_indices,
        energy_matrix=np.concatenate((np.asarray(state.energy_matrix, dtype=float), auxiliary_energies), axis=-1),
        field_representation=BlochFieldRepresentation.PERIODIC_PART,
        symmetry=resolve_channel_analysis_context(state, l_offset=l_offset),
        zero_modes=zero_modes, band_channels=channels,
        auxiliary_zero_mode_bands={'longitudinal': longitudinal_zero}, base_hamiltonians=hamiltonians,
    )


def prepare_longitudinal_bundle(state, *, settings: NLCSettings | None = None) -> NLCCompletionArtifacts:
    """Complete frozen T with symmetric neighbor-derived L, entirely in memory.

    ``state`` must contain internally orthonormal periodic magnetic fields on a
    full 3D k mesh, with mu=1 and a prepared target symmetry context. The result
    bundle can be passed directly to the standard symmetry/Wannier pipeline.
    Exact Gamma is handled as a joint constant space, without assigning its
    missing constant a separate L irrep. Negative Ritz energies distinguish
    auxiliary branches; they are not measured photonic eigenvalues.
    """
    _validate_input(state)
    settings = settings or NLCSettings.from_config(state.config)
    problem = NeighborLongitudinalCompletion(state, settings)
    ell, optimization = problem.optimize()
    diagnostics = dict(problem.metadata)
    diagnostics.update(problem.validate(ell))
    diagnostics['initial_candidate_ranks'] = optimization['initial_candidate_ranks']
    augmented = _build_augmented_bundle(problem, ell)
    history = tuple(optimization['history'])
    result = NLCCompletionResult(
        transverse_dimension=problem.nt, auxiliary_dimension=problem.nl,
        wannier_dimension=problem.nt + problem.nl, settings=settings,
        converged=optimization['converged'], iterations=int(history[-1]['iteration']),
        initial_cost=history[0]['cost'], final_cost=history[-1]['cost'],
        history=history, diagnostics=diagnostics,
    )
    if not result.converged:
        LOGGER.warning('NLC stopped without meeting convergence tolerances after %d iterations; '
                       'inspect nlc_report_file before using the interpolated bands.', result.iterations)
    return NLCCompletionArtifacts(result=result, augmented_bundle=augmented)

