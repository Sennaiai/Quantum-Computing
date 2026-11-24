"""Scientific checks for the fixed-orbital HCN benchmark."""
import unittest
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autograd import value_and_grad
import numpy as np
import pennylane as qml

from hcn_benchmark import (
    build_problem, full_fci_reference, geometry_angstrom, make_circuits,
    map_active_integrals, optimize_seed, pbe_grid_control, sector_indices,
    state_diagnostics, validate_run,
)


class HCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem()
        cls.good_run = optimize_seed(cls.problem)
        cls.full_fci = full_fci_reference(cls.problem)

    def test_electrons_active_space_and_orbital_character(self):
        ref = self.problem.reference
        self.assertEqual((ref["total_electrons"], ref["spatial_orbitals"]), (14, 11))
        self.assertEqual(ref["core_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(ref["active_indices"], [5, 6, 7, 8])
        self.assertEqual(set(ref["mo_symmetry_labels"][5:7]), {"B1", "B2"})
        self.assertEqual(set(ref["mo_symmetry_labels"][7:9]), {"B1", "B2"})
        self.assertEqual(len(sector_indices()), 36)

    def test_matched_references_and_energy_convention(self):
        ref = self.problem.reference
        self.assertAlmostEqual(ref["casci_hartree"], -91.775084878382, delta=1e-8)
        self.assertAlmostEqual(ref["sector_exact_hartree"], ref["casci_hartree"], delta=1e-8)
        self.assertAlmostEqual(ref["hf_qubit_hartree"], ref["rhf_hartree"], delta=1e-8)
        self.assertGreater(ref["nuclear_repulsion_hartree"], 20)
        self.assertGreater(ref["core_energy_hartree"], -100)
        self.assertAlmostEqual(sum(ref["active_natural_occupations"]), 4, delta=1e-10)

    def test_integral_axes_and_constant_counting(self):
        model = self.problem
        np.testing.assert_array_equal(model.two_body_physicist, model.two_body_chemist.swapaxes(1, 3))
        _, _, _, no_core = map_active_integrals(0., model.one_body, model.two_body_chemist)
        np.testing.assert_allclose(model.matrix-no_core, model.core_energy*np.eye(256), atol=1e-10, rtol=0)
        with self.assertRaises(ValueError):
            map_active_integrals(0., np.zeros((3, 3)), np.zeros((4, 4, 4, 4)))

    def test_full_fci_quantifies_active_space_truncation(self):
        full = self.full_fci
        self.assertTrue(full["full_fci_converged"])
        self.assertAlmostEqual(full["full_fci_hartree"], -91.84392763770316, delta=1e-8)
        self.assertEqual(full["ci_vector_shape"], [330, 330])
        self.assertGreater(full["casci_minus_full_fci_hartree"], .05)
        self.assertLess(abs(full["full_fci_spin_squared"]), 1e-8)

    def test_invalid_geometry_and_controls(self):
        for value in [0., -1., np.nan, np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    geometry_angstrom(r_cn=value)
                with self.assertRaises(ValueError):
                    geometry_angstrom(r_hc=value)
        with self.assertRaises(ValueError):
            build_problem(grid_level=10)
        with self.assertRaises(ValueError):
            make_circuits(self.problem, layers=0)
        with self.assertRaises(ValueError):
            optimize_seed(self.problem, maxiter=0)

    def test_gatefabric_conserves_number_and_total_spin(self):
        shape, _, state = make_circuits(self.problem, layers=3)
        diagnostics = state_diagnostics(self.problem, state(np.random.default_rng(12).normal(size=shape)))
        self.assertAlmostEqual(diagnostics["particle_number"], 4, delta=1e-10)
        self.assertLess(abs(diagnostics["spin_squared"]), 1e-8)
        self.assertLess(diagnostics["sector_leakage"], 1e-12)

    def test_fixed_pi_gates_change_zero_weight_initialization(self):
        shape, _, state = make_circuits(self.problem, layers=6)
        prepared = np.asarray(state(np.zeros(shape)))
        self.assertAlmostEqual(abs(prepared[15])**2, 1., delta=1e-12)
        self.assertLess(abs(prepared[240])**2, 1e-12)

    def test_optimizer_accuracy_checkpoint_and_trace(self):
        validate_run(self.good_run)
        weights = np.asarray(self.good_run["best_parameters"])
        _, energy, _ = make_circuits(self.problem)
        self.assertAlmostEqual(float(energy(weights)), self.good_run["best_energy_hartree"], delta=1e-10)
        trace = self.good_run["energy_trace"]
        self.assertEqual(trace[0]["stage"], "initial")
        self.assertEqual(trace[-1]["stage"], "final")
        self.assertAlmostEqual(min(x["energy_hartree"] for x in trace), self.good_run["best_energy_hartree"], delta=1e-10)

    def test_failed_optimizer_retains_best_evaluated_parameters(self):
        checkpoint = np.asarray(self.good_run["best_parameters"]).ravel()
        def overshoot(objective, initial, **kwargs):
            objective(checkpoint)
            return SimpleNamespace(x=initial, success=False, message="budget exhausted", nit=1, nfev=1)
        with patch("hcn_benchmark.minimize", side_effect=overshoot):
            result = optimize_seed(self.problem, seed=5, maxiter=1)
        self.assertFalse(result["optimizer_success"])
        self.assertGreater(result["final_energy_hartree"], result["best_energy_hartree"]+.001)
        np.testing.assert_allclose(result["best_parameters"], self.good_run["best_parameters"], atol=1e-12, rtol=0)
        validate_run(result, require_accuracy=False)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            validate_run(result)

    def test_bad_physical_results_are_rejected(self):
        for key, value in [("particle_number", 3.), ("spin_squared", 2.),
                           ("sector_leakage", .1), ("best_energy_hartree", np.nan),
                           ("gap_hartree", -.01), ("best_energy_hartree", -90.)]:
            with self.subTest(key=key):
                result = dict(self.good_run)
                result[key] = value
                with self.assertRaises(ValueError):
                    validate_run(result, require_accuracy=False)

    def test_nonconverged_references_are_rejected(self):
        with patch("hcn_benchmark.scf.RHF") as rhf:
            rhf.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, "RHF did not converge"):
                build_problem()
        with patch("hcn_benchmark.mcscf.CASCI") as casci:
            casci.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, "CASCI did not converge"):
                build_problem()
        with patch("hcn_benchmark.dft.RKS") as rks:
            rks.return_value.converged = False
            with self.assertRaisesRegex(RuntimeError, "RKS-PBE did not converge"):
                build_problem()

    def test_adjoint_gradient_matches_finite_difference(self):
        shape, energy, _ = make_circuits(self.problem, layers=2)
        weights = qml.numpy.array(np.random.default_rng(4).normal(scale=.2, size=shape), requires_grad=True)
        _, gradient = value_and_grad(energy)(weights)
        index = np.unravel_index(np.argmax(np.abs(gradient)), shape)
        plus, minus = np.array(weights), np.array(weights)
        plus[index] += 1e-5
        minus[index] -= 1e-5
        finite = (float(energy(plus))-float(energy(minus)))/2e-5
        self.assertAlmostEqual(float(gradient[index]), finite, delta=1e-6)

    def test_pbe_grid_reference_control(self):
        control = pbe_grid_control()
        self.assertLess(abs(control["delta_hartree"]), 1e-6)

    def test_saved_review_coverage_and_fresh_references(self):
        evidence = json.loads((Path(__file__).parent / "benchmarks/hcn/results.json").read_text())
        self.assertEqual(evidence["profile"], "default")
        self.assertEqual([row["reference"]["r_cn_angstrom"] for row in evidence["scan"]],
                         [1.067, 1.167, 1.267])
        self.assertEqual(len(evidence["provenance"]["benchmark_source_sha256"]), 64)
        for row in evidence["scan"]:
            recorded = row["reference"]
            fresh = build_problem(recorded["r_cn_angstrom"]).reference
            for key in ["rhf_hartree", "casci_hartree", "rks_pbe_hartree", "sector_exact_hartree"]:
                self.assertAlmostEqual(recorded[key], fresh[key], delta=1e-8)
            self.assertEqual([run["seed"] for run in row["runs"]], [0, 1, 2])
            for run in row["runs"]:
                self.assertEqual((run["layers"], run["maxiter"]), (6, 300))
                validate_run(run)
        self.assertAlmostEqual(evidence["full_fci_control"]["full_fci_hartree"],
                               self.full_fci["full_fci_hartree"], delta=1e-8)

    def test_saved_parameters_replay_in_their_recorded_orbital_basis(self):
        evidence = json.loads((Path(__file__).parent / "benchmarks/hcn/results.json").read_text())
        for row in evidence["scan"]:
            tensors = row["active_integrals"]
            self.assertEqual(tensors["orbital_indices"], [5, 6, 7, 8])
            one = np.asarray(tensors["one_body_hartree"])
            two = np.asarray(tensors["two_body_chemist_hartree"])
            core = tensors["core_energy_hartree"]
            physicist, fermion, operator, matrix = map_active_integrals(core, one, two)
            replay = replace(self.problem, reference=row["reference"], matrix=matrix,
                             one_body=one, two_body_chemist=two,
                             two_body_physicist=physicist, core_energy=core,
                             fermionic=fermion, hamiltonian=operator)
            exact = np.linalg.eigvalsh(matrix[np.ix_(replay.sector, replay.sector)])[0]
            self.assertAlmostEqual(exact, row["reference"]["casci_hartree"], delta=1e-8)
            for run in row["runs"]:
                with self.subTest(distance=row["reference"]["r_cn_angstrom"], seed=run["seed"]):
                    shape, _, state = make_circuits(replay, run["layers"])
                    parameters = np.asarray(run["best_parameters"])
                    self.assertEqual(parameters.shape, shape)
                    actual = state_diagnostics(replay, state(parameters))
                    for key in ["best_energy_hartree", "gap_hartree", "energy_variance_hartree2",
                                "particle_number", "spin_squared", "sector_leakage"]:
                        self.assertAlmostEqual(actual[key], run[key], delta=1e-10)


if __name__ == "__main__":
    unittest.main()
