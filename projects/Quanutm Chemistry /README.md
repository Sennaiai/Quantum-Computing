# Quantum chemistry

Three independent notebook studies, refined one at a time:

| Study | Focus | Status |
|---|---|---|
| [H₂](Quantum_Simulation_of_the_Hydrogen_Molecule_H2_Hamiltonian.ipynb) | Integrals → fermions → qubits; VQE, FCI, RHF and DFT benchmarks | Validated locally; [run instructions](README_H2.md) |
| [HCN](Chemistry_HCN_VQE_Study.ipynb) | Active-space VQE and C–N potential-energy scans | Next refinement; existing results require revalidation |
| [Active-space scaling](Molecular_Hamiltonian_Construction_Active_Space_Scaling.ipynb) | Hamiltonian construction and resource/accuracy tradeoffs | Refinement pending |

Each notebook must show its molecular model, units, basis/active space,
Hamiltonian construction, relevant quantum/classical methods, and numerical
checks. Solver agreement and physical-model accuracy are assessed separately.
See [the refinement roadmap](REFINEMENT_PLAN.md) for pending work.
