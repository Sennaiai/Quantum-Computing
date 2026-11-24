# Quantum chemistry refinement roadmap

Refine the notebooks independently and in sequence: H₂, HCN, then active-space
scaling. Preserve each study's scientific question and show the relevant
construction and validation steps directly in its notebook. References:
[HHL validation baseline](../HHL/README.md) and broad draft
[PR #3](https://github.com/Sennaiai/Quantum-Computing/pull/3).

## Required scientific chain

1. Geometry, units, charge, spin, basis, and justified active space.
2. Molecular integrals, nuclear/frozen-core constants, fermionic Hamiltonian,
   and explicitly specified qubit mapping.
3. Reference state, ansatz, objective, optimizer, and convergence evidence when
   the study solves for energies; resource metrics when it studies construction.
4. Matched classical references, conserved-quantity checks, and numerical
   tolerances distinct from physical-model accuracy.
5. Reproducible execution, saved evidence, bounded resource requirements, and
   limitations supported by the calculations.

## H₂: validated locally

The notebook now exposes the full integral-to-circuit workflow. See
[README_H2.md](README_H2.md) and [H2_VALIDATION.md](H2_VALIDATION.md).
Remote Linux CI remains pending; human review is required before merge/release.

## HCN: validated locally, review pending

The notebook now exposes a π/π* CAS(4,4) construction, spin-conserving VQE,
matched-sector CASCI checks, and a short fixed-H–C potential-energy scan.
Full STO-3G FCI at one geometry separates active-space truncation from solver
error. The pinned differentiable-HF backend was rejected by reference checks;
PBE grids were refined after a failed numerical control. See
[README_HCN.md](README_HCN.md) and [HCN_VALIDATION.md](HCN_VALIDATION.md).
Review HCN before committing its refinement or starting active-space scaling.

## Active-space scaling: pending

Replace hand-entered results with computed summaries; validate units, electron
counts, active-space capacity, and geometry generation. Distinguish the Pauli
identity coefficient from nuclear/frozen-core energy. Compare measured operator
resources with tractable small-space energy references. Large constructions
must be explicit opt-ins; avoid exponentially large dense matrices/statevectors.
