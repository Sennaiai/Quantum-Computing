"""Optional check of the pinned PennyLane DHF backend against converged PySCF.

This diagnostic is separate from the validated PySCF-integral study. A large
HF discrepancy rejects this backend for this geometry; it is not a chemistry
result. No quantum optimization or large matrix allocation is performed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pennylane as qml

from hcn_benchmark import build_problem, geometry_angstrom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/hcn/dhf_diagnostic.json'))
    args = parser.parse_args()
    reference = build_problem().reference['rhf_hartree']
    coordinates = qml.numpy.array(geometry_angstrom()/0.529177210903, requires_grad=False)
    molecule = qml.qchem.Molecule(['H', 'C', 'N'], coordinates, basis_name='sto-3g')
    # Use the public mean-field energy routine with the pinned default SCF budget.
    candidate = float(np.asarray(qml.qchem.hf_energy(molecule)()).item())
    difference = candidate-reference
    result = dict(pennylane_version=qml.__version__, pyscf_rhf_hartree=reference,
                  dhf_hartree=candidate, difference_hartree=difference,
                  agreement_tolerance_hartree=1e-6, backend_accepted=abs(difference)<1e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
