"""Optional geometry regeneration; default studies consume fixed JSON snapshots.

Requires RDKit 2026.09.2 in a separate environment. UFF geometries are modeling
inputs, not experimental structures or quantum-optimized stationary points.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/scaling/regenerated_geometries.json'))
    args = parser.parse_args()
    if rdBase.rdkitVersion != '2026.09.2':
        raise RuntimeError('Pinned snapshot generator requires RDKit 2026.09.2')
    data = json.loads((Path(__file__).parent/'data/scaling_geometries.json').read_text())
    for name in ['carbonic_acid', 'bicarbonate']:
        spec = data['molecules'][name]
        mol = Chem.AddHs(Chem.MolFromSmiles(spec['provenance']['smiles']))
        params = AllChem.ETKDGv3()
        params.randomSeed = 1729
        embedding = AllChem.EmbedMolecule(mol, params)
        if embedding != 0 or not AllChem.UFFHasAllMoleculeParams(mol):
            raise RuntimeError(f'{name}: embedding or force-field parameter failure')
        optimization = AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        if optimization != 0:
            raise RuntimeError(f'{name}: UFF did not converge')
        spec['coordinates_angstrom'] = np.asarray(mol.GetConformer().GetPositions()).tolist()
        spec['provenance'].update(embedding_status=embedding, uff_status=optimization)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n')
    print('Saved geometry snapshot:', args.output)


if __name__ == '__main__':
    main()
