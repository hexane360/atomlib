from __future__ import annotations

import typing as t

import polars

from ..core import AtomCell
from ..util import FileOrPath, open_file

def write_qe(cell: AtomCell, f: FileOrPath, pseudo: t.Mapping[str, str]):
    if not isinstance(cell, AtomCell):
        raise TypeError("'qe' format requires an AtomCell.")

    atoms = cell.wrap().get_atoms('cell_frac').with_mass()

    types = atoms.select(('symbol', 'mass')).unique(subset='symbol')
    types = types.with_columns(polars.col('symbol').apply(lambda sym: pseudo[str(sym)]).alias('pot'))

    with open_file(f, 'w') as f:
        print(f"&SYSTEM ibrav=0 nat={len(atoms)} ntyp={len(types)}", file=f)

        ortho = cell.cell.get_transform('local', 'cell_box').to_linear().inner
        print(f"\nCELL_PARAMETERS angstrom", file=f)
        for row in ortho.T:
            print(f"  {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}", file=f)

        print(f"\nATOMIC_SPECIES", file=f)
        for (symbol, mass, pot) in types.select(('symbol', 'mass', 'pot')).rows():
            print(f"{symbol:>4} {mass:10.3f}  {pot}", file=f)

        print(f"\nATOMIC_POSITIONS crystal", file=f)
        for (symbol, x, y, z) in atoms.select(('symbol', 'x', 'y', 'z')).rows():
            print(f"{symbol:>4} {x:.8f} {y:.8f} {z:.8f}", file=f)