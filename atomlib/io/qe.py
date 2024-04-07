from __future__ import annotations

import typing as t

import polars

from ..atoms import _get_symbol_mapping
from ..atomcell import HasAtomCell
from ..util import FileOrPath, open_file


def write_qe(atomcell: HasAtomCell, f: FileOrPath, pseudo: t.Optional[t.Mapping[str, str]] = None):
    """
    Write a structure to a Quantum Espresso pw.x file.

    Args:
      atomcell: Structure to write
      f: File or path to write to
      pseudo: Mapping from atom symbol
    """
    if not isinstance(atomcell, HasAtomCell):
        raise TypeError("'qe' format requires an AtomCell.")

    atoms = atomcell.wrap().get_atoms('cell_box').with_mass()

    types = atoms.select(('symbol', 'mass')).unique(subset='symbol').sort('mass')
    if pseudo is not None:
        types = types.with_columns(_get_symbol_mapping(types, pseudo, ty=polars.Utf8).alias('pot'))
    else:
        types = types.with_columns((polars.col('symbol') + polars.lit('.UPF')).alias('pot'))
        #types = types.with_columns(polars.col('symbol').apply(lambda sym: f"{sym}.UPF").alias('pot'))

    with open_file(f, 'w') as f:
        print(f"""\
&SYSTEM
  ibrav=0,
  nat={len(atoms)},
  ntyp={len(types)}
/""", file=f)

        ortho = atomcell.get_transform('local', 'cell_box').to_linear().inner
        print(f"\nCELL_PARAMETERS angstrom", file=f)
        for row in ortho.T:
            print(f"  {row[0]:12.8f} {row[1]:12.8f} {row[2]:12.8f}", file=f)

        print(f"\nATOMIC_SPECIES", file=f)
        for (symbol, mass, pot) in types.select(('symbol', 'mass', 'pot')).rows():
            print(f"{symbol:>4} {mass:10.3f}  {pot}", file=f)

        print(f"\nATOMIC_POSITIONS crystal", file=f)
        for (symbol, (x, y, z)) in atoms.select(('symbol', 'coords')).rows():
            print(f"{symbol:>4} {x:.8f} {y:.8f} {z:.8f}", file=f)

        print(file=f)  # allows for easy concatenation