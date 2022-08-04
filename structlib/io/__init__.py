from __future__ import annotations

from pathlib import Path
import typing as t

import polars

from .cif import CIF
from .xyz import XYZ
from .xsf import XSF

from ..core import AtomCollection, AtomFrame, Cell, PeriodicCell, Lattice
from ..vec import Vec3
from ..transform import LinearTransform
from ..elem import get_elem, get_sym
from ..util import FileOrPath


def read(path: t.Union[str, Path]) -> AtomCollection:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.cif':
        return read_cif(path)
    if ext == '.xyz':
        return read_xyz(path)
    if ext == '.xsf':
        return read_xsf(path)
    raise ValueError(f"Unknown file type '{path.suffix}'")


def read_cif(f: t.Union[FileOrPath, CIF]) -> AtomCollection:
    if isinstance(f, CIF):
        cif = f
    else:
        cif = list(CIF.from_file(f))
        if len(cif) == 0:
            raise ValueError("No data in CIF file.")
        if len(cif) > 1:
            raise NotImplementedError()
        [cif] = cif

    df = polars.DataFrame(cif.stack_tags('atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z',
                                         'atom_site_type_symbol', 'atom_site_occupancy'))
    df.columns = ['x','y','z','symbol','frac_occupancy']
    df = df.with_column(get_elem(df['symbol']))

    atoms = AtomFrame(df)

    if (cell_size := cif.cell_size()) is not None:
        cell_size = Vec3.make(cell_size)
        if (cell_angle := cif.cell_angle()) is not None:
            cell_angle = Vec3.make(cell_angle)
            return Lattice(atoms, cell_size, cell_angle)
        return PeriodicCell(atoms, cell_size)
    return Cell(atoms)


def read_xyz(f: t.Union[FileOrPath, XYZ]) -> AtomCollection:
    if isinstance(f, XYZ):
        xyz = f
    else:
        xyz = XYZ.from_file(f)

    df = xyz.atoms.with_column(get_elem(xyz.atoms['symbol']))
    atoms = AtomFrame(df)

    if (cell_matrix := xyz.cell_matrix()) is not None:
        return Lattice(atoms, ortho=LinearTransform(cell_matrix))
    return Cell(atoms)


def read_xsf(f: t.Union[FileOrPath, XSF]) -> AtomCollection:
    if isinstance(f, XSF):
        xsf = f
    else:
        xsf = XSF.from_file(f)

    atoms = xsf.get_atoms()
    atoms = atoms.with_column(get_sym(atoms['elem']))

    if (primitive_cell := xsf.primitive_cell) is not None:
        return Lattice(atoms, ortho=primitive_cell)
    return Cell(atoms)



__all__ = [
    'CIF', 'XYZ', 'XSF', 'read', 'read_cif', 'read_xyz', 'read_xsf',
]