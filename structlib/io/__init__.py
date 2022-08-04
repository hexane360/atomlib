from __future__ import annotations

from pathlib import Path
from io import IOBase
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

FileType = t.Union[t.Literal['cif'], t.Literal['xyz'], t.Literal['xsf']]


@t.overload
def read(path: FileOrPath, ty: FileType) -> AtomCollection:
    ...

@t.overload
def read(path: t.Union[str, Path], ty: t.Literal[None] = None) -> AtomCollection:
    ...

def read(path: FileOrPath, ty: t.Optional[FileType] = None) -> AtomCollection:
    """
    Read a structure from a file.

    Currently, supported file types are 'cif', 'xyz', and 'xsf'.
    If no `ty` is specified, it is inferred from the file's extension.
    """
    if ty is not None:
        ty_strip = str(ty).lstrip('.').lower()
        if ty_strip == 'cif':
            return read_cif(path)
        if ty_strip == 'xyz':
            return read_xyz(path)
        if ty_strip == 'xsf':
            return read_xsf(path)
        raise ValueError(f"Unknown file type '{ty}'")

    if isinstance(path, (t.IO, IOBase)):
        try:
            ext = Path(path.name).suffix  # type: ignore
            if len(ext) == 0:
                raise AttributeError()
        except AttributeError:
            raise TypeError("read() must be passed a file-type when reading an already-open file.") from None
    else:
        ext = Path(path).suffix

    return read(path, t.cast(FileType, ext))


def read_cif(f: t.Union[FileOrPath, CIF]) -> AtomCollection:
    """Read a structure from a CIF file."""

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
    """Read a structure from an XYZ file."""
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
    """Read a structure from a XSF file."""
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