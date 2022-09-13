from __future__ import annotations

from pathlib import Path
from io import IOBase
import logging
import typing as t

import numpy

from .cif import CIF
from .xyz import XYZ
from .xsf import XSF
from .cfg import CFG
from .mslice import write_mslice

from ..core import AtomCollection, AtomFrame, SimpleAtoms, AtomCell
from ..vec import Vec3
from ..transform import LinearTransform
from ..elem import get_elem, get_sym
from ..util import FileOrPath

FileType = t.Union[t.Literal['cif'], t.Literal['xyz'], t.Literal['xsf']]


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
    logging.debug("cif data: %r", cif.data)

    df = cif.stack_tags('atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z',
                        'atom_site_type_symbol', 'atom_site_occupancy',
                        rename=('x', 'y', 'z', 'symbol', 'frac_occupancy'))
    df = df.with_column(get_elem(df['symbol']))

    atoms = AtomFrame(df)

    if (cell_size := cif.cell_size()) is not None:
        cell_size = Vec3.make(cell_size)
        if (cell_angle := cif.cell_angle()) is not None:
            cell_angle = Vec3.make(cell_angle) * numpy.pi/180.
        return AtomCell(atoms, cell_size, cell_angle, frac=True)
    return SimpleAtoms(atoms)


def read_xyz(f: t.Union[FileOrPath, XYZ]) -> AtomCollection:
    """Read a structure from an XYZ file."""
    if isinstance(f, XYZ):
        xyz = f
    else:
        xyz = XYZ.from_file(f)

    df = xyz.atoms.with_column(get_elem(xyz.atoms['symbol']))
    atoms = AtomFrame(df)

    if (cell_matrix := xyz.cell_matrix()) is not None:
        return AtomCell(atoms, ortho=LinearTransform(cell_matrix))
    return SimpleAtoms(atoms)


def read_xsf(f: t.Union[FileOrPath, XSF]) -> AtomCollection:
    """Read a structure from a XSF file."""
    if isinstance(f, XSF):
        xsf = f
    else:
        xsf = XSF.from_file(f)

    atoms = xsf.get_atoms()
    atoms = atoms.with_column(get_sym(atoms['elem']))

    if (primitive_cell := xsf.primitive_cell) is not None:
        return AtomCell(atoms, ortho=primitive_cell)
        #return cell.transform_atoms(cell.ortho, 'local')  # transform to real-space coordinates
    return SimpleAtoms(atoms)


def read_cfg(f: t.Union[FileOrPath, CFG]) -> AtomCell:
    """Read a structure from an AtomEye CFG file."""
    if isinstance(f, CFG):
        cfg = f
    else:
        cfg = CFG.from_file(f)

    ortho = cfg.cell
    if cfg.eta is not None:
        ortho = LinearTransform(LinearTransform().inner + 2. * cfg.eta.inner) @ ortho
    if cfg.transform is not None:
        ortho = cfg.transform @ ortho

    # TODO transform velocities to local coordinates?

    return AtomCell(
        AtomFrame(cfg.atoms), ortho=cfg.cell, frac=True
    )


def write_xsf(atoms: t.Union[AtomCollection, XSF], f: FileOrPath):
    """Write a structure to an XSF file."""
    if isinstance(atoms, XSF):
        xsf = atoms
    elif isinstance(atoms, AtomCell):
        xsf = XSF.from_cell(atoms)
    else:
        xsf = XSF.from_atoms(atoms)

    xsf.write(f)


@t.overload
def read(path: FileOrPath, ty: FileType) -> AtomCollection:
    ...

@t.overload
def read(path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None) -> AtomCollection:
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
        if ty_strip == 'cfg':
            return read_cfg(path)
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


@t.overload
def write(atoms: AtomCollection, path: FileOrPath, ty: FileType):
    ...

@t.overload
def write(atoms: AtomCollection, path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None):
    ...

def write(atoms: AtomCollection, path: FileOrPath, ty: t.Optional[FileType] = None):
    """
    Write this structure to a file.

    A file type may be specified using `ty`.
    If no `ty` is specified, it is inferred from the path's extension.
    """

    if ty is None:
        if isinstance(path, (t.IO, IOBase)):
            try:
                ext = Path(path.name).suffix  # type: ignore
                if len(ext) == 0:
                    raise AttributeError()
            except AttributeError:
                raise TypeError("read() must be passed a file-type when reading an already-open file.") from None
        else:
            ext = Path(path).suffix

        return write(atoms, path, t.cast(FileType, ext))

    ty_strip = str(ty).lstrip('.').lower()

    if ty_strip in ('cif', 'xyz', 'cfg'):
        raise NotImplementedError()
    if ty_strip == 'xsf':
        return write_xsf(atoms, path)
    if ty_strip == 'mslice':
        if not isinstance(atoms, AtomCell):
            raise TypeError("mslice format requires an AtomCell.")
        return write_mslice(atoms, path)
    raise ValueError(f"Unknown file type '{ty}'")


__all__ = [
    'CIF', 'XYZ', 'XSF', 'CFG',
    'read', 'read_cif', 'read_xyz', 'read_xsf', 'read_cfg',
    'write_xsf', 'write_mslice',
]
