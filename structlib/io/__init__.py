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
from .lmp import write_lmp

from ..core import AtomCollection, Atoms, SimpleAtoms, AtomCell
from ..types import to_vec3
from ..transform import LinearTransform3D
from ..elem import get_sym
from ..util import FileOrPath

FileType = t.Union[t.Literal['cif'], t.Literal['xyz'], t.Literal['xsf'], t.Literal['cfg'], t.Literal['lmp'], t.Literal['mslice']]


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
    atoms = Atoms(df)

    # parse and apply symmetry
    sym_atoms = []
    for sym in cif.get_symmetry():
        sym_atoms.append(atoms.transform(sym))

    if len(sym_atoms) > 0:
        atoms = AtomCell.from_ortho(Atoms.concat(sym_atoms), LinearTransform3D()) \
            .wrap().atoms.deduplicate()

    if (cell_size := cif.cell_size()) is not None:
        cell_size = to_vec3(cell_size)
        if (cell_angle := cif.cell_angle()) is not None:
            cell_angle = to_vec3(cell_angle) * numpy.pi/180.
        return AtomCell.from_unit_cell(atoms, cell_size, cell_angle, frame='cell_frac')
    return SimpleAtoms(atoms)


def read_xyz(f: t.Union[FileOrPath, XYZ]) -> AtomCollection:
    """Read a structure from an XYZ file."""
    if isinstance(f, XYZ):
        xyz = f
    else:
        xyz = XYZ.from_file(f)

    atoms = Atoms(xyz.atoms)

    if (cell_matrix := xyz.cell_matrix()) is not None:
        return AtomCell.from_ortho(atoms, LinearTransform3D(cell_matrix))
    return SimpleAtoms(atoms)


def read_xsf(f: t.Union[FileOrPath, XSF]) -> AtomCollection:
    """Read a structure from a XSF file."""
    if isinstance(f, XSF):
        xsf = f
    else:
        xsf = XSF.from_file(f)

    atoms = xsf.get_atoms()
    atoms = atoms.with_columns(get_sym(atoms['elem']))

    if (primitive_cell := xsf.primitive_cell) is not None:
        return AtomCell.from_ortho(atoms, primitive_cell)
        #return cell.transform_atoms(cell.ortho, 'local')  # transform to real-space coordinates
    return SimpleAtoms(atoms)


def read_cfg(f: t.Union[FileOrPath, CFG]) -> AtomCell:
    """Read a structure from an AtomEye CFG file."""
    if isinstance(f, CFG):
        cfg = f
    else:
        cfg = CFG.from_file(f)

    ortho = cfg.cell
    if cfg.transform is not None:
        ortho = cfg.transform @ ortho

    if cfg.length_scale is not None:
        ortho = ortho.scale(all=cfg.length_scale)

    if cfg.eta is not None:
        m = numpy.eye(3) + 2. * cfg.eta.inner
        # matrix sqrt using eigenvals, eigenvecs
        eigenvals, eigenvecs = numpy.linalg.eigh(m)
        sqrtm = (eigenvecs * numpy.sqrt(eigenvals)) @ eigenvecs.T
        ortho = LinearTransform3D(sqrtm) @ ortho

    frame = Atoms(cfg.atoms).transform(ortho, transform_velocities=True)
    return AtomCell.from_ortho(frame, ortho)


def write_xsf(atoms: t.Union[AtomCollection, XSF], f: FileOrPath):
    """Write a structure to an XSF file."""
    if isinstance(atoms, XSF):
        xsf = atoms
    elif isinstance(atoms, AtomCell):
        xsf = XSF.from_cell(atoms)
    else:
        xsf = XSF.from_atoms(atoms)

    xsf.write(f)


ReadFunc = t.Callable[[FileOrPath], AtomCollection]
_READ_TABLE: t.Mapping[FileType, t.Optional[ReadFunc]] = {
    'cif': read_cif,
    'xyz': read_xyz,
    'xsf': read_xsf,
    'cfg': read_cfg,
    'mslice': None,
    'lmp': None,
}

WriteFunc = t.Callable[[AtomCollection, FileOrPath], None]
_WRITE_TABLE: t.Mapping[FileType, t.Optional[WriteFunc]] = {
    'cif': None,
    'xyz': None,
    'xsf': write_xsf,
    'cfg': None,
    'mslice': t.cast(WriteFunc, write_mslice),
    'lmp': write_lmp,
}


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
    if ty is None:
        if isinstance(path, (t.IO, IOBase)):
            try:
                name = path.name  # type: ignore
                if name is None:
                    raise AttributeError()
                ext = Path(name).suffix
            except AttributeError:
                raise TypeError("read() must be passed a file-type when reading an already-open file.") from None
        else:
            name = Path(path).name
            ext = Path(path).suffix

        if len(ext) == 0:
            raise ValueError(f"Can't infer extension for file '{name}'")

        return read(path, t.cast(FileType, ext))

    ty_strip = str(ty).lstrip('.').lower()
    try:
        read_fn = _READ_TABLE[t.cast(FileType, ty_strip)]
    except KeyError:
        raise ValueError(f"Unknown file type '{ty}'") from None
    if read_fn is None:
        raise ValueError(f"Reading is not supported for file type '{ty_strip}'")
    return read_fn(path)


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
                name = path.name  # type: ignore
                if name is None:
                    raise AttributeError()
                ext = Path(name).suffix
            except AttributeError:
                raise TypeError("write() must be passed a file-type when reading an already-open file.") from None
        else:
            name = Path(path).name
            ext = Path(path).suffix

        if len(ext) == 0:
            raise ValueError(f"Can't infer extension for file '{name}'")

        return write(atoms, path, t.cast(FileType, ext))

    ty_strip = str(ty).lstrip('.').lower()
    try:
        write_fn = _WRITE_TABLE[t.cast(FileType, ty_strip)]
    except KeyError:
        raise ValueError(f"Unknown file type '{ty}'") from None
    if write_fn is None:
        raise ValueError(f"Writing is not supported for file type '{ty_strip}'")

    return write_fn(atoms, path)


__all__ = [
    'CIF', 'XYZ', 'XSF', 'CFG',
    'read', 'read_cif', 'read_xyz', 'read_xsf', 'read_cfg',
    'write', 'write_xsf', 'write_lmp', 'write_mslice',
]
