from __future__ import annotations

from pathlib import Path
from io import IOBase
import logging
import typing as t

import numpy
import polars

from .cif import CIF, CIFDataBlock
from .xyz import XYZ, XYZFormat
from .xsf import XSF
from .cfg import CFG
from .mslice import write_mslice, read_mslice
from .lmp import write_lmp
from .qe import write_qe

from ..atoms import Atoms, HasAtoms
from ..atomcell import AtomCell, Cell
from ..types import to_vec3
from ..transform import LinearTransform3D
from ..elem import get_sym, get_elem
from ..util import FileOrPath

FileType = t.Literal['cif', 'xyz', 'xsf', 'cfg', 'lmp', 'mslice', 'qe']


def read_cif(f: t.Union[FileOrPath, CIF, CIFDataBlock], block: t.Union[int, str, None] = None) -> HasAtoms:
    """
    Read a structure from a CIF file.

    If `block` is specified, read data from the given block of the CIF file (index or name).
    """

    if isinstance(f, (CIF, CIFDataBlock)):
        cif = f
    else:
        cif = CIF.from_file(f)

    if isinstance(cif, CIF):
        if len(cif) == 0:
            raise ValueError("No data present in CIF file.")
        if block is None:
            if len(cif) > 1:
                logging.warn("Multiple blocks present in CIF file. Defaulting to reading first block.")
            cif = cif.data_blocks[0]
        else:
            cif = cif.get_block(block)

    logging.debug("cif data: %r", cif.data_dict)

    # TODO: support atom_site_Cartn_[xyz]
    df = cif.stack_tags('atom_site_fract_x', 'atom_site_fract_y', 'atom_site_fract_z',
                        'atom_site_type_symbol', 'atom_site_label', 'atom_site_occupancy',
                        'atom_site_U_iso_or_equiv', 'atom_site_B_iso_or_equiv',
                        rename=('x', 'y', 'z', 'symbol', 'label', 'frac_occupancy', 'wobble', 'wobble_B'),
                        required=(True, True, True, False, False, False, False, False))
    if 'wobble_B' in df.columns:
        if 'wobble' in df.columns:
            raise ValueError("CIF file specifies both 'atom_site_U_iso_or_equiv' and 'atom_site_B_iso_or_equiv'")
        df = df.rename({'wobble_B': 'wobble'}) \
            .with_columns(polars.col('wobble') * (3./8. / numpy.pi**2))
    if 'symbol' not in df.columns:
        if 'label' not in df.columns:
            raise ValueError("Tag 'atom_site_type_symbol' or 'atom_site_label' missing from CIF file")
        # infer symbol from label, insert at beginning
        df = df.insert_column(0, get_sym(get_elem(df['label'])))
    atoms = Atoms(df)

    # parse and apply symmetry
    sym_atoms = []
    for sym in cif.get_symmetry():
        sym_atoms.append(atoms.transform(sym))

    s = '\n'.join(map(str, sym_atoms))
    print(f"sym_atoms:\n{s}")
    print(f"atoms: {atoms!s}")

    if len(sym_atoms) > 0:
        atoms = Atoms.concat(sym_atoms)._wrap().deduplicate()

    if (cell_size := cif.cell_size()) is not None:
        cell_size = to_vec3(cell_size)
        if (cell_angle := cif.cell_angle()) is not None:
            # degrees to radians
            cell_angle = to_vec3(cell_angle) * numpy.pi/180.
        return AtomCell.from_unit_cell(atoms, cell_size, cell_angle, frame='cell_frac')
    return Atoms(atoms)


def write_cif(atoms: t.Union[HasAtoms, CIF, CIFDataBlock], f: FileOrPath):
    """Write a structure to an XSF file."""
    if isinstance(atoms, (CIF, CIFDataBlock)):
        cif = atoms
    elif isinstance(atoms, AtomCell):
        cif = CIF((CIFDataBlock.from_atomcell(atoms),))
    else:
        cif = CIF((CIFDataBlock.from_atoms(atoms),))

    cif.write(f)


def read_xyz(f: t.Union[FileOrPath, XYZ]) -> HasAtoms:
    """Read a structure from an XYZ file."""
    if isinstance(f, XYZ):
        xyz = f
    else:
        xyz = XYZ.from_file(f)

    atoms = Atoms(xyz.atoms)

    if (cell_matrix := xyz.cell_matrix()) is not None:
        cell = Cell.from_ortho(LinearTransform3D(cell_matrix), pbc=xyz.pbc())
        return AtomCell(atoms, cell, frame='local')
    return Atoms(atoms)


def write_xyz(atoms: t.Union[HasAtoms, XYZ], f: FileOrPath, fmt: XYZFormat = 'exyz'):
    if not isinstance(atoms, XYZ):
        atoms = XYZ.from_atoms(atoms)
    atoms.write(f, fmt)


def read_xsf(f: t.Union[FileOrPath, XSF]) -> HasAtoms:
    """Read a structure from a XSF file."""
    if isinstance(f, XSF):
        xsf = f
    else:
        xsf = XSF.from_file(f)

    atoms = xsf.get_atoms()
    atoms = atoms.with_columns(get_sym(atoms['elem']))

    if (primitive_cell := xsf.primitive_cell) is not None:
        cell = Cell.from_ortho(primitive_cell, pbc=xsf.get_pbc())
        return AtomCell(atoms, cell, frame='local')
    return Atoms(atoms)


def write_xsf(atoms: t.Union[HasAtoms, XSF], f: FileOrPath):
    """Write a structure to an XSF file."""
    if isinstance(atoms, XSF):
        xsf = atoms
    elif isinstance(atoms, AtomCell):
        xsf = XSF.from_cell(atoms)
    else:
        xsf = XSF.from_atoms(atoms)

    xsf.write(f)


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


def write_cfg(atoms: t.Union[HasAtoms, CFG], f: FileOrPath):
    if not isinstance(atoms, CFG):
        atoms = CFG.from_atoms(atoms)
    atoms.write(f)


ReadFunc = t.Callable[[FileOrPath], HasAtoms]
_READ_TABLE: t.Mapping[FileType, t.Optional[ReadFunc]] = {
    'cif': read_cif,
    'xyz': read_xyz,
    'xsf': read_xsf,
    'cfg': read_cfg,
    'mslice': read_mslice,
    'lmp': None,
    'qe': None
}

WriteFunc = t.Callable[[HasAtoms, FileOrPath], None]
_WRITE_TABLE: t.Mapping[FileType, t.Optional[WriteFunc]] = {
    'cif': write_cif,
    'xyz': write_xyz,
    'xsf': write_xsf,
    'cfg': write_cfg,
    'lmp': write_lmp,
    # some methods only take HasAtomCell. These must be checked at runtime
    'mslice': t.cast(WriteFunc, write_mslice),
    'qe': t.cast(WriteFunc, write_qe),
}


@t.overload
def read(path: FileOrPath, ty: FileType) -> HasAtoms:
    ...

@t.overload
def read(path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None) -> HasAtoms:
    ...

def read(path: FileOrPath, ty: t.Optional[FileType] = None) -> HasAtoms:
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
def write(atoms: HasAtoms, path: FileOrPath, ty: FileType):
    ...

@t.overload
def write(atoms: HasAtoms, path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None):
    ...

def write(atoms: HasAtoms, path: FileOrPath, ty: t.Optional[FileType] = None):
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
    'write', 'write_xyz', 'write_xsf', 'write_cfg', 'write_lmp', 'write_mslice', 'write_qe',
]
