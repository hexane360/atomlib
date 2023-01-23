from __future__ import annotations

import abc
from dataclasses import dataclass, fields, replace
import warnings
from pathlib import Path
import copy
import typing as t

import numpy
import polars

from .bbox import BBox
from .types import VecLike, Vec3, to_vec3
from .transform import LinearTransform, AffineTransform, Transform, IntoTransform
from .cell import CoordinateFrame, Cell
from .atoms import Atoms, AtomSelection, IntoAtoms


if t.TYPE_CHECKING:
    from .io import CIF, XYZ, XSF, CFG, FileOrPath, FileType  # pragma: no cover
    from .io.mslice import MSliceTemplate                     # pragma: no cover


AtomCollectionT = t.TypeVar('AtomCollectionT', bound='AtomCollection')
AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')
SimpleAtomsT = t.TypeVar('SimpleAtomsT', bound='SimpleAtoms')


class AtomCollection(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    @abc.abstractmethod
    def transform(self: AtomCollectionT, transform: AffineTransform, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Transforms cell boxes in addition to atoms.
        """
        ...

    @abc.abstractmethod
    def transform_atoms(self: AtomCollectionT, transform: IntoTransform, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Never transforms cell boxes.
        """
        ...

    @abc.abstractmethod
    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        ...

    @abc.abstractmethod
    def bbox(self) -> BBox:
        """Return the bounding box of self, in global coordinates."""
        ...

    @abc.abstractmethod
    def clone(self: AtomCollectionT) -> AtomCollectionT:
        ...

    @abc.abstractmethod
    def _str_parts(self) -> t.Iterable[t.Any]:
        ...

    @abc.abstractmethod
    def _replace_atoms(self: AtomCollectionT, atoms: Atoms, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        ...

    def __repr__(self) -> str:
        return "\n".join(map(str, self._str_parts()))

    __str__ = __repr__

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @t.overload
    @staticmethod
    def read(path: FileOrPath, ty: FileType) -> AtomCollection:
        ...

    @t.overload
    @staticmethod
    def read(path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None) -> AtomCollection:
        ...

    @staticmethod
    def read(path, ty=None) -> AtomCollection:
        """
        Read a structure from a file.

        Currently, supported file types are 'cif', 'xyz', and 'xsf'.
        If no `ty` is specified, it is inferred from the file's extension.
        """
        return io.read(path, ty)

    @staticmethod
    def read_cif(f: t.Union[FileOrPath, CIF]) -> AtomCollection:
        """Read a structure from a CIF file."""
        return io.read_cif(f)

    @staticmethod
    def read_xyz(f: t.Union[FileOrPath, XYZ]) -> AtomCollection:
        """Read a structure from an XYZ file."""
        return io.read_xyz(f)

    @staticmethod
    def read_xsf(f: t.Union[FileOrPath, XSF]) -> AtomCollection:
        """Read a structure from an XSF file."""
        return io.read_xsf(f)

    @staticmethod
    def read_cfg(f: t.Union[FileOrPath, CFG]) -> AtomCell:
        """Read a structure from a CFG file."""
        return io.read_cfg(f)

    @t.overload
    def write(self, path: FileOrPath, ty: FileType):
        ...

    @t.overload
    def write(self, path: t.Union[str, Path, t.TextIO], ty: t.Literal[None] = None):
        ...

    def write(self, path, ty=None):
        """
        Write this structure to a file.

        A file type may be specified using `ty`.
        If no `ty` is specified, it is inferred from the path's extension.
        """
        io.write(self, path, ty)


@dataclass(init=False, repr=False, frozen=True)
class SimpleAtoms(AtomCollection):
    """
    Cell of atoms with no known structure or periodicity.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    def __init__(self, atoms: IntoAtoms):
        object.__setattr__(self, 'atoms', Atoms(atoms))

    def bbox(self) -> BBox:
        """Get this structure's bounding box."""
        return self.atoms.bbox()

    def _check_allowed_frame(self, frame: CoordinateFrame):
        if frame.lower() not in ('local', 'global'):
            raise ValueError("Can't use 'cell'/'ortho' coordinate frames when box is unknown.")

    def with_bounds(self, cell_size: t.Optional[VecLike] = None, cell_origin: t.Optional[VecLike] = None) -> AtomCell:
        """
        Return a periodic cell with the given orthogonal cell dimensions.

        If cell_size is not specified, it will be assumed (and may be incorrect).
        """
        if cell_size is None:
            warnings.warn("Cell boundary unknown. Defaulting to cell BBox")
            cell_size = self.atoms.bbox().size
            cell_origin = self.atoms.bbox().min

        # TODO test this origin code
        cell = Cell.from_unit_cell(cell_size)
        if cell_origin is not None:
            cell = cell.transform_cell(AffineTransform.translate(to_vec3(cell_origin)))

        return AtomCell(self.atoms, cell, frame='local')

    def transform(self, transform: IntoTransform, frame: CoordinateFrame = 'local') -> SimpleAtoms:
        self._check_allowed_frame(frame)
        return replace(self, atoms=self.atoms.transform(transform))

    transform_atoms = transform

    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        self._check_allowed_frame(frame)
        return self.atoms

    def _replace_atoms(self: SimpleAtomsT, atoms: Atoms, frame: CoordinateFrame = 'local') -> SimpleAtomsT:
        self._check_allowed_frame(frame)
        return replace(self, frame=atoms)

    def clone(self: SimpleAtomsT) -> SimpleAtomsT:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (self.atoms,)

    def __len__(self) -> int:
        return self.atoms.__len__()


@dataclass(init=False, repr=False, frozen=True)
class AtomCell(AtomCollection):
    """
    Cell of atoms with known size and periodic boundary conditions.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    cell: Cell
    """Cell coordinate system."""

    frame: CoordinateFrame = 'local'
    """Coordinate frame 'atoms' are stored in."""

    @classmethod
    def from_ortho(cls, atoms: IntoAtoms, ortho: LinearTransform, *,
                   n_cells: t.Optional[VecLike] = None,
                   frame: CoordinateFrame = 'local',
                   keep_frame: bool = False):
        """
        Make an atom cell given a list of atoms and an orthogonalization matrix.
        Atoms are assumed to be in the coordinate system `frame`.
        """
        cell = Cell.from_ortho(ortho, n_cells)
        return cls(atoms, cell, frame=frame, keep_frame=keep_frame)

    @classmethod
    def from_unit_cell(cls, atoms: IntoAtoms, cell_size: VecLike,
                       cell_angle: t.Optional[VecLike] = None, *,
                       n_cells: t.Optional[VecLike] = None,
                       frame: CoordinateFrame = 'local',
                       keep_frame: bool = False):
        """
        Make a cell given a list of atoms and unit cell parameters.
        Atoms are assumed to be in the coordinate system `frame`.
        """
        cell = Cell.from_unit_cell(cell_size, cell_angle, n_cells=n_cells)
        return cls(atoms, cell, frame=frame, keep_frame=keep_frame)

    def __init__(self, atoms: IntoAtoms, cell: Cell, *,
                 frame: CoordinateFrame = 'local',
                 keep_frame: bool = False):
        atoms = Atoms(atoms)
        # by default, store in local coordinates
        if not keep_frame and frame != 'local':
            atoms = atoms.transform(cell.get_transform('local', frame))
            frame = 'local'

        object.__setattr__(self, 'atoms', atoms)
        object.__setattr__(self, 'cell', cell)
        object.__setattr__(self, 'frame', frame)

        self.__post_init__()

    def __post_init__(self):
        pass

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size:  {self.cell.cell_size!s}",
            f"Cell angle: {self.cell.cell_angle!s}",
            f"# Cells: {self.cell.n_cells!s}",
            self.atoms,
        )

    def __len__(self) -> int:
        return self.atoms.__len__()

    def transform(self, transform: AffineTransform, frame: CoordinateFrame = 'local') -> AtomCell:
        if isinstance(transform, Transform) and not isinstance(transform, AffineTransform):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use `transform_atoms` instead.")
        # coordinate change the transform into atomic coordinates
        new_cell = self.cell.transform_cell(transform, frame)
        transform = self.cell.change_transform(transform, self.frame, frame)  # type: ignore
        return AtomCell(self.atoms.transform(transform), cell=new_cell)

    def transform_atoms(self, transform: IntoTransform, frame: CoordinateFrame = 'local') -> AtomCell:
        transform = self.cell.change_transform(Transform.make(transform), self.frame, frame)
        return AtomCell(self.atoms.transform(transform), cell=self.cell)

    def crop(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
             frame: CoordinateFrame = 'local') -> AtomCell:
        new_atoms = self.get_atoms(frame).crop(x_min, x_max, y_min, y_max, z_min, z_max)
        new_cell = self.cell.crop(x_min, x_max, y_min, y_max, z_min, z_max, frame=frame)
        return self._replace_atoms(new_atoms, frame)._replace_cell(new_cell)

    def crop_atoms(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
                   y_min: float = -numpy.inf, y_max: float = numpy.inf,
                   z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
                   frame: CoordinateFrame = 'local') -> AtomCell:
        new_atoms = self.get_atoms(frame).crop(x_min, x_max, y_min, y_max, z_min, z_max)
        return self._replace_atoms(new_atoms, frame)

    def wrap(self, eps: float = 1e-5):
        atoms = self.get_atoms('cell_box')
        coords = atoms.coords()
        coords = (coords + eps) % 1. - eps
        return self._replace_atoms(atoms.with_coords(coords), frame='cell_box')

    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        if frame == self.frame:
            return self.atoms
        return self.atoms.transform(self.cell.get_transform(frame, self.frame))

    def _replace_atoms(self, atoms: Atoms, frame: CoordinateFrame = 'local') -> AtomCell:
        if frame != self.frame:
            atoms = atoms.transform(self.cell.get_transform(self.frame, frame))
        return AtomCell(atoms, self.cell, frame=self.frame, keep_frame=True)

    def _replace_cell(self, cell: Cell) -> AtomCell:
        return AtomCell(self.atoms, cell, frame=self.frame, keep_frame=True)

    def bbox(self) -> BBox:
        return self.atoms.bbox() | self.cell.bbox()

    def is_orthogonal(self) -> bool:
        return self.cell.is_orthogonal()

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal:
            return OrthoCell(self.atoms, self.cell, frame=self.frame)
        raise NotImplementedError()

    def repeat(self, n: t.Union[int, VecLike]) -> AtomCell:
        """Tile the cell"""
        ns = numpy.broadcast_to(n, 3)
        if not numpy.issubdtype(ns.dtype, numpy.integer):
            raise ValueError(f"repeat() argument must be an integer or integer array.")

        cells = numpy.stack(numpy.meshgrid(*map(numpy.arange, ns))) \
            .reshape(3, -1).T.astype(float)

        atoms = self.get_atoms('cell_frac')
        atoms = Atoms.concat([
            atoms.transform(AffineTransform.translate(cell))
            for cell in cells
        ]) #.transform(self.cell.get_transform('local', 'cell_frac'))
        return AtomCell(atoms, self.cell.repeat(ns), frame='cell_frac')

    def explode(self) -> AtomCell:
        self.get_atoms('local')
        return AtomCell(self.atoms, self.cell.explode(), frame='local')

    def clone(self: AtomCellT) -> AtomCellT:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def write_mslice(self, f: FileOrPath, template: t.Optional[MSliceTemplate] = None):
        """Read this structure to an mslice file."""
        return io.write_mslice(self, f, template)

    __mul__ = repeat

    def assert_equal(self, other):
        assert isinstance(other, AtomCell)
        self.cell.assert_equal(other.cell)
        self.get_atoms('local').assert_equal(other.get_atoms('local'))


class OrthoCell(AtomCell):
    def __post_init__(self):
        if not numpy.allclose(self.cell.cell_angle, numpy.pi/2.):
            raise ValueError(f"OrthoCell constructed with non-orthogonal angles: {self.cell.cell_angle}")

    def is_orthogonal(self) -> t.Literal[True]:
        return True


from . import io


__ALL__ = [
    'CoordinateFrame', 'Atoms', 'IntoAtoms', 'AtomSelection', 'AtomCollection', 'AtomCell', 'Lattice',
]
