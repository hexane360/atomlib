"""
Core atomic structure types.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, fields, replace
import warnings
from pathlib import Path
import copy
import typing as t

import numpy

from .bbox import BBox3D
from .types import VecLike, to_vec3
from .transform import LinearTransform3D, AffineTransform3D, Transform3D, IntoTransform3D
from .cell import CoordinateFrame, Cell
from .atoms import Atoms, IntoAtoms


if t.TYPE_CHECKING:
    from .io import CIF, XYZ, XSF, CFG, FileOrPath, FileType  # pragma: no cover
    from .io.mslice import MSliceTemplate                     # pragma: no cover


AtomCollectionT = t.TypeVar('AtomCollectionT', bound='AtomCollection')
AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')
SimpleAtomsT = t.TypeVar('SimpleAtomsT', bound='SimpleAtoms')


class AtomCollection(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    @abc.abstractmethod
    def transform(self: AtomCollectionT, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Transforms cell boxes in addition to atoms.
        """
        ...

    @abc.abstractmethod
    def transform_atoms(self: AtomCollectionT, transform: IntoTransform3D, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Never transforms cell boxes.
        """
        ...

    @abc.abstractmethod
    def get_atoms(self, frame: CoordinateFrame = 'local') -> Atoms:
        ...

    @abc.abstractmethod
    def bbox(self) -> BBox3D:
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
    def read(path: FileOrPath, ty: t.Optional[FileType] = None) -> AtomCollection:
        """
        Read a structure from a file.

        Currently, supported file types are 'cif', 'xyz', and 'xsf'.
        If no `ty` is specified, it is inferred from the file's extension.
        """
        return io.read(path, ty)  # type: ignore

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

    def write(self, path: FileOrPath, ty: t.Optional[FileType] = None):
        """
        Write this structure to a file.

        A file type may be specified using `ty`.
        If no `ty` is specified, it is inferred from the path's extension.
        """
        io.write(self, path, ty)  # type: ignore


@dataclass(init=False, repr=False, frozen=True)
class SimpleAtoms(AtomCollection):
    """
    Cell of atoms with no known structure or periodicity.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    def __init__(self, atoms: IntoAtoms):
        object.__setattr__(self, 'atoms', Atoms(atoms))

    def bbox(self) -> BBox3D:
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
            cell = cell.transform_cell(AffineTransform3D.translate(to_vec3(cell_origin)))

        return AtomCell(self.atoms, cell, frame='local')

    def transform(self, transform: IntoTransform3D, frame: CoordinateFrame = 'local') -> SimpleAtoms:
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
    def from_ortho(cls, atoms: IntoAtoms, ortho: LinearTransform3D, *,
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

    def transform(self, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> AtomCell:
        if isinstance(transform, Transform3D) and not isinstance(transform, AffineTransform3D):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use `transform_atoms` instead.")
        # coordinate change the transform into atomic coordinates
        new_cell = self.cell.transform_cell(transform, frame)
        transform = self.cell.change_transform(transform, self.frame, frame)  # type: ignore
        return AtomCell(self.atoms.transform(transform), cell=new_cell)

    def transform_atoms(self, transform: IntoTransform3D, frame: CoordinateFrame = 'local') -> AtomCell:
        transform = self.cell.change_transform(Transform3D.make(transform), self.frame, frame)
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

    def crop_to_box(self, eps: float = 1e-5) -> AtomCell:
        new_atoms = self.get_atoms('cell_box').crop(*([-eps, 1-eps]*3))
        return self._replace_atoms(new_atoms, 'cell_box')

    def wrap(self, eps: float = 1e-5) -> AtomCell:
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

    def bbox(self) -> BBox3D:
        return self.atoms.bbox() | self.cell.bbox()

    def is_orthogonal(self) -> bool:
        return self.cell.is_orthogonal()

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal():
            return OrthoCell(self.atoms, self.cell, frame=self.frame)
        raise NotImplementedError()

    def _repeat_to_contain(self, pts: numpy.ndarray, pad: int = 0, frame: CoordinateFrame = 'cell_frac') -> AtomCell:
        #print(f"pts: {pts} in frame {frame}")
        pts = self.cell.get_transform('cell_frac', frame) @ pts

        bbox = BBox3D.unit() | BBox3D.from_pts(pts)
        min_bounds = numpy.floor(bbox.min).astype(int) - pad
        max_bounds = numpy.ceil(bbox.max).astype(int) + pad
        #print(f"tiling to {min_bounds}, {max_bounds}")
        repeat = max_bounds - min_bounds
        cells = numpy.stack(numpy.meshgrid(*map(numpy.arange, repeat))).reshape(3, -1).T.astype(float)

        atoms = self.get_atoms('cell_frac')
        atoms = Atoms.concat([
            atoms.transform(AffineTransform3D.translate(cell))
            for cell in cells
        ])
        cell = self.cell.repeat(repeat) \
            .transform_cell(AffineTransform3D.translate(min_bounds), 'cell_frac')
        return AtomCell(atoms, cell, frame='cell_frac')

    def repeat(self, n: t.Union[int, VecLike]) -> AtomCell:
        """Tile the cell"""
        ns = numpy.broadcast_to(n, 3)
        if not numpy.issubdtype(ns.dtype, numpy.integer):
            raise ValueError(f"repeat() argument must be an integer or integer array.")

        cells = numpy.stack(numpy.meshgrid(*map(numpy.arange, ns))) \
            .reshape(3, -1).T.astype(float)
        cells = cells * self.cell.box_size

        atoms = self.get_atoms('cell')
        atoms = Atoms.concat([
            atoms.transform(AffineTransform3D.translate(cell))
            for cell in cells
        ]) #.transform(self.cell.get_transform('local', 'cell_frac'))
        return AtomCell(atoms, self.cell.repeat(ns), frame='cell')

    def repeat_to(self, size: VecLike, crop: t.Union[bool, t.Sequence[bool]] = False) -> AtomCell:
        """
        Repeat the cell so it is at least ``size`` along the crystal's axes.

        If ``crop``, then crop the cell to exactly ``size``. This may break periodicity.
        ``crop`` may be a vector, in which case you can specify cropping only along some axes.
        """
        size = to_vec3(size)
        cell_size = self.cell.cell_size * self.cell.n_cells
        repeat = numpy.maximum(numpy.ceil(size / cell_size).astype(int), 1)
        atom_cell = self.repeat(repeat)

        crop_v = to_vec3(crop, dtype=numpy.bool_)
        if numpy.any(crop_v):
            crop_x, crop_y, crop_z = crop_v
            return atom_cell.crop(
                x_max = size[0] if crop_x else numpy.inf,
                y_max = size[1] if crop_y else numpy.inf,
                z_max = size[2] if crop_z else numpy.inf,
                frame='cell'
            )

        return atom_cell

    def repeat_x(self, n: int) -> AtomCell:
        """Tile the cell in the x axis."""
        return self.repeat((n, 1, 1))

    def repeat_y(self, n: int) -> AtomCell:
        """Tile the cell in the y axis."""
        return self.repeat((1, n, 1))

    def repeat_z(self, n: int) -> AtomCell:
        """Tile the cell in the z axis."""
        return self.repeat((1, 1, n))

    def repeat_to_x(self, size: float, crop: bool = False) -> AtomCell:
        """Repeat the cell so it is at least size ``size`` along the x axis."""
        return self.repeat_to([size, 0., 0.], [crop, False, False])

    def repeat_to_y(self, size: float, crop: bool = False) -> AtomCell:
        """Repeat the cell so it is at least size ``size`` along the y axis."""
        return self.repeat_to([0., size, 0.], [False, crop, False])

    def repeat_to_z(self, size: float, crop: bool = False) -> AtomCell:
        """Repeat the cell so it is at least size ``size`` along the z axis."""
        return self.repeat_to([0., 0., size], [False, False, crop])

    def repeat_to_aspect(self, plane: t.Literal['xy', 'xz', 'yz'] = 'xy', *,
                         aspect: float = 1., max_size: t.Optional[VecLike] = None):
        """
        Repeat to optimize the aspect ratio in ``plane``,
        while staying under ``max_size``.
        """
        if max_size is None:
            max_n = numpy.array([3, 3, 3], numpy.int_)
        else:
            max_n = numpy.maximum(numpy.floor(to_vec3(max_size) / self.cell.box_size), 1).astype(numpy.int_)

        if plane == 'xy':
            indices = [0, 1]
        elif plane == 'xz':
            indices = [0, 2]
        elif plane == 'yz':
            indices = [1, 2]
        else:
            raise ValueError(f"Invalid plane '{plane}'. Exepcted 'xy', 'xz', 'or 'yz'.")

        na = numpy.arange(1, max_n[indices[0]])
        nb = numpy.arange(1, max_n[indices[1]])
        (na, nb) = numpy.meshgrid(na, nb)

        aspects = na * self.cell.box_size[indices[0]] / (nb * self.cell.box_size[indices[1]])
        # cost function: log(aspect)^2  (so cost(0.5) == cost(2))
        min_i = numpy.argmin(numpy.log(aspects / aspect)**2)
        repeat = numpy.array([1, 1, 1], numpy.int_)
        repeat[indices] = na.flatten()[min_i], nb.flatten()[min_i]
        return self.repeat(repeat)

    def explode(self) -> AtomCell:
        """
        Forget any cell repetitions.

        Afterwards, ``self.explode().cell.cell_size == self.cell.box_size``.
        """
        self.get_atoms('local')
        return AtomCell(self.atoms, self.cell.explode(), frame='local')

    def clone(self: AtomCellT) -> AtomCellT:
        """Make a deep copy of ``self``."""
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def write_mslice(self, f: FileOrPath, template: t.Optional[MSliceTemplate] = None):
        """Write this structure to an mslice file."""
        return io.write_mslice(self, f, template)

    __mul__ = repeat

    def assert_equal(self, other: t.Any):
        """Assert this structure is equal to """
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
