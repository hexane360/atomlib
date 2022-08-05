from __future__ import annotations

import abc
from dataclasses import dataclass, fields, replace
import warnings
from pathlib import Path
import copy
import typing as t

import numpy
import polars

from .vec import Vec3, BBox
from .transform import LinearTransform, AffineTransform, Transform
from .cell import cell_to_ortho, ortho_to_cell
from .frame import AtomFrame, AtomSelection, IntoAtoms


if t.TYPE_CHECKING:
    from .io import CIF, XYZ, XSF, FileOrPath, FileType


CoordinateFrame = t.Union[t.Literal['local'], t.Literal['global'], t.Literal['frac']]
"""
A coordinate frame to use.
There are three main coordinate frames:
 - 'crystal', uses crystallographic axes
 - 'local', orthogonal coordinate system defined by the crystal's bounding box
 - 'global', global coordinate frame

In addition, the 'crystal' and 'local' coordinate frames support fractional
coordinates as well as realspace (in angstrom) coordinates.
"""


AtomCollectionT = t.TypeVar('AtomCollectionT', bound='AtomCollection')
AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')


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
    def transform_atoms(self: AtomCollectionT, transform: Transform, frame: CoordinateFrame = 'local') -> AtomCollectionT:
        """
        Transform atoms by `transform`, in the coordinate frame `frame`.
        Never transforms cell boxes.
        """
        ...

    @abc.abstractmethod
    def get_atoms(self, frame: CoordinateFrame = 'local') -> AtomFrame:
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

    def __repr__(self) -> str:
        return "\n".join(map(str, self._str_parts()))

    __str__ = __repr__

    @t.overload
    def read(path: FileOrPath, ty: FileType) -> AtomCollection:
        ...

    @t.overload
    def read(path: t.Union[str, Path], ty: t.Literal[None] = None) -> AtomCollection:
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
        """Read a structure from a XSF file."""
        return io.read_xsf(f)


@dataclass(init=False, repr=False, frozen=True)
class SimpleAtoms(AtomCollection):
    """
    Cell of atoms with no known structure or periodicity.
    """

    atoms: AtomFrame
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    def __init__(self, atoms: IntoAtoms):
        object.__setattr__(self, 'atoms', AtomFrame(atoms))

    def bbox(self) -> BBox:
        """Get this structure's bounding box."""
        return self.atoms.bbox

    def with_bounds(self, cell_size: t.Optional[Vec3] = None) -> AtomCell:
        """
        Return a periodic cell with the given orthogonal cell dimensions.

        If cell_size is not specified, it will be assumed (and may be incorrect).
        """
        if cell_size is None:
            warnings.warn("Cell boundary unknown. Defaulting to cell BBox")
            cell_size = self.atoms.bbox.size

        # TODO be careful with the origin here
        return AtomCell(self.atoms, cell_size=cell_size)

    def transform(self, transform: Transform, frame: CoordinateFrame = 'local') -> SimpleAtoms:
        if frame.lower() == 'frac':
            raise ValueError("Can't use 'frac' coordinate frame when box is unknown.")

        return replace(self, atoms=self.atoms.transform(transform))

    transform_atoms = transform

    def get_atoms(self, frame: CoordinateFrame = 'local') -> AtomFrame:
        if frame.lower() == 'frac':
            raise ValueError("Can't use 'frac' coordinate frame when box is unknown.")

        return self.atoms

    def clone(self) -> SimpleAtoms:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (self.atoms,)


@dataclass(init=False, repr=False, frozen=True)
class AtomCell(AtomCollection):
    """
    Cell of atoms with known size and periodic boundary conditions.
    """

    atoms: AtomFrame
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    cell_size: Vec3
    """Cell size (a, b, c), neglecting n_cells."""

    cell_angle: Vec3
    """Cell angles (alpha, beta, gamma)"""

    n_cells: Vec3
    """Number of cells (n_a, n_b, n_c)"""

    ortho: LinearTransform
    """Orthogonalization transform. Converts fractional coordinates to local real-space coordinates."""
    #ortho_inv: LinearTransform
    #"""Inverse orthogonalization transform. Converts local real-space coordinates to fractional coordinates."""
    metric: LinearTransform
    """Metric tensor. p dot q = p.T @ M @ q forall p, q"""

    @t.overload
    def __init__(self, atoms: IntoAtoms, cell_size: Vec3,
                 cell_angle: t.Optional[Vec3] = None, *,
                 n_cells: t.Optional[Vec3] = None,
                 ortho: t.Literal[None] = None):
        ...

    @t.overload
    def __init__(self, atoms: IntoAtoms, *,
                 n_cells: t.Optional[Vec3] = None,
                 ortho: LinearTransform):
        ...

    def __init__(self, atoms: IntoAtoms,
                 cell_size: t.Optional[Vec3] = None,
                 cell_angle: t.Optional[Vec3] = None, *,
                 n_cells: t.Optional[Vec3] = None,
                 ortho: t.Optional[LinearTransform] = None):

        object.__setattr__(self, 'atoms', AtomFrame(atoms))

        if n_cells is None:
            n_cells = numpy.ones((3,), dtype=int).view(Vec3)
        else:
            n_cells = numpy.broadcast_to(n_cells, (3,)).view(Vec3)
            if not numpy.issubdtype(self.n_cells.dtype, numpy.integer):
                raise TypeError(f"n_cells must be an integer dtype. Instead got dtype '{self.n_cells.dtype}'")
        object.__setattr__(self, 'n_cells', n_cells)

        if ortho is not None:
            if cell_size is not None or cell_angle is not None:
                raise ValueError("Crystal: 'ortho' cannot be specified with 'cell_size' or 'cell_angle'.")

            (cell_size, cell_angle) = ortho_to_cell(ortho)
        else:
            if cell_size is None:
                raise ValueError("Crystal: Either 'cell_size' or 'ortho' must be specified.")

            cell_angle = n_cells if n_cells is not None else numpy.full(3, numpy.pi/2.).view(Vec3)
            ortho = cell_to_ortho(self.cell_size, self.cell_angle)

        object.__setattr__(self, 'ortho', ortho)
        object.__setattr__(self, 'cell_size', cell_size)
        object.__setattr__(self, 'cell_angle', cell_angle)
        object.__setattr__(self, 'metric', self.ortho.T @ self.ortho)

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size:  {self.cell_size!s}",
            f"Cell angle: {self.cell_angle!s}",
            f"# Cells: {self.n_cells!s}",
            self.atoms,
        )

    def transform(self, transform: AffineTransform, frame: CoordinateFrame = 'local') -> AtomCell:
        if isinstance(transform, Transform) and not isinstance(transform, AffineTransform):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use `transform_atoms` instead.")

        if frame.lower() == 'frac':
            # coordinate change the transform
            transform = self.ortho @ transform @ self.ortho.inverse()

        ortho = transform.to_linear() @ self.ortho
        return AtomCell(self.atoms.transform(transform), ortho=ortho)

    def transform_atoms(self, transform: Transform, frame: CoordinateFrame = 'local') -> AtomCell:
        if frame.lower() == 'frac':
            # coordinate change the transform
            transform = self.ortho @ transform @ self.ortho.inverse()

        return AtomCell(self.atoms.transform(transform), ortho=self.ortho)

    def get_atoms(self, frame: CoordinateFrame = 'local') -> AtomFrame:
        if frame.lower() == 'frac':
            return self.atoms.transform(self.ortho.inverse())

        return self.atoms

    def bbox(self) -> BBox:
        return self.atoms.bbox

    def is_orthogonal(self) -> bool:
        return numpy.allclose(self.cell_angle, numpy.pi/2.)

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal:
            return OrthoCell(self.atoms, ortho=self.ortho, n_cells=self.n_cells)
        raise NotImplementedError()

    def repeat(self, n: t.Union[int, t.Tuple[int, int, int]], /, *, explode: bool = False):
        """Tile the cell"""

        n_cells = (self.n_cells * numpy.broadcast_to(n, 3)).view(Vec3)
        new = replace(self, n_cells=n_cells)
        return new.explode() if explode else new

    def explode(self):
        """Turn a tiled cell into one large cell, duplicating atoms along the way."""
        if numpy.prod(self.n_cells) == 1:
            return

        a, b, c = map(numpy.arange, self.n_cells)
        # Nx3 ndarray of cell offsets
        cells = numpy.dstack(numpy.meshgrid(a, b, c)).reshape(-1, 3).astype(float)
        #cell_offsets = self.cell_size * cells
        offset_frame = polars.DataFrame(cells, columns=('dx', 'dy', 'dz'))
        df = self.atoms.join(offset_frame, how='cross')
        df['x'] += df['da']
        df['x'] += df['db']
        df['y'] += df['dc']

        ortho = self.ortho @ LinearTransform().scale(a, b, c)  # type: ignore
        return replace(self, atoms=AtomFrame(df.drop(['dx', 'dy', 'dz'])))

    def clone(self: AtomCellT) -> AtomCellT:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    __mul__ = repeat


class OrthoCell(AtomCell):
    def __post_init__(self):
        if not numpy.allclose(self.cell_angle, numpy.pi/2.):
            raise ValueError(f"OrthoCell constructed with non-orthogonal angles: {self.cell_angle}")

    def is_orthogonal(self) -> t.Literal[True]:
        return True


from . import io


__ALL__ = [
    'CoordinateFrame', 'AtomFrame', 'IntoAtoms', 'AtomSelection', 'AtomCollection', 'Cell', 'PeriodicCell', 'Lattice',
]