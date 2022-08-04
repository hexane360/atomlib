from __future__ import annotations

import abc
from dataclasses import dataclass, fields, replace
import warnings
import copy
import typing as t

import numpy
import polars

from .vec import Vec3, BBox
from .transform import LinearTransform, AffineTransform, Transform
from .cell import cell_to_ortho, ortho_to_cell


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

IntoAtoms: t.TypeAlias = t.Union[t.Dict[str, t.Sequence[t.Any]], t.Sequence[t.Any], numpy.ndarray, polars.DataFrame]
"""
A type convertible into `Atoms`.
"""

AtomCollectionT = t.TypeVar('AtomCollectionT', bound='AtomCollection')
CellT = t.TypeVar('CellT', bound='Cell')


class AtomFrame(polars.DataFrame):
    """
    A collection of atoms, absent any implied coordinate system.
    Implemented as a wrapper around a Polars DataFrame.

    Must contain the following columns:
    - x: x position, float
    - y: y position, float
    - z: z position, float
    - elem: atomic number, int
    - symbol: atomic symbol (may contain charges)

    In addition, it commonly contains the following columns:
    - i: Initial atom number
    - wobble: Isotropic Debye-Waller standard deviation (MSD, <u^2> = B*3/8pi^2, dimensions of [Length^2])
    - frac: Fractional occupancy, [0., 1.]
    """

    def __new__(cls, data: t.Optional[IntoAtoms] = None, columns: t.Optional[t.Sequence[str]] = None) -> AtomFrame:
        if data is None:
            return super().__new__(cls)
        if isinstance(data, polars.DataFrame):
            obj = data.clone()
        else:
            obj = polars.DataFrame(data, columns)
        obj.__class__ = cls

        return t.cast(AtomFrame, obj)

    def __init__(self, data: IntoAtoms, columns: t.Optional[t.Sequence[str]] = None):
        self._validate_atoms()
        self._bbox: t.Optional[BBox] = None

    def _validate_atoms(self):
        missing = set(('x', 'y', 'z', 'elem', 'symbol')) - set(self.columns)
        if len(missing):
            raise ValueError(f"'Atoms' missing column(s) {list(missing)}")

    def coords(self) -> numpy.ndarray:
        """Returns a (N, 3) ndarray of atom coordinates."""
        # TODO find a way to get a view
        return self.select(('x', 'y', 'z')).to_numpy()

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self._bbox = BBox.from_pts(self.coords())

        return self._bbox


AtomSelection = t.NewType('AtomSelection', polars.Expr)
"""
Polars expression selecting a subset of atoms from an AtomFrame. Can be used with DataFrame.filter()
"""


class AtomCollection(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    @abc.abstractmethod
    def transform(self, transform: Transform, frame: CoordinateFrame = 'local'):
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


@dataclass(init=False, repr=False)
class Cell(AtomCollection):
    """
    Cell of atoms with no known periodicity.
    """

    atoms: AtomFrame
    """Atoms in the cell. The coordinates this is stored in depends on the subclass. Use 'coords' instead."""

    def __init__(self, atoms: IntoAtoms):
        self.atoms = AtomFrame(atoms)

    def bbox(self) -> BBox:
        return self.atoms.bbox

    def with_bounds(self, cell_size: Vec3) -> PeriodicCell:
        return PeriodicCell(self.atoms, cell_size)

    def assume_bounds(self) -> PeriodicCell:
        warnings.warn("Cell boundary unknown. Defaulting to cell BBox")
        bbox = self.atoms.bbox
        # TODO be careful with the origin here
        atoms = self.atoms.transform(AffineTransform().translate(*-bbox.min).scale(*1/bbox.size))
        return PeriodicCell(atoms, bbox.size)

    def transform(self, transform: Transform, frame: CoordinateFrame = 'local'):
        self.atoms.transform(transform)

    def clone(self: CellT) -> CellT:
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (self.atoms,)


@dataclass(init=False, repr=False)
class PeriodicCell(Cell):
    """
    Cell of atoms with known size and periodic boundary conditions.

    Unlike regular `Cell`s, atoms are stored in fractional coordinates.
    Be careful when converting between the two representations.
    """

    cell_size: Vec3

    n_cells: Vec3
    """Number of cells (n_a, n_b, n_c)"""

    def __init__(self, atoms: IntoAtoms, cell_size: Vec3, n_cells: t.Optional[Vec3] = None):
        super().__init__(atoms)
        self.cell_size = numpy.broadcast_to(cell_size, (3,)).view(Vec3)

        if n_cells is None:
            self.n_cells = numpy.ones((3,), dtype=int).view(Vec3)
        else:
            self.n_cells = numpy.broadcast_to(n_cells, (3,)).view(Vec3)
            if not numpy.issubdtype(self.n_cells.dtype, numpy.integer):
                raise TypeError(f"n_cells must be an integer dtype. Instead got dtype '{self.n_cells.dtype}'")

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size: {self.cell_size!s}",
            f"# Cells: {self.n_cells!s}",
            self.atoms,
        )

    def with_bounds(self, cell_size: Vec3) -> PeriodicCell:
        if isinstance(self, Lattice):
            raise NotImplementedError()
        return replace(self, cell_size=cell_size)

    def assume_bounds(self) -> PeriodicCell:
        return self

    def repeat(self, n: t.Union[int, t.Tuple[int, int, int]], /, *, explode: bool = False):
        """Tile the cell"""

        self.n_cells = (self.n_cells * numpy.broadcast_to(n, 3)).view(Vec3)
        if explode:
            self.explode()

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
        self.atoms = AtomFrame(df.drop(['dx', 'dy', 'dz']))

    __mul__ = repeat


@dataclass(init=False, repr=False)
class Lattice(PeriodicCell):
    """
    Full crystal with arbitrary Bravais lattice.
    """

    cell_angle: Vec3
    """Cell angles (alpha, beta, gamma)"""

    ortho: LinearTransform
    """Orthogonalization transform. Converts fractional coordinates to local real-space coordinates."""
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

        self.atoms = AtomFrame(atoms)
        self.n_cells = n_cells if n_cells is not None else numpy.ones((3,), dtype=int).view(Vec3)

        if ortho is not None:
            if cell_size is not None or cell_angle is not None:
                raise ValueError("Crystal: 'ortho' cannot be specified with 'cell_size' or 'cell_angle'.")

            self.ortho = ortho
            (self.cell_size, self.cell_angle) = ortho_to_cell(ortho)
        else:
            if cell_size is None:
                raise ValueError("Crystal: Either 'cell_size' or 'ortho' must be specified.")

            self.cell_size = cell_size
            self.cell_angle = n_cells if n_cells is not None else numpy.full(3, numpy.pi/2.).view(Vec3)
            self.ortho = cell_to_ortho(self.cell_size, self.cell_angle)

        self.metric = self.ortho.T @ self.ortho

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size:  {self.cell_size!s}",
            f"Cell angle: {self.cell_angle!s}",
            f"# Cells: {self.n_cells!s}",
            self.atoms,
        )

    def is_orthogonal(self) -> bool:
        return numpy.allclose(self.cell_angle, numpy.pi/2.)

    def orthogonalize(self) -> PeriodicCell:
        if self.is_orthogonal:
            return PeriodicCell(self.atoms, self.cell_size, self.n_cells)
        raise NotImplementedError()