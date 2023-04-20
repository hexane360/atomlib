"""
:py:class:`Atoms` with an associated :py:class:`Cell`.

This module defines :py:class:`HasAtomCell` and the concrete :py:class:`AtomCell`,
which combines the functionality of :py:class:`HasAtoms` and :py:class:`HasCell`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
import copy
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike
import polars

from .bbox import BBox3D
from .types import VecLike, to_vec3, ParamSpec
from .transform import LinearTransform3D, AffineTransform3D, Transform3D, IntoTransform3D
from .cell import CoordinateFrame, HasCell, Cell
from .atoms import HasAtoms, Atoms, IntoAtoms, AtomSelection, AtomValues

# pyright: reportImportCycles=false
from .mixins import AtomCellIOMixin


AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')
HasAtomCellT = t.TypeVar('HasAtomCellT', bound='HasAtomCell')
P = ParamSpec('P')
T = t.TypeVar('T')


def _fwd_atoms_get(f: t.Callable[P, T]) -> t.Callable[P, T]:
    """Forward getter method on :py:`HasAtomCell` to method on :py:`HasAtoms`"""
    def inner(self, *args, frame: t.Optional[CoordinateFrame] = None, **kwargs):
        return getattr(self.get_atoms(frame), f.__name__)(*args, **kwargs)

    return t.cast(t.Callable[P, T], inner)


def _fwd_atoms_transform(f: t.Callable[P, T]) -> t.Callable[P, T]:
    """Forward transformation method on :py:`HasAtomCell` to method on :py:`HasAtoms`"""
    def inner(self, *args, frame: t.Optional[CoordinateFrame] = None, **kwargs):
        return self.with_atoms(self._transform_atoms_in_frame(frame, lambda atoms: getattr(atoms, f.__name__)(*args, **kwargs)))

    return t.cast(t.Callable[P, T], inner)


class HasAtomCell(HasAtoms, HasCell, abc.ABC):
    @abc.abstractmethod
    def get_frame(self) -> CoordinateFrame:
        """Get the coordinate frame atoms are stored in."""
        ...

    @abc.abstractmethod
    def with_atoms(self: HasAtomCellT, atoms: HasAtoms, frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Replace the atoms in ``self``. If no coordinate frame is specified, keep the coordinate frame unchanged.
        """
        ...

    def with_cell(self: HasAtomCellT, cell: Cell) -> HasAtomCellT:
        """
        Replace the cell in ``self``, without touching the atomic coordinates.
        """
        return self.to_frame('local').with_cell(cell)

    def get_atomcell(self) -> AtomCell:
        frame = self.get_frame()
        return AtomCell(self.get_atoms(frame), self.get_cell(), frame=frame, keep_frame=True)

    @abc.abstractmethod
    def get_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> Atoms:
        """Get atoms contained in ``self``, in the given coordinate frame."""
        ...

    def bbox_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> BBox3D:
        """Return the bounding box of all the atoms in ``self``, in the given coordinate frame."""
        return self.get_atoms(frame).bbox()

    def bbox(self, frame: CoordinateFrame = 'local') -> BBox3D:
        """
        Return the combined bounding box of the cell and atoms in the given coordinate system.
        To get the cell or atoms bounding box only, use :py:meth:`bbox_cell` or :py:meth:`bbox_atoms`.
        """
        return self.bbox_atoms(frame) | self.bbox_cell(frame)

    # transformation

    def _transform_atoms_in_frame(self, frame: t.Optional[CoordinateFrame], f: t.Callable[[Atoms], Atoms]) -> Atoms:
        # ugly code
        if frame is None or frame == self.get_frame():
            return f(self.get_atoms())
        return f(self.get_atoms(frame)).transform(self.get_transform(self.get_frame(), frame))

    def to_frame(self: HasAtomCellT, frame: CoordinateFrame) -> HasAtomCellT:
        """Convert the stored Atoms to the given coordinate frame."""
        return self.with_atoms(self.get_atoms(frame), frame)

    def transform_atoms(self: HasAtomCellT, transform: IntoTransform3D, selection: t.Optional[AtomSelection] = None, *,
                        frame: CoordinateFrame = 'local', transform_velocities: bool = False) -> HasAtomCellT:
        """
        Transform the atoms in `self` by `transform`.
        If `selection` is given, only transform the atoms in `selection`.
        """
        transform = self.change_transform(Transform3D.make(transform), self.get_frame(), frame)
        return self.with_atoms(self.get_atoms(self.get_frame()).transform(transform, selection, transform_velocities=transform_velocities))

    def transform_cell(self: HasAtomCellT, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> HasAtomCellT:
        """
        Apply the given transform to the unit cell, without changing atom positions.
        The transform is applied in coordinate frame 'frame'.
        """
        return self.with_cell(self.get_cell().transform_cell(transform, frame=frame))

    def transform(self: HasAtomCellT, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> HasAtomCellT:
        if isinstance(transform, Transform3D) and not isinstance(transform, AffineTransform3D):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use 'transform_atoms' instead.")
        # TODO: cleanup once tests pass
        # coordinate change the transform into atomic coordinates
        new_cell = self.get_cell().transform_cell(transform, frame)
        transform = self.get_cell().change_transform(transform, self.get_frame(), frame)
        return self.with_atoms(self.get_atoms().transform(transform), self.get_frame()).with_cell(new_cell)

    # crop methods

    def crop(self: HasAtomCellT, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
             frame: CoordinateFrame = 'local') -> HasAtomCellT:
        """
        Crop atoms and cell to the given extents. For a non-orthogonal
        cell, this must be specified in cell coordinates. This
        function implicity `explode`s the cell as well.

        To crop atoms only, use :py:meth:`crop_atoms` instead.
        """

        cell = self.get_cell().crop(x_min, x_max, y_min, y_max, z_min, z_max, frame=frame)
        atoms = self._transform_atoms_in_frame(frame, lambda atoms: atoms.crop_atoms(x_min, x_max, y_min, y_max, z_min, z_max))
        return self.with_cell(cell).with_atoms(atoms)

    def crop_atoms(self: HasAtomCellT, x_min: float = -numpy.inf, x_max: float = numpy.inf,
                   y_min: float = -numpy.inf, y_max: float = numpy.inf,
                   z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
                   frame: CoordinateFrame = 'local') -> HasAtomCellT:
        atoms = self._transform_atoms_in_frame(frame, lambda atoms: atoms.crop_atoms(x_min, x_max, y_min, y_max, z_min, z_max))
        return self.with_atoms(atoms)

    def crop_to_box(self: HasAtomCellT, eps: float = 1e-5) -> HasAtomCellT:
        atoms = self._transform_atoms_in_frame('cell_box', lambda atoms: atoms.crop_atoms(*([-eps, 1-eps]*3)))
        return self.with_atoms(atoms)

    def wrap(self: HasAtomCellT, eps: float = 1e-5) -> HasAtomCellT:
        """Wrap atoms around the cell boundaries."""
        def transform(atoms):
            coords = atoms.coords()
            coords = (coords + eps) % 1. - eps
            return atoms.with_coords(coords)

        return self.with_atoms(self._transform_atoms_in_frame('cell_box', transform))

    """
    def explode(self: HasAtomCellT) -> HasAtomCellT:
        \"""
        Forget any cell repetitions.

        Afterwards, ``self.explode().cell.cell_size == self.cell.box_size``.
        \"""
        # when we explode, we need to make sure atoms aren't stored in cell coordinates
        return self.to_frame('local').with_cell(self.get_cell().explode())
    """

    def _repeat_to_contain(self: HasAtomCellT, pts: numpy.ndarray, pad: int = 0, frame: CoordinateFrame = 'cell_frac') -> HasAtomCellT:
        #print(f"pts: {pts} in frame {frame}")
        pts = self.get_transform('cell_frac', frame) @ pts

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
        #print(f"atoms:\n{atoms}")
        cell = self.get_cell().repeat(repeat) \
            .transform_cell(AffineTransform3D.translate(min_bounds), 'cell_frac')
        return self.with_cell(cell).with_atoms(atoms, 'cell_frac')

    def repeat(self: HasAtomCellT, n: t.Union[int, VecLike]) -> HasAtomCellT:
        """Tile the cell"""
        ns = numpy.broadcast_to(n, 3)
        if not numpy.issubdtype(ns.dtype, numpy.integer):
            raise ValueError(f"repeat() argument must be an integer or integer array.")

        cells = numpy.stack(numpy.meshgrid(*map(numpy.arange, ns))) \
            .reshape(3, -1).T.astype(float)
        cells = cells * self.box_size

        atoms = self.get_atoms('cell')
        atoms = Atoms.concat([
            atoms.transform(AffineTransform3D.translate(cell))
            for cell in cells
        ]) #.transform(self.cell.get_transform('local', 'cell_frac'))
        return self.with_atoms(atoms, 'cell').with_cell(self.get_cell().repeat(ns))

    def repeat_to(self: HasAtomCellT, size: VecLike, crop: t.Union[bool, t.Sequence[bool]] = False) -> HasAtomCellT:
        """
        Repeat the cell so it is at least ``size`` along the crystal's axes.

        If ``crop``, then crop the cell to exactly ``size``. This may break periodicity.
        ``crop`` may be a vector, in which case you can specify cropping only along some axes.
        """
        size = to_vec3(size)
        cell_size = self.cell_size * self.n_cells
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

    def repeat_x(self: HasAtomCellT, n: int) -> HasAtomCellT:
        """Tile the cell in the x axis."""
        return self.repeat((n, 1, 1))

    def repeat_y(self: HasAtomCellT, n: int) -> HasAtomCellT:
        """Tile the cell in the y axis."""
        return self.repeat((1, n, 1))

    def repeat_z(self: HasAtomCellT, n: int) -> HasAtomCellT:
        """Tile the cell in the z axis."""
        return self.repeat((1, 1, n))

    def repeat_to_x(self: HasAtomCellT, size: float, crop: bool = False) -> HasAtomCellT:
        """Repeat the cell so it is at least size ``size`` along the x axis."""
        return self.repeat_to([size, 0., 0.], [crop, False, False])

    def repeat_to_y(self: HasAtomCellT, size: float, crop: bool = False) -> HasAtomCellT:
        """Repeat the cell so it is at least size ``size`` along the y axis."""
        return self.repeat_to([0., size, 0.], [False, crop, False])

    def repeat_to_z(self: HasAtomCellT, size: float, crop: bool = False) -> HasAtomCellT:
        """Repeat the cell so it is at least size ``size`` along the z axis."""
        return self.repeat_to([0., 0., size], [False, False, crop])

    def repeat_to_aspect(self: HasAtomCellT, plane: t.Literal['xy', 'xz', 'yz'] = 'xy', *,
                         aspect: float = 1., max_size: t.Optional[VecLike] = None) -> HasAtomCellT:
        """
        Repeat to optimize the aspect ratio in ``plane``,
        while staying under ``max_size``.
        """
        if max_size is None:
            max_n = numpy.array([3, 3, 3], numpy.int_)
        else:
            max_n = numpy.maximum(numpy.floor(to_vec3(max_size) / self.box_size), 1).astype(numpy.int_)

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

        aspects = na * self.box_size[indices[0]] / (nb * self.box_size[indices[1]])
        # cost function: log(aspect)^2  (so cost(0.5) == cost(2))
        min_i = numpy.argmin(numpy.log(aspects / aspect)**2)
        repeat = numpy.array([1, 1, 1], numpy.int_)
        repeat[indices] = na.flatten()[min_i], nb.flatten()[min_i]
        return self.repeat(repeat)

    # add frame to some HasAtoms methods

    @_fwd_atoms_transform
    def filter(self: HasAtomCellT, selection: t.Optional[AtomSelection] = None, *,
               frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """Filter ``self``, removing rows which evaluate to ``False``."""
        ...

    @_fwd_atoms_get
    def select(self, exprs: t.Union[str, polars.Expr, polars.Series, t.Sequence[t.Union[str, polars.Expr, polars.Series]]], *,
               frame: t.Optional[CoordinateFrame] = None) -> polars.DataFrame:
        """
        Select ``exprs`` from ``self``, and return as a :py:class:`polars.DataFrame`.

        Expressions may either be columns or expressions of columns.
        """
        ...

    @_fwd_atoms_get
    def try_select(self, exprs: t.Union[str, polars.Expr, polars.Series, t.Sequence[t.Union[str, polars.Expr, polars.Series]]], *,
                   frame: t.Optional[CoordinateFrame] = None) -> t.Optional[polars.DataFrame]:
        """
        Try to select ``exprs`` from ``self``, and return as a :py:class:`polars.DataFrame`.

        Expressions may either be columns or expressions of columns.
        Return ``None`` if any columns are missing.
        """
        ...

    @_fwd_atoms_transform
    def sort(self: HasAtomCellT, by: t.Union[str, polars.Expr, t.List[str], t.List[polars.Expr]],
             descending: t.Union[bool, t.List[bool]] = False, *,
             frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        ...

    @_fwd_atoms_transform
    def with_column(self: HasAtomCellT, column: t.Union[polars.Series, polars.Expr], *,
                    frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """Return a copy of ``self`` with the given column added."""
        ...

    def with_columns(self: HasAtomCellT,
                     exprs: t.Union[t.Literal[None], polars.Series, polars.Expr, t.Sequence[t.Union[polars.Series, polars.Expr]]], *,
                     frame: t.Optional[CoordinateFrame] = None,
                     **named_exprs: t.Union[polars.Expr, polars.Series]) -> HasAtomCellT:
        """Return a copy of ``self`` with the given columns added."""
        ...

    @_fwd_atoms_transform
    def round_near_zero(self: HasAtomCellT, tol: float = 1e-14, *,
                        frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Round atom position values near zero to zero.
        """
        ...

    @_fwd_atoms_get
    def coords(self, selection: t.Optional[AtomSelection] = None, *, frame: t.Optional[CoordinateFrame] = None) -> NDArray[numpy.float64]:
        """Returns a ``(N, 3)`` ndarray of atom coordinates (dtype ``numpy.float64``)."""
        ...

    @_fwd_atoms_get
    def velocities(self, selection: t.Optional[AtomSelection] = None, *, frame: t.Optional[CoordinateFrame] = None) -> t.Optional[NDArray[numpy.float64]]:
        """Returns a ``(N, 3)`` ndarray of atom velocities (dtype ``numpy.float64``)."""
        ...

    @t.overload
    def add_atom(self: HasAtomCellT, elem: t.Union[int, str], x: ArrayLike, /, *,
                 y: None = None, z: None = None, frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> HasAtomCellT:
        ...

    @t.overload
    def add_atom(self: HasAtomCellT, elem: t.Union[int, str], /,
                 x: float, y: float, z: float, *,
                 frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> HasAtomCellT:
        ...

    @_fwd_atoms_transform
    def add_atom(self: HasAtomCellT, elem: t.Union[int, str], /,
                 x: t.Union[ArrayLike, float],
                 y: t.Optional[float] = None,
                 z: t.Optional[float] = None, *,
                 frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> HasAtomCellT:
        """
        Return a copy of ``self`` with an extra atom.

        By default, all extra columns present in ``self`` must be specified as ``**kwargs``.

        Try to avoid calling this in a loop (Use :py:meth:`concat` instead).
        """
        ...

    @_fwd_atoms_transform
    def with_index(self: HasAtomCellT, index: t.Optional[AtomValues] = None, *,
                   frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Returns ``self`` with a row index added in column 'i' (dtype polars.Int64).
        If ``index`` is not specified, defaults to an existing index or a new index.
        """
        ...

    @_fwd_atoms_transform
    def with_wobble(self: HasAtomCellT, wobble: t.Optional[AtomValues] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` with the given displacements in column 'wobble' (dtype polars.Float64).
        If ``wobble`` is not specified, defaults to the already-existing wobbles or 0.
        """
        ...

    @_fwd_atoms_transform
    def with_occupancy(self: HasAtomCellT, frac_occupancy: t.Optional[AtomValues] = None, *,
                       frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return self with the given fractional occupancies. If ``frac_occupancy`` is not specified,
        defaults to the already-existing occupancies or 1.
        """
        ...

    @_fwd_atoms_transform
    def apply_wobble(self: HasAtomCellT, rng: t.Union[numpy.random.Generator, int, None] = None,
                     frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Displace the atoms in ``self`` by the amount in the `wobble` column.
        ``wobble`` is interpretated as a mean-squared displacement, which is distributed
        equally over each axis.
        """
        ...

    @_fwd_atoms_transform
    def with_type(self: HasAtomCellT, types: t.Optional[AtomValues] = None, *,
                  frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` with the given atom types in column 'type'.
        If ``types`` is not specified, use the already existing types or auto-assign them.

        When auto-assigning, each symbol is given a unique value, case-sensitive.
        Values are assigned from lowest atomic number to highest.
        For instance: ``["Ag+", "Na", "H", "Ag"]`` => ``[3, 11, 1, 2]``
        """
        ...

    @_fwd_atoms_transform
    def with_mass(self: HasAtomCellT, mass: t.Optional[ArrayLike] = None, *,
                  frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` with the given atom masses in column 'mass'.
        If ``mass`` is not specified, use the already existing masses or auto-assign them.
        """
        ...

    @_fwd_atoms_transform
    def with_symbol(self: HasAtomCellT, symbols: ArrayLike, selection: t.Optional[AtomSelection] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` with the given atomic symbols.
        """
        ...

    @_fwd_atoms_transform
    def with_coords(self: HasAtomCellT, pts: ArrayLike, selection: t.Optional[AtomSelection] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` replaced with the given atomic positions.
        """
        ...

    @_fwd_atoms_transform
    def with_velocity(self: HasAtomCellT, pts: t.Optional[ArrayLike] = None,
                      selection: t.Optional[AtomSelection] = None, *,
                      frame: t.Optional[CoordinateFrame] = None) -> HasAtomCellT:
        """
        Return ``self`` replaced with the given atomic velocities.
        If ``pts`` is not specified, use the already existing velocities or zero.
        """
        ...


@dataclass(init=False, repr=False, frozen=True)
class AtomCell(AtomCellIOMixin, HasAtomCell):
    """
    Cell of atoms with known size and periodic boundary conditions.
    """

    atoms: Atoms
    """Atoms in the cell. Stored in 'local' coordinates (i.e. relative to the enclosing group but not relative to box dimensions)."""

    cell: Cell
    """Cell coordinate system."""

    frame: CoordinateFrame = 'local'
    """Coordinate frame 'atoms' are stored in."""

    def get_cell(self) -> Cell:
        return self.cell

    def with_cell(self: AtomCellT, cell: Cell) -> AtomCellT:
        return self.__class__(self.atoms, cell, frame=self.frame, keep_frame=True)

    def get_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> Atoms:
        """Get atoms contained in ``self``, in the given coordinate frame."""

        if frame is None or frame == self.get_frame():
            return self.atoms
        return self.atoms.transform(self.get_transform(frame, self.get_frame()))

    def with_atoms(self: AtomCellT, atoms: HasAtoms, frame: t.Optional[CoordinateFrame] = None) -> AtomCellT:
        frame = frame if frame is not None else self.frame
        return self.__class__(atoms.get_atoms(), cell=self.cell, frame=frame, keep_frame=True)
        #return replace(self, atoms=atoms, frame = frame if frame is not None else self.frame, keep_frame=True)

    def get_frame(self) -> CoordinateFrame:
        """Get the coordinate frame atoms are stored in."""
        return self.frame

    @classmethod
    def _combine_metadata(cls: t.Type[AtomCellT], *atoms: HasAtoms) -> AtomCellT:
        """
        When combining multiple :py:`HasAtoms`, check that they are compatible with each other,
        and return a 'representative' which best represents the combined metadata.
        Implementors should treat :py:`Atoms` as acceptable, but having no metadata.
        """
        atom_cells = [a for a in atoms if isinstance(a, AtomCell)]
        if len(atom_cells) == 0:
            raise TypeError(f"No AtomCells to combine")
        cell = atom_cells[0].cell
        frame = atom_cells[0].frame
        if not all(a.cell == cell for a in atom_cells[1:]):
            raise TypeError(f"Can't combine AtomCells with different cells")
        return cls(Atoms.empty(), frame=frame, cell=cell)

    @classmethod
    def from_ortho(cls, atoms: IntoAtoms, ortho: LinearTransform3D, *,
                   n_cells: t.Optional[VecLike] = None,
                   frame: CoordinateFrame = 'local',
                   keep_frame: bool = False):
        """
        Make an atom cell given a list of atoms and an orthogonalization matrix.
        Atoms are assumed to be in the coordinate system ``frame``.
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
        Atoms are assumed to be in the coordinate system ``frame``.
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

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal():
            return OrthoCell(self.atoms, self.cell, frame=self.frame)
        raise NotImplementedError()

    def clone(self: AtomCellT) -> AtomCellT:
        """Make a deep copy of ``self``."""
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def assert_equal(self, other: t.Any):
        """Assert this structure is equal to """
        assert isinstance(other, AtomCell)
        self.cell.assert_equal(other.cell)
        self.get_atoms('local').assert_equal(other.get_atoms('local'))

    def _str_parts(self) -> t.Iterable[t.Any]:
        return (
            f"Cell size:  {self.cell.cell_size!s}",
            f"Cell angle: {self.cell.cell_angle!s}",
            f"# Cells: {self.cell.n_cells!s}",
            f"Frame: {self.frame}",
            self.atoms,
        )

    def __str__(self) -> str:
        return "\n".join(map(str, self._str_parts()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.atoms!r}, cell={self.cell!r}, frame={self.frame})"

    def _repr_pretty_(self, p, cycle: bool) -> None:
        p.text(f'{self.__class__.__name__}(...)') if cycle else p.text(str(self))


class OrthoCell(AtomCell):
    def __post_init__(self):
        if not numpy.allclose(self.cell.cell_angle, numpy.pi/2.):
            raise ValueError(f"OrthoCell constructed with non-orthogonal angles: {self.cell.cell_angle}")

    def is_orthogonal(self, tol: float = 1e-8) -> t.Literal[True]:
        """Returns whether this cell is orthogonal (axes are at right angles.)"""
        return True


__ALL__ = [
    'HasAtomCell', 'AtomCell',
]
