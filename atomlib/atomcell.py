"""
[`Atoms`][atomlib.atoms.Atoms] with an associated [`Cell`][atomlib.cell.Cell].

This module defines [`HasAtomCell`][atomlib.atomcell.HasAtomCell] and the concrete [`AtomCell`][atomlib.atomcell.AtomCell],
which combines the functionality of [`HasAtoms`][atomlib.atoms.HasAtoms] and [`HasCell`][atomlib.cell.HasCell].
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
import copy
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike
import polars
import polars.dataframe.group_by

from .bbox import BBox3D
from .types import VecLike, to_vec3, ParamSpec, Concatenate, Self
from .transform import LinearTransform3D, AffineTransform3D, Transform3D, IntoTransform3D
from .cell import CoordinateFrame, HasCell, Cell
from .atoms import HasAtoms, Atoms, IntoAtoms, AtomSelection, AtomValues
from .atoms import IntoExpr, IntoExprColumn, FillNullStrategy, RollingInterpolationMethod

# pyright: reportImportCycles=false
from .mixins import AtomCellIOMixin


AtomCellT = t.TypeVar('AtomCellT', bound='AtomCell')
HasAtomCellT = t.TypeVar('HasAtomCellT', bound='HasAtomCell')
P = ParamSpec('P')
T = t.TypeVar('T')


def _fwd_atoms_get(f: t.Callable[P, T]) -> t.Callable[P, T]:
    """Forward getter method on `HasAtomCell` to method on `HasAtoms`"""
    def inner(self, *args, frame: t.Optional[CoordinateFrame] = None, **kwargs):
        return getattr(self.get_atoms(frame), f.__name__)(*args, **kwargs)

    return t.cast(t.Callable[P, T], inner)


def _fwd_atoms_transform(f: t.Callable[P, T]) -> t.Callable[P, T]:
    """Forward transformation method on `HasAtomCell` to method on `HasAtoms`"""
    def inner(self, *args, frame: t.Optional[CoordinateFrame] = None, **kwargs):
        return self.with_atoms(self._transform_atoms_in_frame(frame, lambda atoms: getattr(atoms, f.__name__)(*args, **kwargs)))

    return t.cast(t.Callable[P, T], inner)


class HasAtomCell(HasAtoms, HasCell, abc.ABC):
    @abc.abstractmethod
    def get_frame(self) -> CoordinateFrame:
        """Get the coordinate frame atoms are stored in."""
        ...

    @abc.abstractmethod
    def with_atoms(self, atoms: HasAtoms, frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Replace the atoms in `self`. If no coordinate frame is specified, keep the coordinate frame unchanged.
        """
        ...

    def with_cell(self, cell: Cell) -> Self:
        """
        Replace the cell in `self`, without touching the atomic coordinates.
        """
        return self.to_frame('local').with_cell(cell)

    def get_atomcell(self) -> AtomCell:
        frame = self.get_frame()
        return AtomCell(self.get_atoms(frame), self.get_cell(), frame=frame, keep_frame=True)

    @abc.abstractmethod
    def get_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> Atoms:
        """Get atoms contained in `self`, in the given coordinate frame."""
        ...

    def bbox_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> BBox3D:
        """Return the bounding box of all the atoms in `self`, in the given coordinate frame."""
        return self.get_atoms(frame).bbox()

    def bbox(self, frame: CoordinateFrame = 'local') -> BBox3D:
        """
        Return the combined bounding box of the cell and atoms in the given coordinate system.
        To get the cell or atoms bounding box only, use [`bbox_cell`][atomlib.atomcell.HasAtomCell.bbox_cell] or [`bbox_atoms`][atomlib.atomcell.HasAtomCell.bbox_atoms].
        """
        return self.bbox_atoms(frame) | self.bbox_cell(frame)

    # transformation

    def _transform_atoms_in_frame(self, frame: t.Optional[CoordinateFrame], f: t.Callable[[Atoms], Atoms]) -> Atoms:
        # ugly code
        if frame is None or frame == self.get_frame():
            return f(self.get_atoms())
        return f(self.get_atoms(frame)).transform(self.get_transform(self.get_frame(), frame))

    def to_frame(self, frame: CoordinateFrame) -> Self:
        """Convert the stored Atoms to the given coordinate frame."""
        return self.with_atoms(self.get_atoms(frame), frame)

    def transform_atoms(self, transform: IntoTransform3D, selection: t.Optional[AtomSelection] = None, *,
                        frame: CoordinateFrame = 'local', transform_velocities: bool = False) -> Self:
        """
        Transform the atoms in `self` by `transform`.
        If `selection` is given, only transform the atoms in `selection`.
        """
        transform = self.change_transform(Transform3D.make(transform), self.get_frame(), frame)
        return self.with_atoms(self.get_atoms(self.get_frame()).transform(transform, selection, transform_velocities=transform_velocities))

    def transform_cell(self, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> Self:
        """
        Apply the given transform to the unit cell, without changing atom positions.
        The transform is applied in coordinate frame 'frame'.
        """
        return self.with_cell(self.get_cell().transform_cell(transform, frame=frame))

    def transform(self, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> Self:
        if isinstance(transform, Transform3D) and not isinstance(transform, AffineTransform3D):
            raise ValueError("Non-affine transforms cannot change the box dimensions. Use 'transform_atoms' instead.")
        # TODO: cleanup once tests pass
        # coordinate change the transform into atomic coordinates
        new_cell = self.get_cell().transform_cell(transform, frame)
        transform = self.get_cell().change_transform(transform, self.get_frame(), frame)
        return self.with_atoms(self.get_atoms().transform(transform), self.get_frame()).with_cell(new_cell)

    # crop methods

    def crop(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
             frame: CoordinateFrame = 'local') -> Self:
        """
        Crop atoms and cell to the given extents. For a non-orthogonal
        cell, this must be specified in cell coordinates. This
        function implicity `explode`s the cell as well.

        To crop atoms only, use `crop_atoms` instead.
        """

        cell = self.get_cell().crop(x_min, x_max, y_min, y_max, z_min, z_max, frame=frame)
        atoms = self._transform_atoms_in_frame(frame, lambda atoms: atoms.crop_atoms(x_min, x_max, y_min, y_max, z_min, z_max))
        return self.with_cell(cell).with_atoms(atoms)

    def crop_atoms(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
                   y_min: float = -numpy.inf, y_max: float = numpy.inf,
                   z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
                   frame: CoordinateFrame = 'local') -> Self:
        atoms = self._transform_atoms_in_frame(frame, lambda atoms: atoms.crop_atoms(x_min, x_max, y_min, y_max, z_min, z_max))
        return self.with_atoms(atoms)

    def crop_to_box(self, eps: float = 1e-5) -> Self:
        atoms = self._transform_atoms_in_frame('cell_box', lambda atoms: atoms.crop_atoms(*([-eps, 1-eps]*3)))
        return self.with_atoms(atoms)

    def wrap(self, eps: float = 1e-5) -> Self:
        """Wrap atoms around the cell boundaries."""
        return self.with_atoms(self._transform_atoms_in_frame('cell_box', lambda a: a._wrap(eps)))

    def _repeat_to_contain(self, pts: numpy.ndarray, pad: int = 0, frame: CoordinateFrame = 'cell_frac') -> Self:
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

    def repeat(self, n: t.Union[int, VecLike]) -> Self:
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

    def repeat_to(self, size: VecLike, crop: t.Union[bool, t.Sequence[bool]] = False) -> Self:
        """
        Repeat the cell so it is at least `size` along the crystal's axes.

        If `crop`, then crop the cell to exactly `size`. This may break periodicity.
        `crop` may be a vector, in which case you can specify cropping only along some axes.
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

    def repeat_x(self, n: int) -> Self:
        """Tile the cell in the x axis."""
        return self.repeat((n, 1, 1))

    def repeat_y(self, n: int) -> Self:
        """Tile the cell in the y axis."""
        return self.repeat((1, n, 1))

    def repeat_z(self, n: int) -> Self:
        """Tile the cell in the z axis."""
        return self.repeat((1, 1, n))

    def repeat_to_x(self, size: float, crop: bool = False) -> Self:
        """Repeat the cell so it is at least size `size` along the x axis."""
        return self.repeat_to([size, 0., 0.], [crop, False, False])

    def repeat_to_y(self, size: float, crop: bool = False) -> Self:
        """Repeat the cell so it is at least size `size` along the y axis."""
        return self.repeat_to([0., size, 0.], [False, crop, False])

    def repeat_to_z(self, size: float, crop: bool = False) -> Self:
        """Repeat the cell so it is at least size `size` along the z axis."""
        return self.repeat_to([0., 0., size], [False, False, crop])

    def repeat_to_aspect(self, plane: t.Literal['xy', 'xz', 'yz'] = 'xy', *,
                         aspect: float = 1., min_size: t.Optional[VecLike] = None,
                         max_size: t.Optional[VecLike] = None) -> Self:
        """
        Repeat to optimize the aspect ratio in `plane`,
        while staying above `min_size` and under `max_size`.
        """
        if min_size is None:
            min_n = numpy.array([1, 1, 1], numpy.int_)
        else:
            min_n = numpy.maximum(numpy.ceil(to_vec3(min_size) / self.box_size), 1).astype(numpy.int_)

        if max_size is None:
            max_n = 3 * min_n
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

        na = numpy.arange(min_n[indices[0]], max_n[indices[0]])
        nb = numpy.arange(min_n[indices[1]], max_n[indices[1]])
        (na, nb) = numpy.meshgrid(na, nb)

        aspects = na * self.box_size[indices[0]] / (nb * self.box_size[indices[1]])
        # cost function: log(aspect)^2  (so cost(0.5) == cost(2))
        min_i = numpy.argmin(numpy.log(aspects / aspect)**2)
        repeat = numpy.array([1, 1, 1], numpy.int_)
        repeat[indices] = na.flatten()[min_i], nb.flatten()[min_i]
        return self.repeat(repeat)

    def explode(self) -> Self:
        """Materialize repeated cells as one supercell."""
        frame = self.get_frame()

        return self.with_atoms(self.get_atoms('local'), 'local') \
            .with_cell(self.get_cell().explode()) \
            .to_frame(frame)

    def periodic_duplicate(self, eps: float = 1e-5) -> Self:
        """
        Add duplicate copies of atoms near periodic boundaries.

        For instance, an atom at a corner will be duplicated into 8 copies.
        This is mostly only useful for visualization.
        """
        frame_save = self.get_frame()
        self = self.to_frame('cell_box').wrap(eps=eps)

        for i in range(3):
            self = self.concat((self,
                self.filter(polars.col('coords').arr.get(i).abs() <= eps, frame='cell_box')
                    .transform_atoms(AffineTransform3D.translate([1. if i == j else 0. for j in range(3)]), frame='cell_box')
            ))

        return self.to_frame(frame_save)

    # add frame to some HasAtoms methods

    @_fwd_atoms_get
    def describe(self, percentiles: t.Union[t.Sequence[float], float, None] = (0.25, 0.5, 0.75), *,
                 interpolation: RollingInterpolationMethod = 'nearest',
                 frame: t.Optional[CoordinateFrame] = None) -> polars.DataFrame:
        """
        Return summary statistics for `self`. See [`DataFrame.describe`][polars.DataFrame.describe] for more information.

        Args:
          percentiles: List of percentiles/quantiles to include. Defaults to 25% (first quartile),
                       50% (median), and 75% (third quartile).

        Returns:
          A dataframe containing summary statistics (mean, std. deviation, percentiles, etc.) for each column.
        """
        ...

    @_fwd_atoms_transform
    def with_columns(self,
                     *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
                     frame: t.Optional[CoordinateFrame] = None,
                     **named_exprs: IntoExpr) -> Self:
        """Return a copy of `self` with the given columns added."""
        ...

    with_column = with_columns

    @_fwd_atoms_get
    def get_column(self, name: str, *, frame: t.Optional[CoordinateFrame] = None) -> polars.Series:
        """
        Get the specified column from `self`, raising [`polars.ColumnNotFoundError`][polars.exceptions.ColumnNotFoundError] if it's not present.

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
        ...

    @_fwd_atoms_get
    def get_columns(self, *, frame: t.Optional[CoordinateFrame] = None) -> t.List[polars.Series]:
        """
        Return all columns from `self` as a list of [`Series`][polars.Series].

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
        ...

    @_fwd_atoms_get
    def group_by(self, *by: t.Union[IntoExpr, t.Iterable[IntoExpr]],
                 maintain_order: bool = False, frame: t.Optional[CoordinateFrame] = None,
                 **named_by: IntoExpr) -> polars.dataframe.group_by.GroupBy:
        """
        Start a group by operation. See [`DataFrame.group_by`][polars.DataFrame.group_by] for more information.
        """
        ...

    def pipe(self: HasAtomCellT, function: t.Callable[Concatenate[HasAtomCellT, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Apply `function` to `self` (in method-call syntax)."""
        return function(self, *args, **kwargs)

    @_fwd_atoms_transform
    def filter(
        self,
        *predicates: t.Union[None, IntoExprColumn, t.Iterable[IntoExprColumn], bool, t.List[bool], numpy.ndarray],
        frame: t.Optional[CoordinateFrame] = None,
        **constraints: t.Any,
    ) -> Self:
        """Filter `self`, removing rows which evaluate to `False`."""
        ...

    @_fwd_atoms_transform
    def sort(
        self,
        by: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        *more_by: IntoExpr,
        descending: t.Union[bool, t.Sequence[bool]] = False,
        nulls_last: bool = False,
    ) -> Self:
        """
        Sort the atoms in `self` by the given columns/expressions.
        """
        ...

    @_fwd_atoms_transform
    def slice(self, offset: int, length: t.Optional[int] = None, *,
              frame: t.Optional[CoordinateFrame] = None) -> Self:
        """Return a slice of the rows in `self`."""
        ...

    @_fwd_atoms_transform
    def head(self, n: int = 5, *, frame: t.Optional[CoordinateFrame] = None) -> Self:
        """Return the first `n` rows of `self`."""
        ...

    @_fwd_atoms_transform
    def tail(self, n: int = 5, *, frame: t.Optional[CoordinateFrame] = None) -> Self:
        """Return the last `n` rows of `self`."""
        ...

    @_fwd_atoms_transform
    def fill_null(
        self, value: t.Any = None, strategy: t.Optional[FillNullStrategy] = None,
        limit: t.Optional[int] = None, matches_supertype: bool = True,
    ) -> Self:
        ...

    @_fwd_atoms_transform
    def fill_nan(self, value: t.Union[polars.Expr, int, float, None], *,
                 frame: t.Optional[CoordinateFrame] = None) -> Self:
        ...

    # TODO: partition_by

    @_fwd_atoms_get
    def select(
        self, *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        frame: t.Optional[CoordinateFrame] = None,
        **named_exprs: IntoExpr
    ) -> polars.DataFrame:
        """
        Select `exprs` from `self`, and return as a [`polars.DataFrame`][polars.DataFrame].

        Expressions may either be columns or expressions of columns.

        [polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
        """
        ...

    @_fwd_atoms_transform
    def select_props(
        self,
        *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        frame: t.Optional[CoordinateFrame] = None,
        **named_exprs: IntoExpr
    ) -> Self:
        """
        Select `exprs` from `self`, while keeping required columns.
        Doesn't affect the cell.

        Returns:
          A [`HasAtomCell`][atomlib.atomcell.HasAtomCell] filtered to contain
          the specified properties (as well as required columns).
        """
        ...

    @_fwd_atoms_get
    def try_select(
        self, *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        frame: t.Optional[CoordinateFrame] = None,
        **named_exprs: IntoExpr
    ) -> t.Optional[polars.DataFrame]:
        """
        Try to select `exprs` from `self`, and return as a [`polars.DataFrame`][polars.DataFrame].

        Expressions may either be columns or expressions of columns. Returns `None` if any
        columns are missing.

        [polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
        """
        ...

    @_fwd_atoms_transform
    def round_near_zero(self, tol: float = 1e-14, *,
                        frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Round atom position values near zero to zero.
        """
        ...

    @_fwd_atoms_get
    def coords(self, selection: t.Optional[AtomSelection] = None, *, frame: t.Optional[CoordinateFrame] = None) -> NDArray[numpy.float64]:
        """
        Return a `(N, 3)` ndarray of atom positions (dtype [`numpy.float64`][numpy.float64])
        in the given coordinate frame.
        """
        ...

    @_fwd_atoms_get
    def velocities(self, selection: t.Optional[AtomSelection] = None, *, frame: t.Optional[CoordinateFrame] = None) -> t.Optional[NDArray[numpy.float64]]:
        """
        Return a `(N, 3)` ndarray of atom velocities (dtype [`numpy.float64`][numpy.float64])
        in the given coordinate frame.
        """
        ...

    @t.overload
    def add_atom(self, elem: t.Union[int, str], x: ArrayLike, /, *,
                 y: None = None, z: None = None, frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> Self:
        ...

    @t.overload
    def add_atom(self, elem: t.Union[int, str], /,
                 x: float, y: float, z: float, *,
                 frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> Self:
        ...

    @_fwd_atoms_transform
    def add_atom(self, elem: t.Union[int, str], /,  # type: ignore (spurious)
                 x: t.Union[ArrayLike, float],
                 y: t.Optional[float] = None,
                 z: t.Optional[float] = None, *,
                 frame: t.Optional[CoordinateFrame] = None,
                 **kwargs: t.Any) -> Self:
        """
        Return a copy of `self` with an extra atom.

        By default, all extra columns present in `self` must be specified as `**kwargs`.

        Try to avoid calling this in a loop (Use [`concat`][atomlib.atomcell.HasAtomCell.concat] instead).
        """
        ...

    @_fwd_atoms_transform
    def with_index(self, index: t.Optional[AtomValues] = None, *,
                   frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Returns `self` with a row index added in column 'i' (dtype [`polars.Int64`][polars.datatypes.Int64]).
        If `index` is not specified, defaults to an existing index or a new index.
        """
        ...

    @_fwd_atoms_transform
    def with_wobble(self, wobble: t.Optional[AtomValues] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` with the given displacements in column 'wobble' (dtype [`polars.Float64`][polars.datatypes.Float64]).
        If `wobble` is not specified, defaults to the already-existing wobbles or 0.
        """
        ...

    @_fwd_atoms_transform
    def with_occupancy(self, frac_occupancy: t.Optional[AtomValues] = None, *,
                       frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return self with the given fractional occupancies (dtype [`polars.Float64`][polars.datatypes.Float64]).
        If `frac_occupancy` is not specified, defaults to the already-existing occupancies or 1.
        """
        ...

    @_fwd_atoms_transform
    def apply_wobble(self, rng: t.Union[numpy.random.Generator, int, None] = None,
                     frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Displace the atoms in `self` by the amount in the `wobble` column.
        `wobble` is interpretated as a mean-squared displacement, which is distributed
        equally over each axis.
        """
        ...

    @_fwd_atoms_transform
    def with_type(self, types: t.Optional[AtomValues] = None, *,
                  frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` with the given atom types in column 'type'.
        If `types` is not specified, use the already existing types or auto-assign them.

        When auto-assigning, each symbol is given a unique value, case-sensitive.
        Values are assigned from lowest atomic number to highest.
        For instance: `["Ag+", "Na", "H", "Ag"]` => `[3, 11, 1, 2]`
        """
        ...

    @_fwd_atoms_transform
    def with_mass(self, mass: t.Optional[ArrayLike] = None, *,
                  frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` with the given atom masses in column `'mass'`.
        If `mass` is not specified, use the already existing masses or auto-assign them.
        """
        ...

    @_fwd_atoms_transform
    def with_symbol(self, symbols: ArrayLike, selection: t.Optional[AtomSelection] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` with the given atomic symbols.
        """
        ...

    @_fwd_atoms_transform
    def with_coords(self, pts: ArrayLike, selection: t.Optional[AtomSelection] = None, *,
                    frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` replaced with the given atomic positions.
        """
        ...

    @_fwd_atoms_transform
    def with_velocity(self, pts: t.Optional[ArrayLike] = None,
                      selection: t.Optional[AtomSelection] = None, *,
                      frame: t.Optional[CoordinateFrame] = None) -> Self:
        """
        Return `self` replaced with the given atomic velocities.
        If `pts` is not specified, use the already existing velocities or zero.
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

    def with_cell(self, cell: Cell) -> Self:
        return self.__class__(self.atoms, cell, frame=self.frame, keep_frame=True)

    def get_atoms(self, frame: t.Optional[CoordinateFrame] = None) -> Atoms:
        """Get atoms contained in ``self``, in the given coordinate frame."""

        if frame is None or frame == self.get_frame():
            return self.atoms
        return self.atoms.transform(self.get_transform(frame, self.get_frame()))

    def with_atoms(self, atoms: HasAtoms, frame: t.Optional[CoordinateFrame] = None) -> Self:
        frame = frame if frame is not None else self.frame
        return self.__class__(atoms.get_atoms(), cell=self.cell, frame=frame, keep_frame=True)
        #return replace(self, atoms=atoms, frame = frame if frame is not None else self.frame, keep_frame=True)

    def get_frame(self) -> CoordinateFrame:
        """Get the coordinate frame atoms are stored in."""
        return self.frame

    @classmethod
    def _combine_metadata(cls: t.Type[AtomCellT], *atoms: HasAtoms, n: t.Optional[int] = None) -> AtomCellT:
        """
        When combining multiple [`HasAtoms`][atomlib.atoms.HasAtoms], check that they are compatible with each other,
        and return a 'representative' which best represents the combined metadata.
        Implementors should treat [`Atoms`][atomlib.atoms.Atoms] as acceptable, but having no metadata.
        """
        if n is not None:
            rep = atoms[n]
            if not isinstance(rep, AtomCell):
                raise ValueError(f"Atoms #{n} has no cell")
        else:
            atom_cells = [a for a in atoms if isinstance(a, AtomCell)]
            if len(atom_cells) == 0:
                raise TypeError(f"No AtomCells to combine")
            rep = atom_cells[0]
            if not all(a.cell == rep.cell for a in atom_cells[1:]):
                raise TypeError(f"Can't combine AtomCells with different cells")

        return cls(Atoms.empty(), frame=rep.frame, cell=rep.cell)

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

    def orthogonalize(self) -> OrthoCell:
        if self.is_orthogonal():
            return OrthoCell(self.atoms, self.cell, frame=self.frame)
        raise NotImplementedError()

    def clone(self: AtomCellT) -> AtomCellT:
        """Make a deep copy of `self`."""
        return self.__class__(**{field.name: copy.deepcopy(getattr(self, field.name)) for field in fields(self)})

    def assert_equal(self, other: t.Any):
        """Assert this structure is equal to `other`"""
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
