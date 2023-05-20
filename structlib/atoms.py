"""
Raw atoms collection

This module defines :class:`Atoms`, which holds a collection of atoms
with no cell or periodicity. :class:`Atoms` is essentially a wrapper
around :class:`polars.DataFrame`.
"""

from __future__ import annotations

import logging
from functools import reduce
import warnings
import operator
import abc
import typing as t

import numpy
import scipy.spatial
from numpy.typing import ArrayLike, NDArray
import polars
import polars.datatypes

from .types import to_vec3, VecLike
from .bbox import BBox3D
from .elem import get_elem, get_sym, get_mass
from .transform import Transform3D, IntoTransform3D, AffineTransform3D
from .util import map_some
from .cell import Cell
from .mixins import AtomsIOMixin


_COLUMN_DTYPES: t.Mapping[str, t.Type[polars.DataType]] = {
    'x': polars.Float64,
    'y': polars.Float64,
    'z': polars.Float64,
    'v_x': polars.Float64,
    'v_y': polars.Float64,
    'v_z': polars.Float64,
    'elem': polars.Int8,
    'mass': polars.Float32,
}

SchemaDict = t.Mapping[str, t.Union[t.Type[polars.DataType], polars.DataType]]
UniqueKeepStrategy = t.Literal['first', 'last']
ConcatMethod = t.Literal['horizontal', 'vertical', 'diagonal', 'inner']


# pyright: reportImportCycles=false
if t.TYPE_CHECKING:  # pragma: no cover
    from .atomcell import AtomCell

    class ColumnNotFoundError(Exception):
        ...
else:
    try:
        # polars 0.16
        ColumnNotFoundError = polars.exceptions.ColumnNotFoundError
    except AttributeError:
        # polars 0.15
        ColumnNotFoundError = polars.NotFoundError


def _is_abstract(cls: t.Type) -> bool:
    return bool(getattr(cls, "__abstractmethods__", False))


def _values_to_series(df: polars.DataFrame, selection: AtomSelection, ty: t.Type[polars.DataType]) -> polars.Series:
    if isinstance(selection, polars.Series):
        return selection.cast(ty)
    if isinstance(selection, polars.Expr):
        return df.select(selection.cast(ty)).to_series()

    selection = numpy.broadcast_to(selection, len(df))
    return polars.Series(selection, dtype=ty)


def _values_to_expr(values: AtomValues, ty: t.Type[polars.DataType]) -> polars.Expr:
    if isinstance(values, polars.Expr):
        return values.cast(ty)
    if isinstance(values, polars.Series):
        return polars.lit(values, dtype=ty)
    if isinstance(values, t.Mapping):
        return polars.col('symbol').apply(lambda s: values[s], ty)  # type: ignore
    arr = numpy.asarray(values)
    return polars.lit(polars.Series(arr, dtype=ty) if arr.size > 1 else values)


def _selection_to_series(df: t.Union[polars.DataFrame, HasAtoms], selection: AtomSelection) -> polars.Series:
    if isinstance(df, HasAtoms):
        df = df.get_atoms().inner
    return _values_to_series(df, selection, ty=polars.Boolean)


def _selection_to_expr(selection: AtomSelection) -> polars.Expr:
    return _values_to_expr(selection, ty=polars.Boolean)


def _selection_to_numpy(df: t.Union[polars.DataFrame, HasAtoms], selection: AtomSelection) -> NDArray[numpy.bool_]:
    series = _selection_to_series(df, selection)
    try:
        import pyarrow  # type: ignore
    except ImportError:
        # workaround without pyarrow
        return numpy.array(series.to_list(), dtype=numpy.bool_)
    return series.to_numpy(zero_copy_only=False)


def _select_schema(df: t.Union[polars.DataFrame, HasAtoms], schema: SchemaDict) -> polars.DataFrame:
    """
    Select columns from ``self`` and cast to the given schema.
    """
    try:
        return df.select([
            polars.col(col).cast(ty, strict=True)
            for (col, ty) in schema.items()
        ])
    except (polars.ComputeError, ColumnNotFoundError):
        raise TypeError(f"Failed to cast '{df.__class__.__name__}' with schema '{df.schema}' to schema '{schema}'.")


HasAtomsT = t.TypeVar('HasAtomsT', bound='HasAtoms')


class HasAtoms(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    # abstract methods

    @abc.abstractmethod
    def get_atoms(self, frame: t.Literal['local'] = 'local') -> Atoms:
        """Get atoms contained in `self`. This should be a low cost method."""
        ...

    @abc.abstractmethod
    def with_atoms(self: HasAtomsT, atoms: HasAtoms, frame: t.Literal['local'] = 'local') -> HasAtomsT:
        ...

    @classmethod
    @abc.abstractmethod
    def _combine_metadata(cls: t.Type[HasAtomsT], *atoms: HasAtoms) -> HasAtomsT:
        """
        When combining multiple ``HasAtoms``, check that they are compatible with each other,
        and return a 'representative' which best represents the combined metadata.
        Implementors should treat `Atoms` as acceptable, but having no metadata.
        """
        ...

    def _get_frame(self) -> polars.DataFrame:
        return self.get_atoms().inner

    # dataframe methods

    @property
    def columns(self) -> t.Sequence[str]:
        """Return the columns in `self`."""
        return self._get_frame().columns

    @property
    def schema(self) -> SchemaDict:
        """Return the schema of `self`."""
        return t.cast(SchemaDict, self._get_frame().schema)

    def with_column(self: HasAtomsT, column: t.Union[polars.Series, polars.Expr]) -> HasAtomsT:
        """Return a copy of ``self`` with the given column added."""
        return self.with_atoms(Atoms(self._get_frame().with_columns((column,)), _unchecked=True))

    def with_columns(self: HasAtomsT,
                     exprs: t.Union[t.Literal[None], polars.Series, polars.Expr, t.Sequence[t.Union[polars.Series, polars.Expr]]],
                     **named_exprs: t.Union[polars.Expr, polars.Series]) -> HasAtomsT:
        """Return a copy of ``self`` with the given columns added."""
        return self.with_atoms(Atoms(self._get_frame().with_columns(exprs, **named_exprs), _unchecked=True))

    def get_column(self, name: str) -> polars.Series:
        """Get the specified column from ``self``, raising :py:`polars.NotFoundError` if it's not present."""
        return self._get_frame().get_column(name)

    def filter(self: HasAtomsT, selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
        """Filter ``self``, removing rows which evaluate to ``False``."""
        if selection is None:
            return self
        return self.with_atoms(Atoms(self._get_frame().filter(_selection_to_expr(selection)), _unchecked=True))

    def select(self, exprs: t.Union[str, polars.Expr, polars.Series, t.Sequence[t.Union[str, polars.Expr, polars.Series]]]
    ) -> polars.DataFrame:
        """
        Select ``exprs`` from ``self``, and return as a :py:class:`polars.DataFrame`.

        Expressions may either be columns or expressions of columns.
        """
        return self._get_frame().select(exprs)

    def sort(self: HasAtomsT, by: t.Union[str, polars.Expr, t.List[str], t.List[polars.Expr]], descending: t.Union[bool, t.List[bool]] = False) -> HasAtomsT:
        return self.with_atoms(Atoms(self._get_frame().sort(by, descending=descending), _unchecked=True))

    @classmethod
    def concat(cls: t.Type[HasAtomsT],
               atoms: t.Union[HasAtomsT, IntoAtoms, t.Iterable[t.Union[HasAtomsT, IntoAtoms]]], *,
               rechunk: bool = True, how: ConcatMethod = 'vertical') -> HasAtomsT:
        # this method is tricky. It needs to accept raw Atoms, as well as HasAtoms of the
        # same type as ``cls``.
        if _is_abstract(cls):
            raise TypeError(f"concat() must be called on a concrete class.")

        if isinstance(atoms, HasAtoms):
            atoms = (atoms,)
        dfs = [a.get_atoms('local').inner if isinstance(a, HasAtoms) else Atoms(a).inner for a in atoms]
        representative = cls._combine_metadata(*(a for a in atoms if isinstance(a, HasAtoms)))

        if len(dfs) == 0:
            return representative.with_atoms(Atoms.empty(), 'local')

        if how == 'inner':
            cols = reduce(operator.and_, (df.schema.keys() for df in dfs))
            schema = t.cast(SchemaDict, {col: dfs[0].schema[col] for col in cols})
            if len(schema) == 0:
                raise ValueError(f"Atoms have no columns in common")

            dfs = [_select_schema(df, schema) for df in dfs]
            how = 'vertical'

        return representative.with_atoms(Atoms(polars.concat(dfs, rechunk=rechunk, how=how)), 'local')

    # some helpers we add

    def select_schema(self, schema: SchemaDict) -> polars.DataFrame:
        """
        Select columns from ``self`` and cast to the given schema.
        Raises :py:`TypeError` if a column is not found or if it can't be cast.
        """
        return _select_schema(self, schema)

    def try_select(self, exprs: t.Union[str, polars.Expr, polars.Series, t.Sequence[t.Union[str, polars.Expr, polars.Series]]]
    ) -> t.Optional[polars.DataFrame]:
        """
        Try to select ``exprs`` from ``self``, and return as a ``DataFrame``.

        Expressions may either be columns or expressions of columns.
        Return ``None`` if any columns are missing.
        """
        try:
            return self._get_frame().select(exprs)
        except ColumnNotFoundError:
            return None

    def try_get_column(self, name: str) -> t.Optional[polars.Series]:
        """Try to get a column from `self`, returning `None` if it doesn't exist."""
        try:
            return self.get_column(name)
        except ColumnNotFoundError:
            return None

    def assert_equal(self, other: t.Any):
        assert isinstance(other, HasAtoms)
        assert self.schema == other.schema
        for (col, dtype) in self.schema.items():
            if dtype in (polars.Float32, polars.Float64):
                numpy.testing.assert_array_almost_equal(self[col].view(ignore_nulls=True), other[col].view(ignore_nulls=True), 5)
            else:
                assert (self[col] == other[col]).all()

    # dunders

    def __len__(self) -> int:
        """Return the number of atoms in `self`."""
        return self._get_frame().__len__()

    def __contains__(self, col: str) -> bool:
        """Return whether `self` contains the given column."""
        return col in self.columns

    def __add__(self: HasAtomsT, other: IntoAtoms) -> HasAtomsT:
        return self.__class__.concat((self, other), how='inner')

    def __radd__(self: HasAtomsT, other: IntoAtoms) -> HasAtomsT:
        return self.__class__.concat((other, self), how='inner')

    __getitem__ = get_column

    # atoms-specific methods

    def bbox_atoms(self) -> BBox3D:
        """Return the bounding box of all the atoms in ``self``."""
        return BBox3D.from_pts(self.coords())

    bbox = bbox_atoms

    def transform_atoms(self: HasAtomsT, transform: IntoTransform3D, selection: t.Optional[AtomSelection] = None, *, transform_velocities: bool = False) -> HasAtomsT:
        """
        Transform the atoms in `self` by `transform`.
        If `selection` is given, only transform the atoms in `selection`.
        """
        transform = Transform3D.make(transform)
        selection = map_some(lambda s: _selection_to_series(self, s), selection)
        transformed = self.with_coords(Transform3D.make(transform) @ self.coords(selection), selection)
        # try to transform velocities as well
        if transform_velocities and (velocities := self.velocities(selection)) is not None:
            return transformed.with_velocity(transform.transform_vec(velocities), selection)
        return transformed

    transform = transform_atoms

    def round_near_zero(self: HasAtomsT, tol: float = 1e-14) -> HasAtomsT:
        """
        Round atom position values near zero to zero.
        """
        return self.with_columns(tuple(
            polars.when(col.abs() >= tol).then(col).otherwise(polars.lit(0.))
            for col in map(polars.col, ('x', 'y', 'z'))
        ))

    def crop(self: HasAtomsT, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf) -> HasAtomsT:
        """
        Crop, removing all atoms outside of the specified region, inclusive.
        """

        return self.filter(
            (polars.col('x') >= x_min) & (polars.col('x') <= x_max) &
            (polars.col('y') >= y_min) & (polars.col('y') <= y_max) &
            (polars.col('z') >= z_min) & (polars.col('z') <= z_max)
        )

    crop_atoms = crop

    def deduplicate(self: HasAtomsT, tol: float = 1e-3, subset: t.Iterable[str] = ('x', 'y', 'z', 'symbol'),
                    keep: UniqueKeepStrategy = 'first') -> HasAtomsT:
        """
        De-duplicate atoms in `self`. Atoms of the same `symbol` that are closer than `tolerance`
        to each other (by Euclidian distance) will be removed, leaving only the atom specified by
        `keep` (defaults to the first atom).

        If `subset` is specified, only those columns will be included while assessing duplicates.
        Floating point columns other than 'x', 'y', and 'z' will not by toleranced.
        """

        cols = set((subset,) if isinstance(subset, str) else subset)

        indices = numpy.arange(len(self))

        spatial_cols = cols.intersection(('x', 'y', 'z'))
        cols -= spatial_cols
        if len(spatial_cols) > 0:
            coords = self.select(list(spatial_cols)).to_numpy()
            print(coords.shape)
            tree = scipy.spatial.KDTree(coords)

            # TODO This is a bad algorithm
            while True:
                changed = False
                for (i, j) in tree.query_pairs(tol, 2.):
                    # whenever we encounter a pair, ensure their index matches
                    i_i, i_j = indices[[i, j]]
                    if i_i != i_j:
                        indices[i] = indices[j] = min(i_i, i_j)
                        changed = True
                if not changed:
                    break

        self = self.with_column(polars.Series('_unique_pts', indices))
        cols.add('_unique_pts')
        new = Atoms(self._get_frame().unique(subset=list(cols), keep=keep).drop('_unique_pts'), _unchecked=True)
        return self.with_atoms(new)

    unique = deduplicate

    def with_bounds(self, cell_size: t.Optional[VecLike] = None, cell_origin: t.Optional[VecLike] = None) -> 'AtomCell':
        """
        Return a periodic cell with the given orthogonal cell dimensions.

        If cell_size is not specified, it will be assumed (and may be incorrect).
        """
        # TODO: test this
        from .atomcell import AtomCell

        if cell_size is None:
            warnings.warn("Cell boundary unknown. Defaulting to cell BBox")
            cell_size = self.bbox().size
            cell_origin = self.bbox().min

        # TODO test this origin code
        cell = Cell.from_unit_cell(cell_size)
        if cell_origin is not None:
            cell = cell.transform_cell(AffineTransform3D.translate(to_vec3(cell_origin)))

        return AtomCell(self.get_atoms(), cell, frame='local')

    # property getters and setters

    def coords(self, selection: t.Optional[AtomSelection] = None) -> NDArray[numpy.float64]:
        """Returns a (N, 3) ndarray of atom coordinates (dtype `numpy.float64`)."""
        # TODO find a way to get a view
        return self.filter(selection).select(('x', 'y', 'z')).to_numpy().astype(numpy.float64)

    def velocities(self, selection: t.Optional[AtomSelection] = None) -> t.Optional[NDArray[numpy.float64]]:
        """Returns a (N, 3) ndarray of atom velocities (dtype `numpy.float64`)."""
        if selection is not None:
            self = self.filter(selection)
        return map_some(lambda df: df.to_numpy().astype(numpy.float64),
                        self.try_select(('v_x', 'v_y', 'v_z')))

    def types(self) -> t.Optional[polars.Series]:
        """Returns a `Series` of atom types (dtype polars.Int32)."""
        return self.try_get_column('type')

    def masses(self) -> t.Optional[polars.Series]:
        """Returns a `Series` of atom masses (dtype polars.Float32)."""
        return self.try_get_column('mass')

    @t.overload
    def add_atom(self: HasAtomsT, elem: t.Union[int, str], x: ArrayLike, /, *,
                 y: None = None, z: None = None,
                 **kwargs: t.Any) -> HasAtomsT:
        ...

    @t.overload
    def add_atom(self: HasAtomsT, elem: t.Union[int, str], /,
                 x: float, y: float, z: float,
                 **kwargs: t.Any) -> HasAtomsT:
        ...

    def add_atom(self: HasAtomsT, elem: t.Union[int, str], /,
                 x: t.Union[ArrayLike, float],
                 y: t.Optional[float] = None,
                 z: t.Optional[float] = None,
                 **kwargs: t.Any) -> HasAtomsT:
        """
        Return a copy of ``self`` with an extra atom.

        By default, all extra columns present in ``self`` must be specified as ``**kwargs``.

        Try to avoid calling this in a loop (Use :function:`Atoms.concat` instead).
        """
        if isinstance(elem, int):
            kwargs.update(elem=elem)
        else:
            kwargs.update(symbol=elem)
        if hasattr(x, '__len__') and len(x) > 1:  # type: ignore
            (x, y, z) = to_vec3(x)
        elif y is None or z is None:
            raise ValueError(f"Must specify vector of positions or x, y, & z.")

        sym = get_sym(elem) if isinstance(elem, int) else elem
        d: t.Dict[str, t.Any] = {'x': x, 'y': y, 'z': z, 'symbol': sym, **kwargs}
        return self.concat(
            (self, Atoms(d).select_schema(self.schema)),
            how='vertical'
        )

    @t.overload
    def pos(self, x: t.Sequence[t.Optional[float]], /, *,
            y: None = None, z: None = None,
            tol: float = 1e-6, **kwargs: t.Any) -> polars.Expr:
        ...

    @t.overload
    def pos(self, x: t.Optional[float] = None, y: t.Optional[float] = None, z: t.Optional[float] = None, *,
            tol: float = 1e-6, **kwargs: t.Any) -> polars.Expr:
        ...

    def pos(self,
            x: t.Union[t.Sequence[t.Optional[float]], float, None] = None,
            y: t.Optional[float] = None, z: t.Optional[float] = None, *,
            tol: float = 1e-6, **kwargs: t.Any) -> polars.Expr:
        """
        Select all atoms at a given position.

        Formally, returns all atoms within a cube of radius ``tol``
        centered at ``(x,y,z)``, exclusive of the cube's surface.

        Additional parameters given as ``kwargs`` will be checked
        as additional parameters (with strict equality).
        """

        if isinstance(x, t.Sequence):
            (x, y, z) = x

        tol = abs(float(tol))
        selection = polars.lit(True)
        if x is not None:
            selection &= (x - tol < polars.col('x')) & (polars.col('x') < x + tol)
        if y is not None:
            selection &= (y - tol < polars.col('y')) & (polars.col('y') < y + tol)
        if z is not None:
            selection &= (z - tol < polars.col('z')) & (polars.col('z') < z + tol)
        for (col, val) in kwargs.items():
            selection &= (polars.col(col) == val)

        return selection

    def with_index(self: HasAtomsT, index: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Returns `self` with a row index added in column 'i' (dtype polars.Int64).
        If `index` is not specified, defaults to an existing index or a new index.
        """
        if index is None and 'i' in self.columns:
            return self
        if index is None:
            index = numpy.arange(len(self), dtype=numpy.int64)
        return self.with_column(_values_to_expr(index, polars.Int64).alias('i'))

    def with_wobble(self: HasAtomsT, wobble: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Return `self` with the given displacements in column 'wobble' (dtype polars.Float64).
        If `wobble` is not specified, defaults to the already-existing wobbles or 0.
        """
        if wobble is None and 'wobble' in self.columns:
            return self
        wobble = 0. if wobble is None else wobble
        return self.with_column(_values_to_expr(wobble, polars.Float64).alias('wobble'))

    def with_occupancy(self: HasAtomsT, frac_occupancy: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Return self with the given fractional occupancies. If `frac_occupancy` is not specified,
        defaults to the already-existing occupancies or 1.
        """
        if frac_occupancy is None and 'frac_occupancy' in self.columns:
            return self
        frac_occupancy = 1. if frac_occupancy is None else frac_occupancy
        return self.with_column(_values_to_expr(frac_occupancy, polars.Float64).alias('frac_occupancy'))

    def apply_wobble(self: HasAtomsT, rng: t.Union[numpy.random.Generator, int, None] = None) -> HasAtomsT:
        """
        Displace the atoms in `self` by the amount in the `wobble` column.
        `wobble` is interpretated as a mean-squared displacement, which is distributed
        equally over each axis.
        """
        if 'wobble' not in self.columns:
            return self
        rng = numpy.random.default_rng(seed=rng)

        stddev = self.select((polars.col('wobble') / 3.).sqrt()).to_series().to_numpy()
        coords = self.coords()
        coords += stddev[:, None] * rng.standard_normal(coords.shape)
        return self.with_coords(coords)

    def apply_occupancy(self: HasAtomsT, rng: t.Union[numpy.random.Generator, int, None] = None) -> HasAtomsT:
        """
        For each atom in `self`, use its `frac_occupancy` to randomly decide whether to remove it.
        """
        if 'frac_occupancy' not in self.columns:
            return self
        rng = numpy.random.default_rng(seed=rng)

        frac = self.select('frac_occupancy').to_series().to_numpy()
        choice = rng.binomial(1, frac).astype(numpy.bool_)
        return self.filter(polars.lit(choice))

    def with_type(self: HasAtomsT, types: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Return `self` with the given atom types in column 'type'.
        If `types` is not specified, use the already existing types or auto-assign them.

        When auto-assigning, each symbol is given a unique value, case-sensitive.
        Values are assigned from lowest atomic number to highest.
        For instance: ["Ag+", "Na", "H", "Ag"] => [3, 11, 1, 2]
        """
        if types is not None:
            return self.with_column(_values_to_expr(types, polars.Int32).alias('type'))
        if 'type' in self.columns:
            return self

        unique = Atoms(self._get_frame().unique(maintain_order=False, subset=['elem', 'symbol']).sort(['elem', 'symbol']), _unchecked=True)
        new = self.with_column(polars.Series('type', values=numpy.zeros(len(self)), dtype=polars.Int32))

        logging.warning("Auto-assigning element types")
        for (i, (elem, sym)) in enumerate(unique.select(('elem', 'symbol')).rows()):
            print(f"Assigning type {i+1} to element '{sym}'")
            new = new.with_column(polars.when((polars.col('elem') == elem) & (polars.col('symbol') == sym))
                                        .then(polars.lit(i+1))
                                        .otherwise(polars.col('type'))
                                        .alias('type'))

        assert (new.get_column('type') == 0).sum() == 0
        return new

    def with_mass(self: HasAtomsT, mass: t.Optional[ArrayLike] = None) -> HasAtomsT:
        """
        Return `self` with the given atom masses in column 'mass'.
        If `mass` is not specified, use the already existing masses or auto-assign them.
        """
        if mass is not None:
            return self.with_column(_values_to_expr(mass, polars.Float32).alias('mass'))
        if 'mass' in self.columns:
            return self

        unique_elems = self.get_column('elem').unique()
        new = self.with_column(polars.Series('mass', values=numpy.zeros(len(self)), dtype=polars.Float32))

        logging.warning("Auto-assigning element masses")
        for elem in unique_elems:
            new = new.with_column(polars.when(polars.col('elem') == elem)
                                        .then(polars.lit(get_mass(elem)))
                                        .otherwise(polars.col('mass'))
                                        .alias('mass'))

        assert (new.get_column('mass').abs() < 1e-10).sum() == 0
        return new

    def with_symbol(self: HasAtomsT, symbols: ArrayLike, selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
        """
        Return `self` with the given atomic symbols.
        """
        if selection is not None:
            selection = _selection_to_numpy(self, selection)
            new_symbols = self.get_column('symbol')
            new_symbols[selection] = polars.Series(list(numpy.broadcast_to(symbols, len(selection))), dtype=polars.Utf8)
            symbols = new_symbols

        # TODO better cast here
        symbols = polars.Series('symbol', list(numpy.broadcast_to(symbols, len(self))), dtype=polars.Utf8)
        return self.with_columns((symbols, get_elem(symbols)))

    def with_coords(self: HasAtomsT, pts: ArrayLike, selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
        """
        Return `self` replaced with the given atomic positions.
        """
        if selection is not None:
            selection = _selection_to_numpy(self, selection)
            new_pts = self.coords()
            pts = numpy.atleast_2d(pts)
            assert pts.shape[-1] == 3
            new_pts[selection] = pts
            pts = new_pts

        pts = numpy.broadcast_to(pts, (len(self), 3))
        return self.with_columns((
            polars.Series(pts[:, 0], dtype=polars.Float64).alias('x'),
            polars.Series(pts[:, 1], dtype=polars.Float64).alias('y'),
            polars.Series(pts[:, 2], dtype=polars.Float64).alias('z'),
        ))

    def with_velocity(self: HasAtomsT, pts: t.Optional[ArrayLike] = None,
                      selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
        """
        Return `self` replaced with the given atomic velocities.
        If `pts` is not specified, use the already existing velocities or zero.
        """
        if pts is None:
            if all(col in self.columns for col in ('v_x', 'v_y', 'v_z')):
                return self
            pts = numpy.zeros((len(self), 3))
        else:
            pts = numpy.broadcast_to(pts, (len(self), 3))

        if selection is None:
            return self.with_columns((
                polars.Series(pts[:, 0], dtype=polars.Float64).alias('v_x'),
                polars.Series(pts[:, 1], dtype=polars.Float64).alias('v_y'),
                polars.Series(pts[:, 2], dtype=polars.Float64).alias('v_z'),
            ))

        selection = _selection_to_series(self, selection)
        return self.__class__(self.with_columns((
            self['v_x'].set_at_idx(selection, pts[:, 0]),  # type: ignore
            self['v_y'].set_at_idx(selection, pts[:, 1]),  # type: ignore
            self['v_z'].set_at_idx(selection, pts[:, 2]),  # type: ignore
        )))


class Atoms(AtomsIOMixin, HasAtoms):
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
    - wobble: Isotropic Debye-Waller mean-squared deviation (<u^2> = B*3/8pi^2, dimensions of [Length^2])
    - frac_occupancy: Fractional occupancy, [0., 1.]
    - mass: Atomic mass, in g/mol (approx. Da)
    - v_[xyz]: Atom velocities, dimensions of length/time
    - atom_type: Numeric atom type, as used by programs like LAMMPS
    """

    def __init__(self, data: t.Optional[IntoAtoms] = None, columns: t.Optional[t.Sequence[str]] = None,
                 orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None,
                 _unchecked: bool = False):
        self._bbox: t.Optional[BBox3D] = None

        if data is None:
            assert columns is None
            self.inner = polars.DataFrame([
                polars.Series('x', (), dtype=polars.Float64),
                polars.Series('y', (), dtype=polars.Float64),
                polars.Series('z', (), dtype=polars.Float64),
                polars.Series('elem', (), dtype=polars.Int8),
                polars.Series('symbol', (), dtype=polars.Utf8),
            ])
        elif isinstance(data, polars.DataFrame):
            self.inner = data
        elif isinstance(data, Atoms):
            self.inner = data.inner
            _unchecked = True
        else:
            self.inner = polars.DataFrame(data, schema=columns, orient=orient)  # type: ignore

        if not _unchecked:
            missing: t.Tuple[str] = tuple(set(['symbol', 'elem']) - set(self.columns))
            if len(missing) > 1:
                raise ValueError("'Atoms' missing columns 'elem' and/or 'symbol'.")
            # fill 'symbol' from 'elem' or vice-versa
            if missing == ('symbol',):
                self.inner = self.inner.with_columns(get_sym(self.inner['elem']))
            elif missing == ('elem',):
                self.inner = self.inner.with_columns(get_elem(self.inner['symbol']))

            # cast to standard dtypes
            self.inner = self.inner.with_columns([
                self.inner[col].cast(dtype)
                for (col, dtype) in _COLUMN_DTYPES.items() if col in self.inner
            ])

            self._validate_atoms()

    @staticmethod
    def empty() -> Atoms:
        """
        Return an empty Atoms with only the mandatory columns.
        """
        return Atoms()

    def _validate_atoms(self):
        missing = [col for col in ['x', 'y', 'z', 'elem', 'symbol'] if col not in self.columns]
        if len(missing):
            raise ValueError(f"'Atoms' missing column(s) {', '.join(map(repr, missing))}")

    def get_atoms(self, frame: t.Literal['local'] = 'local') -> Atoms:
        if frame != 'local':
            raise ValueError(f"Atoms without a cell only support the 'local' coordinate frame, not '{frame}'.")
        return self

    def with_atoms(self, atoms: HasAtoms, frame: t.Literal['local'] = 'local') -> Atoms:
        if frame != 'local':
            raise ValueError(f"Atoms without a cell only support the 'local' coordinate frame, not '{frame}'.")
        return atoms.get_atoms()

    @classmethod
    def _combine_metadata(cls: t.Type[Atoms], *atoms: HasAtoms) -> Atoms:
        return cls.empty()

    def bbox(self) -> BBox3D:
        """Return the bounding box of all the points in `self`."""
        if self._bbox is None:
            self._bbox = BBox3D.from_pts(self.coords())

        return self._bbox

    def __str__(self) -> str:
        return f"Atoms, {self.inner!s}"

    def __repr__(self) -> str:
        # real __repr__ that polars doesn't provide
        lines = ["Atoms(["]
        for (col, series) in self.inner.to_dict().items():
            lines.append(f"    Series({col!r}, {list(series)!r}, {series.dtype!r}),")
        lines.append("])")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle: bool) -> None:
        p.text('Atoms(...)') if cycle else p.text(str(self))


IntoAtoms = t.Union[t.Dict[str, t.Sequence[t.Any]], t.Sequence[t.Any], numpy.ndarray, polars.DataFrame, Atoms]
"""
A type convertible into an `Atoms`.
"""


AtomSelection = t.Union[polars.Series, polars.Expr, ArrayLike]
"""
Polars expression selecting a subset of atoms from an Atoms.
Can be used with many Atoms methods.
"""

AtomValues = t.Union[polars.Series, polars.Expr, ArrayLike, t.Mapping[str, t.Any]]
"""
Array, value, or polars expression mapping atoms to values.
Can be used with `with_*` methods on Atoms
"""

__all__ = [
    'Atoms', 'HasAtoms', 'IntoAtoms', 'AtomSelection', 'AtomValues',
]
