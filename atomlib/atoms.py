"""
Raw atoms collection

This module defines [`HasAtoms`][atomlib.atoms.HasAtoms] and the concrete [`Atoms`][atomlib.atoms.Atoms],
which holds a collection of atoms with no cell or periodicity. [`Atoms`][atomlib.atoms.Atoms]
is essentially a wrapper around a [`polars.DataFrame`][polars.DataFrame].

[polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
[polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from functools import reduce, wraps
import warnings
import operator
import abc
from io import StringIO
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import polars
import polars.dataframe.group_by
import polars.datatypes
import polars.interchange.dataframe
import polars.testing
import polars.type_aliases

from .types import to_vec3, VecLike, ParamSpec, Concatenate, TypeAlias
from .bbox import BBox3D
from .elem import get_elem, get_sym, get_mass
from .transform import Transform3D, IntoTransform3D, AffineTransform3D
from .cell import Cell
from .mixins import AtomsIOMixin

# pyright: reportImportCycles=false
if t.TYPE_CHECKING:  # pragma: no cover
    from .atomcell import AtomCell


_COLUMN_DTYPES: t.Mapping[str, t.Union[polars.DataType, t.Type[polars.DataType]]] = {
    'coords': polars.Array(polars.Float64, 3),
    'velocity': polars.Array(polars.Float64, 3),
    'elem': polars.Int8,
    'mass': polars.Float32,
    'i': polars.Int64,
    'wobble': polars.Float64,
    'frac_occupancy': polars.Float64,
    'type': polars.Int32,
    'symbol': polars.Utf8,
}
_REQUIRED_COLUMNS: t.Tuple[str, ...] = ('coords', 'elem', 'symbol')


def _is_abstract(cls: t.Type) -> bool:
    return bool(getattr(cls, "__abstractmethods__", False))


def _polars_to_numpy_dtype(dtype: t.Type[polars.DataType]) -> numpy.dtype:
    from polars.datatypes import dtype_to_ctype
    if dtype == polars.Boolean:
        return numpy.dtype(numpy.bool_)
    try:
        return numpy.dtype(dtype_to_ctype(dtype))
    except NotImplementedError:
        return numpy.dtype(object)


def _get_symbol_mapping(df: t.Union[polars.DataFrame, HasAtoms], mapping: t.Mapping[str, t.Any], ty: t.Type[polars.DataType]) -> polars.Expr:
    syms = df['symbol'].unique()
    if (missing := set(syms) - set(mapping.keys())):
        raise ValueError(f"Could not remap symbols {', '.join(map(repr, missing))}") 

    return polars.col('symbol').replace(mapping, default=None, return_dtype=ty)


def _values_to_expr(df: t.Union[polars.DataFrame, HasAtoms], values: AtomValues, ty: t.Type[polars.DataType]) -> polars.Expr:
    if isinstance(values, polars.Expr):
        return values.cast(ty)
    if isinstance(values, polars.Series):
        return polars.lit(values, dtype=ty)
    if isinstance(values, t.Mapping):
        return _get_symbol_mapping(df, values, ty)
    arr = numpy.asarray(values)
    return polars.lit(polars.Series(arr, dtype=ty) if arr.size > 1 else arr[()])


def _values_to_numpy(df: t.Union[polars.DataFrame, HasAtoms], values: AtomValues, ty: t.Type[polars.DataType]) -> numpy.ndarray:
    if isinstance(values, polars.Expr):
        values = df.select(values).to_series()
    elif isinstance(values, t.Mapping):
        values = df.select(_get_symbol_mapping(df, values, ty)).to_series()
        #if values.is_null().any():
        #    syms = df.select(polars.col('symbol').filter(values.is_null())).unique().to_series().to_list()
        #    raise ValueError(f"Could not remap symbols {', '.join(map(repr, syms))}") 
    if isinstance(values, polars.Series):
        if ty == polars.Boolean:
            # force conversion to numpy (unpacked) bool
            return values.cast(polars.UInt8).to_numpy().astype(numpy.bool_)
        return numpy.broadcast_to(values.cast(ty).to_numpy(), len(df))

    dtype = _polars_to_numpy_dtype(ty)
    return numpy.broadcast_to(numpy.asarray(values, dtype), len(df))


def _selection_to_expr(df: t.Union[polars.DataFrame, HasAtoms], selection: t.Optional[AtomSelection] = None) -> polars.Expr:
    if selection is None:
        return polars.lit(True, dtype=polars.Boolean)
    return _values_to_expr(df, selection, ty=polars.Boolean)


def _selection_to_numpy(df: t.Union[polars.DataFrame, HasAtoms], selection: t.Optional[AtomSelection]) -> NDArray[numpy.bool_]:
    if selection is None:
        return numpy.full_like(len(df), 1, dtype=numpy.bool_)
    return _values_to_numpy(df, selection, polars.Boolean)


def _select_schema(df: t.Union[polars.DataFrame, HasAtoms], schema: SchemaDict) -> polars.DataFrame:
    """
    Select columns from `self` and cast to the given schema.
    """
    try:
        return df.select([
            polars.col(col).cast(ty, strict=True)
            for (col, ty) in schema.items()
        ])
    except (polars.ComputeError, polars.ColumnNotFoundError):
        raise TypeError(f"Failed to cast '{df.__class__.__name__}' with schema '{df.schema}' to schema '{schema}'.")


def _with_columns_stacked(df: polars.DataFrame, cols: t.Sequence[str], out_col: str) -> polars.DataFrame:
    if not all(c in df for c in cols):
        return df

    i = df.get_column_index(cols[0])
    dtype = df[cols[0]].dtype

    arr = numpy.array(tuple(df[c].to_numpy() for c in cols)).T

    return df.drop(cols).insert_column(i, polars.Series(out_col, arr, polars.Array(dtype, arr.shape[-1])))


HasAtomsT = t.TypeVar('HasAtomsT', bound='HasAtoms')
P = ParamSpec('P')
T = t.TypeVar('T')


def _fwd_frame_map(
    f: t.Callable[Concatenate[HasAtomsT, P], polars.DataFrame]
) -> t.Callable[Concatenate[HasAtomsT, P], HasAtomsT]:

    @wraps(f)
    def wrapper(self: HasAtomsT, *args: P.args, **kwargs: P.kwargs) -> HasAtomsT:
        frame = f(self, *args, **kwargs)
        return self.with_atoms(Atoms(frame, _unchecked=True))

    return wrapper


def _fwd_frame(
    impl_f: t.Callable[Concatenate[polars.DataFrame, P], T]
) -> t.Callable[[t.Callable[Concatenate[HasAtomsT, P], t.Any]], t.Callable[Concatenate[HasAtomsT, P], T]]:
    def inner(f: t.Callable[Concatenate[HasAtomsT, P], t.Any]) -> t.Callable[Concatenate[HasAtomsT, P], T]:
        @wraps(f)
        def wrapper(self: HasAtoms, *args: P.args, **kwargs: P.kwargs) -> T:
            return impl_f(self._get_frame(), *args, **kwargs)

        return wrapper
    return inner


def _coord_expr(col: t.Union[str, int]) -> polars.Expr:
    if isinstance(col, str):
        col = "xyz".index(col)
    return polars.col('coords').arr.get(col).alias("xyz"[col])


class HasAtoms(abc.ABC):
    """Abstract class representing any (possibly compound) collection of atoms."""

    # abstract methods

    @abc.abstractmethod
    def get_atoms(self, frame: t.Literal['local'] = 'local') -> Atoms:
        """
        Get atoms contained in `self`. This should be a low cost method.

        Args:
          frame: Coordinate frame to return atoms in. For a plain [`HasAtoms`][atomlib.atoms.HasAtoms],
                 only `'local'` is supported.

        Return:
          The contained atoms
        """
        ...

    @abc.abstractmethod
    def with_atoms(self: HasAtomsT, atoms: HasAtoms, frame: t.Literal['local'] = 'local') -> HasAtomsT:
        """
        Return a copy of self with the inner [`Atoms`][atomlib.atoms.Atoms] replaced.

        Args:
          atoms: [`HasAtoms`][atomlib.atoms.HasAtoms] to replace these with.
          frame: Coordinate frame inside atoms are in. For a plain [`HasAtoms`][atomlib.atoms.HasAtoms],
                 only `'local'` is supported.

        Return:
          A copy of `self` updated with the given atoms
        """
        ...

    @classmethod
    @abc.abstractmethod
    def _combine_metadata(cls: t.Type[HasAtomsT], *atoms: HasAtoms) -> HasAtomsT:
        """
        When combining multiple `HasAtoms`, check that they are compatible with each other,
        and return a 'representative' which best represents the combined metadata.
        Implementors should treat `Atoms` as acceptable, but having no metadata.
        """
        ...

    def _get_frame(self) -> polars.DataFrame:
        return self.get_atoms().inner

    # dataframe methods

    @property
    @_fwd_frame(lambda df: df.columns)
    def columns(self) -> t.Sequence[str]:
        """
        Return the column names in `self`.

        Returns:
          A sequence of column names
        """
        ...

    @property
    @_fwd_frame(lambda df: df.dtypes)
    def dtypes(self) -> t.Sequence[polars.DataType]:
        """
        Return the datatypes in `self`.

        Returns:
          A sequence of column [`DataType`][polars.datatypes.DataType]s
        """
        ...

    @property
    @_fwd_frame(lambda df: df.schema)
    def schema(self) -> SchemaDict:
        """
        Return the schema of `self`.

        Returns:
          A dictionary of column names and [`DataType`][polars.datatypes.DataType]s
        """
        ...

    @_fwd_frame(polars.DataFrame.describe)
    def describe(self, percentiles: t.Union[t.Sequence[float], float, None] = (0.25, 0.5, 0.75), *,
                 interpolation: RollingInterpolationMethod = 'nearest') -> polars.DataFrame:
        """
        Return summary statistics for `self`. See [`DataFrame.describe`][polars.DataFrame.describe] for more information.

        Args:
          percentiles: List of percentiles/quantiles to include. Defaults to 25% (first quartile),
                       50% (median), and 75% (third quartile).

        Returns:
          A dataframe containing summary statistics (mean, std. deviation, percentiles, etc.) for each column.
        """
        ...

    @_fwd_frame_map
    def with_columns(self,
                     *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
                     **named_exprs: IntoExpr) -> polars.DataFrame:
        """Return a copy of `self` with the given columns added."""
        return self._get_frame().with_columns(*exprs, **named_exprs)

    with_column = with_columns

    @_fwd_frame_map
    def insert_column(self, index: int, column: polars.Series) -> polars.DataFrame:
        return self._get_frame().insert_column(index, column)

    @_fwd_frame(polars.DataFrame.get_column)
    def get_column(self, name: str) -> polars.Series:
        """
        Get the specified column from `self`, raising [`polars.ColumnNotFoundError`][polars.exceptions.ColumnNotFoundError] if it's not present.

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
        ...

    @_fwd_frame(polars.DataFrame.get_columns)
    def get_columns(self) -> t.List[polars.Series]:
        """
        Return all columns from `self` as a list of [`Series`][polars.Series].

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
        ...

    @_fwd_frame(polars.DataFrame.get_column_index)
    def get_column_index(self, name: str) -> int:
        """Get the index of a column by name, raising [`polars.ColumnNotFoundError`][polars.exceptions.ColumnNotFoundError] if it's not present."""
        ...

    @_fwd_frame(polars.DataFrame.group_by)
    def group_by(self, *by: t.Union[IntoExpr, t.Iterable[IntoExpr]], maintain_order: bool = False,
                 **named_by: IntoExpr) -> polars.dataframe.group_by.GroupBy:
        """
        Start a group by operation. See [`DataFrame.group_by`][polars.DataFrame.group_by] for more information.
        """
        ...

    def pipe(self: HasAtomsT, function: t.Callable[Concatenate[HasAtomsT, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Apply `function` to `self` (in method-call syntax)."""
        return function(self, *args, **kwargs)

    @_fwd_frame_map
    def clone(self) -> polars.DataFrame:
        """Return a copy of `self`."""
        return self._get_frame().clone()

    def drop(self, *columns: t.Union[str, t.Iterable[str]]) -> polars.DataFrame:
        """Return `self` with the specified columns removed."""
        return self._get_frame().drop(*columns)

    # row-wise operations

    def filter(
        self: HasAtomsT,
        *predicates: t.Union[None, IntoExprColumn, t.Iterable[IntoExprColumn], bool, t.List[bool], numpy.ndarray],
        **constraints: t.Any,
    ) -> HasAtomsT:
        """Filter `self`, removing rows which evaluate to `False`."""
        # TODO clean up
        preds_not_none: t.Tuple[t.Union[IntoExprColumn, t.Iterable[IntoExprColumn], bool, t.List[bool], numpy.ndarray], ...]
        preds_not_none = tuple(filter(lambda p: p is not None, predicates))  # type: ignore
        if not len(preds_not_none) and not len(constraints):
            return self
        return self.with_atoms(Atoms(self._get_frame().filter(*preds_not_none, **constraints), _unchecked=True))

    @_fwd_frame_map
    def sort(
        self,
        by: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        *more_by: IntoExpr,
        descending: t.Union[bool, t.Sequence[bool]] = False,
        nulls_last: bool = False,
    ) -> polars.DataFrame:
        """
        Sort the atoms in `self` by the given columns/expressions.
        """
        return self._get_frame().sort(
            by, *more_by, descending=descending, nulls_last=nulls_last
        )

    @_fwd_frame_map
    def slice(self, offset: int, length: t.Optional[int] = None) -> polars.DataFrame:
        """Return a slice of the rows in `self`."""
        return self._get_frame().slice(offset, length)

    @_fwd_frame_map
    def head(self, n: int = 5) -> polars.DataFrame:
        """Return the first `n` rows of `self`."""
        return self._get_frame().head(n)

    @_fwd_frame_map
    def tail(self, n: int = 5) -> polars.DataFrame:
        """Return the last `n` rows of `self`."""
        return self._get_frame().tail(n)

    @_fwd_frame_map
    def drop_nulls(self, subset: t.Union[str, t.Collection[str], None] = None) -> polars.DataFrame:
        """Drop rows that contain nulls in any of columns `subset`."""
        return self._get_frame().drop_nulls(subset)

    @_fwd_frame_map
    def fill_null(
        self, value: t.Any = None, strategy: t.Optional[FillNullStrategy] = None,
        limit: t.Optional[int] = None, matches_supertype: bool = True,
    ) -> polars.DataFrame:
        """Fill null values in `self`, using the specified value or strategy."""
        return self._get_frame().fill_null(value, strategy, limit, matches_supertype=matches_supertype)

    @_fwd_frame_map
    def fill_nan(self, value: t.Union[polars.Expr, int, float, None]) -> polars.DataFrame:
        """Fill floating-point NaN values in `self`."""
        return self._get_frame().fill_nan(value)

    @classmethod
    def concat(cls: t.Type[HasAtomsT],
               atoms: t.Union[HasAtomsT, IntoAtoms, t.Iterable[t.Union[HasAtomsT, IntoAtoms]]], *,
               rechunk: bool = True, how: ConcatMethod = 'vertical') -> HasAtomsT:
        """Concatenate multiple `Atoms` together, handling metadata appropriately."""
        # this method is tricky. It needs to accept raw Atoms, as well as HasAtoms of the
        # same type as ``cls``.
        if _is_abstract(cls):
            raise TypeError(f"concat() must be called on a concrete class.")

        if isinstance(atoms, HasAtoms):
            atoms = (atoms,)
        dfs = [a.get_atoms('local').inner if isinstance(a, HasAtoms) else Atoms(t.cast(IntoAtoms, a)).inner for a in atoms]
        representative = cls._combine_metadata(*(a for a in atoms if isinstance(a, HasAtoms)))

        if len(dfs) == 0:
            return representative.with_atoms(Atoms.empty(), 'local')

        if how in ('vertical', 'vertical_relaxed'):
            # get order from first member
            cols = dfs[0].columns
            dfs = [df.select(cols) for df in dfs]
        elif how == 'inner':
            cols = reduce(operator.and_, (df.schema.keys() for df in dfs))
            schema = OrderedDict((col, dfs[0].schema[col]) for col in cols)
            if len(schema) == 0:
                raise ValueError(f"Atoms have no columns in common")

            dfs = [_select_schema(df, schema) for df in dfs]
            how = 'vertical'

        return representative.with_atoms(Atoms(polars.concat(dfs, rechunk=rechunk, how=how)), 'local')

    @t.overload
    def partition_by(
        self: HasAtomsT, by: t.Union[str, t.Sequence[str]], *more_by: str,
        maintain_order: bool = True, include_key: bool = True, as_dict: t.Literal[False] = False
    ) -> t.List[HasAtomsT]:
        ...

    @t.overload
    def partition_by(
        self: HasAtomsT, by: t.Union[str, t.Sequence[str]], *more_by: str,
        maintain_order: bool = True, include_key: bool = True, as_dict: t.Literal[True] = ...
    ) -> t.Dict[t.Any, HasAtomsT]:
        ...

    def partition_by(
        self: HasAtomsT, by: t.Union[str, t.Sequence[str]], *more_by: str,
        maintain_order: bool = True, include_key: bool = True, as_dict: bool = False
    ) -> t.Union[t.List[HasAtomsT], t.Dict[t.Any, HasAtomsT]]:
        """
        Group by the given columns and partition into separate dataframes.

        Return the partitions as a dictionary by specifying `as_dict=True`.
        """
        if as_dict:
            d = self._get_frame().partition_by(by, *more_by, maintain_order=maintain_order, include_key=include_key, as_dict=True)
            return {k: self.with_atoms(Atoms(df, _unchecked=True)) for (k, df) in d.items()}

        return [
            self.with_atoms(Atoms(df, _unchecked=True))
            for df in self._get_frame().partition_by(by, *more_by, maintain_order=maintain_order, include_key=include_key, as_dict=False)
        ]

    # column-wise operations

    @_fwd_frame(polars.DataFrame.select)
    def select(
        self,
        *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        **named_exprs: IntoExpr,
    ):
        """
        Select `exprs` from `self`, and return as a [`polars.DataFrame`][polars.DataFrame].

        Expressions may either be columns or expressions of columns.

        [polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
        """
        ...

    # some helpers we add

    def select_schema(self, schema: SchemaDict) -> polars.DataFrame:
        """
        Select columns from `self` and cast to the given schema.
        Raises [`TypeError`][TypeError] if a column is not found or if it can't be cast.
        """
        return _select_schema(self, schema)

    def select_props(
        self: HasAtomsT,
        *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        **named_exprs: IntoExpr
    ) -> HasAtomsT:
        """
        Select `exprs` from `self`, while keeping required columns.

        Returns:
          A [`HasAtoms`][atomlib.atoms.HasAtoms] filtered to contain the
          specified properties (as well as required columns).
        """
        props = self._get_frame().lazy().select(*exprs, **named_exprs).drop(_REQUIRED_COLUMNS).collect(_eager=True)
        return self.with_atoms(
            Atoms(self._get_frame().select(_REQUIRED_COLUMNS).hstack(props), _unchecked=False)
        )

    def try_select(
        self,
        *exprs: t.Union[IntoExpr, t.Iterable[IntoExpr]],
        **named_exprs: IntoExpr,
    ) -> t.Optional[polars.DataFrame]:
        """
        Try to select `exprs` from `self`, and return as a [`polars.DataFrame`][polars.DataFrame].

        Expressions may either be columns or expressions of columns. Returns `None` if any
        columns are missing.

        [polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
        """
        try:
            return self._get_frame().select(*exprs, **named_exprs)
        except polars.ColumnNotFoundError:
            return None

    def try_get_column(self, name: str) -> t.Optional[polars.Series]:
        """Try to get a column from `self`, returning `None` if it doesn't exist."""
        try:
            return self.get_column(name)
        except polars.ColumnNotFoundError:
            return None

    def assert_equal(self, other: t.Any):
        assert isinstance(other, HasAtoms)
        assert dict(self.schema) == dict(other.schema)
        for col in self.schema.keys():
            polars.testing.assert_series_equal(self[col], other[col], check_names=False, rtol=1e-3, atol=1e-8)

    # dunders

    @_fwd_frame(polars.DataFrame.__len__)
    def __len__(self) -> int:
        """Return the number of atoms in `self`."""
        ...

    @_fwd_frame(polars.DataFrame.__contains__)
    def __contains__(self, key: str) -> bool:
        """Return whether `self` contains the given column."""
        ...

    def __add__(self: HasAtomsT, other: IntoAtoms) -> HasAtomsT:
        return self.__class__.concat((self, other), how='inner')

    def __radd__(self: HasAtomsT, other: IntoAtoms) -> HasAtomsT:
        return self.__class__.concat((other, self), how='inner')

    def __getitem__(self, column: str) -> polars.Series:
        try:
            return self.get_column(column)
        except polars.ColumnNotFoundError:
            if column in ('x', 'y', 'z'):
                return self.select(_coord_expr(column)).to_series()
            raise

    @_fwd_frame(polars.DataFrame.__dataframe__)
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> polars.interchange.dataframe.PolarsDataFrame:
        ...

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
        selection = _selection_to_numpy(self, selection)
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
        return self.with_columns(coords=polars.concat_list(
            polars.when(_coord_expr(col).abs() >= tol).then(_coord_expr(col)).otherwise(polars.lit(0.))
            for col in range(3)
        ).list.to_array(3))

    def crop(self: HasAtomsT, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf) -> HasAtomsT:
        """
        Crop, removing all atoms outside of the specified region, inclusive.
        """

        return self.filter(
            self.x().is_between(x_min, x_max, closed='both'),
            self.y().is_between(y_min, y_max, closed='both'),
            self.z().is_between(z_min, z_max, closed='both'),
        )

    crop_atoms = crop

    def _wrap(self: HasAtomsT, eps: float = 1e-5) -> HasAtomsT:
        coords = (self.coords() + eps) % 1. - eps
        return self.with_coords(coords)

    def deduplicate(self: HasAtomsT, tol: float = 1e-3, subset: t.Iterable[str] = ('x', 'y', 'z', 'symbol'),
                    keep: UniqueKeepStrategy = 'first', maintain_order: bool = True) -> HasAtomsT:
        """
        De-duplicate atoms in `self`. Atoms of the same `symbol` that are closer than `tolerance`
        to each other (by Euclidian distance) will be removed, leaving only the atom specified by
        `keep` (defaults to the first atom).

        If `subset` is specified, only those columns will be included while assessing duplicates.
        Floating point columns other than 'x', 'y', and 'z' will not by toleranced.
        """
        import scipy.spatial

        cols = set((subset,) if isinstance(subset, str) else subset)

        indices = numpy.arange(len(self))

        spatial_cols = cols.intersection(('x', 'y', 'z'))
        cols -= spatial_cols
        if len(spatial_cols) > 0:
            coords = self.select([_coord_expr(col).alias(col) for col in spatial_cols]).to_numpy()
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

        frame = self._get_frame().unique(subset=list(cols), keep=keep, maintain_order=maintain_order)
        if len(spatial_cols) > 0:
            frame = frame.drop('_unique_pts')

        return self.with_atoms(Atoms(frame, _unchecked=True))

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

    def coords(self, selection: t.Optional[AtomSelection] = None, *, frame: t.Literal['local'] = 'local') -> NDArray[numpy.float64]:
        """Return a `(N, 3)` ndarray of atom coordinates (dtype [`numpy.float64`][numpy.float64])."""
        df = self if selection is None else self.filter(_selection_to_expr(self, selection))
        return df.get_column('coords').to_numpy().astype(numpy.float64)

    def x(self) -> polars.Expr:
        return polars.col('coords').arr.get(0).alias('x')

    def y(self) -> polars.Expr:
        return polars.col('coords').arr.get(1).alias('y')

    def z(self) -> polars.Expr:
        return polars.col('coords').arr.get(2).alias('z')

    def velocities(self, selection: t.Optional[AtomSelection] = None) -> t.Optional[NDArray[numpy.float64]]:
        """Return a `(N, 3)` ndarray of atom velocities (dtype [`numpy.float64`][numpy.float64])."""
        if 'velocity' not in self:
            return None

        df = self if selection is None else self.filter(_selection_to_expr(self, selection))
        return df.get_column('velocity').to_numpy().astype(numpy.float64)

    def types(self) -> t.Optional[polars.Series]:
        """
        Returns a [`Series`][polars.Series] of atom types (dtype [`polars.Int32`][polars.datatypes.Int32]).

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
        return self.try_get_column('type')

    def masses(self) -> t.Optional[polars.Series]:
        """
        Returns a [`Series`][polars.Series] of atom masses (dtype [`polars.Float32`][polars.datatypes.Float32]).

        [polars.Series]: https://docs.pola.rs/py-polars/html/reference/series/index.html
        """
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
        Return a copy of `self` with an extra atom.

        By default, all extra columns present in `self` must be specified as `**kwargs`.

        Try to avoid calling this in a loop (Use [`HasAtoms.concat`][atomlib.atoms.HasAtoms.concat] instead).
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
            selection &= self.x().is_between(x - tol, x + tol, closed='none')
        if y is not None:
            selection &= self.y().is_between(y - tol, y + tol, closed='none')
        if z is not None:
            selection &= self.z().is_between(z - tol, z + tol, closed='none')
        for (col, val) in kwargs.items():
            selection &= (polars.col(col) == val)

        return selection

    def with_index(self: HasAtomsT, index: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Returns `self` with a row index added in column 'i' (dtype [`polars.Int64`][polars.datatypes.Int64]).
        If `index` is not specified, defaults to an existing index or a new index.
        """
        if index is None and 'i' in self.columns:
            return self
        if index is None:
            index = numpy.arange(len(self), dtype=numpy.int64)
        return self.with_column(_values_to_expr(self, index, polars.Int64).alias('i'))

    def with_wobble(self: HasAtomsT, wobble: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Return `self` with the given displacements in column 'wobble' (dtype [`polars.Float64`][polars.datatypes.Float64]).
        If `wobble` is not specified, defaults to the already-existing wobbles or 0.
        """
        if wobble is None and 'wobble' in self.columns:
            return self
        wobble = 0. if wobble is None else wobble
        return self.with_column(_values_to_expr(self, wobble, polars.Float64).alias('wobble'))

    def with_occupancy(self: HasAtomsT, frac_occupancy: t.Optional[AtomValues] = None) -> HasAtomsT:
        """
        Return self with the given fractional occupancies (dtype [`polars.Float64`][polars.datatypes.Float64]).
        If `frac_occupancy` is not specified, defaults to the already-existing occupancies or 1.
        """
        if frac_occupancy is None and 'frac_occupancy' in self.columns:
            return self
        frac_occupancy = 1. if frac_occupancy is None else frac_occupancy
        return self.with_column(_values_to_expr(self, frac_occupancy, polars.Float64).alias('frac_occupancy'))

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
        For instance: `["Ag+", "Na", "H", "Ag"]` => `[3, 11, 1, 2]`
        """
        if types is not None:
            return self.with_columns(type=_values_to_expr(self, types, polars.Int32))
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
        Return `self` with the given atom masses in column `'mass'`.
        If `mass` is not specified, use the already existing masses or auto-assign them.
        """
        if mass is not None:
            return self.with_column(_values_to_expr(self, mass, polars.Float32).alias('mass'))
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

    def with_coords(self: HasAtomsT, pts: ArrayLike, selection: t.Optional[AtomSelection] = None, *, frame: t.Literal['local'] = 'local') -> HasAtomsT:
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
        return self.with_columns(polars.Series('coords', pts, polars.Array(polars.Float64, 3)))

    def with_velocity(self: HasAtomsT, pts: t.Optional[ArrayLike] = None,
                      selection: t.Optional[AtomSelection] = None) -> HasAtomsT:
        """
        Return `self` replaced with the given atomic velocities.
        If `pts` is not specified, use the already existing velocities or zero.
        """
        if pts is None:
            if 'velocity' in self:
                return self
            all_pts = numpy.zeros((len(self), 3))
        else:
            all_pts = self['velocity'].to_numpy()

        if selection is None:
            all_pts = pts or all_pts
        elif pts is not None:
            selection = _selection_to_numpy(self, selection)
            all_pts = numpy.require(all_pts, requirements=['WRITEABLE'])
            pts = numpy.atleast_2d(pts)
            assert pts.shape[-1] == 3
            all_pts[selection] = pts

        all_pts = numpy.broadcast_to(all_pts, (len(self), 3))
        return self.with_columns(polars.Series('velocity', all_pts, polars.Array(polars.Float64, 3)))


class Atoms(AtomsIOMixin, HasAtoms):
    r"""
    A collection of atoms, absent any implied coordinate system.
    Implemented as a wrapper around a [`polars.DataFrame`][polars.DataFrame].

    Must contain the following columns:

    - coords: array of `[x, y, z]` positions, float
    - elem: atomic number, int
    - symbol: atomic symbol (may contain charges)

    In addition, it commonly contains the following columns:

    - i: Initial atom number
    - wobble: Isotropic Debye-Waller mean-squared deviation ($\left<u^2\right> = B \cdot \frac{3}{8 \pi^2}$, dimensions of [Length^2])
    - frac_occupancy: Fractional occupancy, in the range [0., 1.]
    - mass: Atomic mass, in g/mol (approx. Da)
    - velocity: array of `[x, y, z]` velocities, float, dimensions of length/time
    - type: Numeric atom type, as used by programs like LAMMPS

    [polars.DataFrame]: https://docs.pola.rs/py-polars/html/reference/dataframe/index.html
    """

    def __init__(self, data: t.Optional[IntoAtoms] = None, columns: t.Optional[t.Sequence[str]] = None,
                 orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None,
                 _unchecked: bool = False):
        self._bbox: t.Optional[BBox3D] = None
        self.inner: polars.DataFrame

        if data is None:
            assert columns is None
            self.inner = polars.DataFrame([
                polars.Series('coords', (), dtype=polars.Array(polars.Float64, 3)),
                polars.Series('elem', (), dtype=polars.Int8),
                polars.Series('symbol', (), dtype=polars.Utf8),
            ])
        elif isinstance(data, polars.DataFrame):
            self.inner = data
        elif isinstance(data, Atoms):
            self.inner = data.inner
            _unchecked = True
        else:
            self.inner = polars.DataFrame(data, schema=columns, orient=orient)

        if not _unchecked:
            # stack ('x', 'y', 'z') -> 'coords'
            self.inner = _with_columns_stacked(self.inner, ('x', 'y', 'z'), 'coords')
            self.inner = _with_columns_stacked(self.inner, ('v_x', 'v_y', 'v_z'), 'velocity')

            missing: t.Tuple[str, ...] = tuple(set(['symbol', 'elem']) - set(self.columns))
            if len(missing) > 1:
                raise ValueError("'Atoms' missing columns 'elem' and/or 'symbol'.")
            # fill 'symbol' from 'elem' or vice-versa
            if missing == ('symbol',):
                self.inner = self.inner.with_columns(get_sym(self.inner['elem']))
            elif missing == ('elem',):
                # by convention, add before 'symbol' column
                self.inner = self.inner.insert_column(
                    self.inner.get_column_index('symbol'),
                    get_elem(self.inner['symbol']),
                )

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
        missing = [col for col in _REQUIRED_COLUMNS if col not in self.columns]
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
        buf = StringIO()
        buf.write("Atoms([\n")

        for series in self.inner.to_dict().values():
            buf.write(f"    Series({series.name!r}, {series.to_list()!r}, dtype={series.dtype!r}),\n")

        buf.write("])\n")
        return buf.getvalue()

    def _repr_pretty_(self, p, cycle: bool) -> None:
        p.text('Atoms(...)') if cycle else p.text(str(self))


SchemaDict: TypeAlias = OrderedDict[str, polars.DataType]
IntoExprColumn: TypeAlias = polars.type_aliases.IntoExprColumn
IntoExpr: TypeAlias = polars.type_aliases.IntoExpr
UniqueKeepStrategy: TypeAlias = polars.type_aliases.UniqueKeepStrategy
FillNullStrategy: TypeAlias = polars.type_aliases.FillNullStrategy
RollingInterpolationMethod: TypeAlias = polars.type_aliases.RollingInterpolationMethod
ConcatMethod: TypeAlias = t.Literal['horizontal', 'vertical', 'diagonal', 'inner', 'align']

IntoAtoms = t.Union[t.Dict[str, t.Sequence[t.Any]], t.Sequence[t.Any], numpy.ndarray, polars.DataFrame, 'Atoms']
"""
A type convertible into an [`Atoms`][atomlib.atoms.Atoms].
"""

AtomSelection = t.Union[IntoExprColumn, NDArray[numpy.bool_], ArrayLike, t.Mapping[str, t.Any]]
"""
Polars expression selecting a subset of atoms.
Can be used with many [`Atoms`][atomlib.atoms.Atoms] methods.
"""

AtomValues = t.Union[IntoExprColumn, NDArray[numpy.generic], ArrayLike, t.Mapping[str, t.Any]]
"""
Array, value, or polars expression mapping atom symbols to values.
Can be used with `with_*` methods on Atoms
"""


__all__ = [
    'Atoms', 'HasAtoms', 'IntoAtoms', 'AtomSelection', 'AtomValues',
]