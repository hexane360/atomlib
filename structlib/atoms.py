from __future__ import annotations

import logging
import typing as t

import numpy
import scipy.spatial
from numpy.typing import ArrayLike, NDArray
import polars
import polars.datatypes

from .types import to_vec3
from .bbox import BBox3D
from .elem import get_elem, get_sym, get_mass
from .transform import Transform3D, IntoTransform3D
from .util import map_some, opt_classmethod


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
UniqueKeepStrategy = t.Literal['first', 'last', 'none']


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
    arr = numpy.asarray(values)
    return polars.lit(polars.Series(arr, dtype=ty) if arr.size > 1 else values)


def _selection_to_series(df: t.Union[polars.DataFrame, Atoms], selection: AtomSelection) -> polars.Series:
    if isinstance(df, Atoms):
        df = df.inner
    return _values_to_series(df, selection, ty=polars.Boolean)


def _selection_to_expr(selection: AtomSelection) -> polars.Expr:
    return _values_to_expr(selection, ty=polars.Boolean)


class Atoms:
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
    - frac: Fractional occupancy, [0., 1.]
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
            self.inner = polars.DataFrame(
                data, columns=columns, orient=orient
            )

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

    def _validate_atoms(self):
        missing = [col for col in ['x', 'y', 'z', 'elem', 'symbol'] if col not in self.columns]
        if len(missing):
            raise ValueError(f"'Atoms' missing column(s) {', '.join(map(repr, missing))}")

    @staticmethod
    def empty() -> Atoms:
        """
        Return an empty Atoms with only the mandatory columns.
        """
        return Atoms()

    # methods which forward to DataFrame

    @property
    def columns(self) -> t.Sequence[str]:
        """Return the columns in `self`."""
        return self.inner.columns

    @property
    def schema(self) -> SchemaDict:
        """Return the schema of `self`."""
        return t.cast(SchemaDict, self.inner.schema)

    def __len__(self) -> int:
        """Return the number of atoms in `self`."""
        return self.inner.__len__()

    def __contains__(self, col: str) -> bool:
        """Return whether `self` contains the given column."""
        return col in self.inner.columns

    def with_column(self, column: t.Union[polars.Series, polars.Expr]) -> Atoms:
        """Return a copy of `self` with the given column added."""
        return Atoms(self.inner.with_column(column), _unchecked=True)

    def with_columns(self, exprs: t.Union[t.Literal[None], polars.Series, polars.Expr, t.Sequence[t.Union[polars.Series, polars.Expr]]],
                     **named_exprs: t.Union[polars.Expr, polars.Series]) -> Atoms:
        """Return a copy of `self` with the given columns added."""
        return Atoms(self.inner.with_columns(exprs, **named_exprs), _unchecked=True)

    def get_column(self, name: str) -> polars.Series:
        """Get the specified column from `self`, raising `polars.NotFoundError` if it's not present."""
        return self.inner.get_column(name)

    __getitem__ = get_column

    def filter(self, selection: t.Optional[AtomSelection] = None) -> Atoms:
        """Filter `self`, removing rows which evaluate to `False`."""
        if selection is None:
            return self
        return Atoms(self.inner.filter(_selection_to_expr(selection)), _unchecked=True)

    def select(self, exprs: t.Union[str, polars.Expr, polars.Series, t.Sequence[t.Union[str, polars.Expr, polars.Series]]]
    ) -> polars.DataFrame:
        """
        Select `exprs` from `self`, and return as a DataFrame.
        Expressions may either be columns or expressions of columns.
        """
        return self.inner.select(exprs)

    def sort(self, by: t.Union[str, polars.Expr, t.List[str], t.List[polars.Expr]], reverse: t.Union[bool, t.List[bool]] = False) -> Atoms:
        return Atoms(self.inner.sort(by, reverse), _unchecked=True)

    @opt_classmethod
    def concat(self, atoms: t.Union[Atoms, t.Iterable[Atoms]], rechunk: bool = True) -> Atoms:
        dfs = []
        if len(self) > 0:
            dfs.append(self.inner)
        dfs.append(atoms.inner) if isinstance(atoms, Atoms) else dfs.extend(a.inner for a in atoms)
        return Atoms(polars.concat(dfs, rechunk=rechunk))

    # methods unique to Atoms

    def try_get_column(self, name: str) -> t.Optional[polars.Series]:
        """Try to get a column from `self`, returning `None` if it doesn't exist."""
        try:
            return self.get_column(name)
        except polars.NotFoundError:
            return None

    def bbox(self) -> BBox3D:
        """Return the bounding box of all the points in `self`."""
        if self._bbox is None:
            self._bbox = BBox3D.from_pts(self.coords())

        return self._bbox

    def transform(self, transform: IntoTransform3D, selection: t.Optional[AtomSelection] = None, *, transform_velocities: bool = False) -> Atoms:
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

    def round_near_zero(self, tolerance: float = 1e-14) -> Atoms:
        """
        Round atom position values near zero to zero.
        """
        return self.with_columns(tuple(
            polars.when(col.abs() >= tolerance).then(col).otherwise(polars.lit(0.))
            for col in map(polars.col, ('x', 'y', 'z'))
        ))

    def crop(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf) -> Atoms:
        """
        Crop, removing all atoms outside of the specified region, inclusive.
        """

        min = to_vec3([x_min, y_min, z_min])
        max = to_vec3([x_max, y_max, z_max])

        return Atoms(self.inner.filter(
            (polars.col('x') >= min[0]) & (polars.col('x') <= max[0]) &
            (polars.col('y') >= min[1]) & (polars.col('y') <= max[1]) &
            (polars.col('z') >= min[2]) & (polars.col('z') <= max[2])
        ))

    def deduplicate(self, tolerance: float = 1e-3, subset: t.Iterable[str] = ('x', 'y', 'z', 'symbol'),
                    keep: UniqueKeepStrategy = 'first') -> Atoms:
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
                for (i, j) in tree.query_pairs(tolerance, 2.):
                    # whenever we encounter a pair, ensure their index matches
                    i_i, i_j = indices[[i, j]]
                    if i_i != i_j:
                        indices[i] = indices[j] = min(i_i, i_j)
                        changed = True
                if not changed:
                    break

        self = self.with_column(polars.Series('_unique_pts', indices))
        cols.add('_unique_pts')
        return Atoms(self.inner.unique(subset=list(cols), keep=keep).drop('_unique_pts'), _unchecked=True)

    unique = deduplicate

    def assert_equal(self, other: t.Any):
        assert isinstance(other, Atoms)
        assert self.schema == other.schema
        for (col, dtype) in self.schema.items():
            if dtype in (polars.Float32, polars.Float64):
                numpy.testing.assert_array_almost_equal(self[col].view(True), other[col].view(True), 5)
            else:
                assert (self[col] == other[col]).all()

    # property getters and setters

    def coords(self, selection: t.Optional[AtomSelection] = None) -> NDArray[numpy.float64]:
        """Returns a (N, 3) ndarray of atom coordinates (dtype `numpy.float64`)."""
        # TODO find a way to get a view
        return self.filter(selection).select(('x', 'y', 'z')).to_numpy().astype(numpy.float64)

    def velocities(self, selection: t.Optional[AtomSelection] = None) -> t.Optional[NDArray[numpy.float64]]:
        """Returns a (N, 3) ndarray of atom velocities (dtype `numpy.float64`)."""
        if selection is not None:
            self = self.filter(selection)
        try:
            return self.select(('v_x', 'v_y', 'v_z')).to_numpy().astype(numpy.float64)
        except polars.NotFoundError:
            return None

    def types(self) -> t.Optional[polars.Series]:
        """Returns a `Series` of atom types (dtype polars.Int32)."""
        return self.try_get_column('type')

    def masses(self) -> t.Optional[polars.Series]:
        """Returns a `Series` of atom masses (dtype polars.Float32)."""
        return self.try_get_column('mass')

    def with_index(self, index: t.Optional[AtomValues] = None) -> Atoms:
        """
        Returns `self` with a row index added in column 'i' (dtype polars.Int64).
        If `index` is not specified, defaults to an existing index or a new index.
        """
        if index is None and 'i' in self.columns:
            return self
        if index is None:
            index = numpy.arange(len(self), dtype=numpy.int64)
        return self.with_column(_values_to_expr(index, polars.Int64).alias('i'))

    def with_wobble(self, wobble: t.Optional[AtomValues] = None) -> Atoms:
        """
        Return `self` with the given displacements in column 'wobble' (dtype polars.Float64).
        If `wobble` is not specified, defaults to the already-existing wobbles or 0.
        """
        if wobble is None and 'wobble' in self.columns:
            return self
        wobble = 0. if wobble is None else wobble
        return self.with_column(_values_to_expr(wobble, polars.Float64).alias('wobble'))

    def with_occupancy(self, frac_occupancy: t.Optional[AtomValues] = None) -> Atoms:
        """
        Return self with the given fractional occupancies. If `frac_occupancy` is not specified,
        defaults to the already-existing occupancies or 1.
        """
        if frac_occupancy is None and 'frac_occupancy' in self.columns:
            return self
        frac_occupancy = 1. if frac_occupancy is None else frac_occupancy
        return self.with_column(_values_to_expr(frac_occupancy, polars.Float64).alias('frac_occupancy'))

    def apply_wobble(self, rng: t.Union[numpy.random.Generator, int, None] = None) -> Atoms:
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

    def apply_occupancy(self, rng: t.Union[numpy.random.Generator, int, None] = None) -> Atoms:
        """
        For each atom in `self`, use its `frac_occupancy` to randomly decide whether to remove it.
        """
        if 'frac_occupancy' not in self.columns:
            return self
        rng = numpy.random.default_rng(seed=rng)

        frac = self.select('frac_occupancy').to_series().to_numpy()
        choice = rng.binomial(1, frac).astype(numpy.bool_)
        return self.filter(polars.lit(choice))

    def with_type(self, types: t.Optional[AtomValues] = None) -> Atoms:
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

        unique = Atoms(self.inner.unique(maintain_order=False, subset=['elem', 'symbol']).sort(['elem', 'symbol']), _unchecked=True)
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

    def with_mass(self, mass: t.Optional[ArrayLike] = None) -> Atoms:
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

    def with_coords(self, pts: ArrayLike, selection: t.Optional[AtomSelection] = None) -> Atoms:
        """
        Return `self` replaced with the given atomic positions.
        """
        if selection is not None:
            selection = _selection_to_series(self, selection).to_numpy()
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

    def with_velocity(self, pts: t.Optional[ArrayLike] = None, selection: t.Optional[AtomSelection] = None) -> Atoms:
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

    # dunder methods

    def __add__(self, other: IntoAtoms) -> Atoms:
        return Atoms.concat((self, Atoms(other)))

    def __radd__(self, other: IntoAtoms) -> Atoms:
        return Atoms.concat((Atoms(other), self))

    def __eq__(self, other: t.Any) -> bool:
        return self.inner == other.inner

    def __str__(self) -> str:
        return f"Atoms, {self.inner!s}"

    def __repr__(self) -> str:
        # real __repr__ that polars doesn't provide
        lines = ["Atoms(["]
        for (col, series) in self.inner.to_dict().items():
            lines.append(f"    Series({col!r}, {list(series)!r}, {series.dtype.string_repr()!r}),")
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

AtomValues = t.Union[polars.Series, polars.Expr, ArrayLike]
"""
Array, value, or polars expression mapping atoms to values.
Can be used with `with_*` methods on Atoms
"""

__all__ = [
    'Atoms', 'IntoAtoms', 'AtomSelection', 'AtomValues',
]