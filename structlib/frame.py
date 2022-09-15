from __future__ import annotations

import logging
import typing as t

import numpy
from numpy.typing import ArrayLike
import polars

from .vec import BBox
from .elem import get_elem, get_sym, get_mass
from .transform import Transform, IntoTransform
from .util import map_some


IntoAtoms: t.TypeAlias = t.Union[t.Dict[str, t.Sequence[t.Any]], t.Sequence[t.Any], numpy.ndarray, polars.DataFrame]
"""
A type convertible into an `AtomFrame`.
"""


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


def _selection_to_series(df: polars.DataFrame, selection: AtomSelection) -> polars.Series:
    if isinstance(selection, polars.Series):
        return selection.cast(polars.Boolean)
    if isinstance(selection, polars.Expr):
        return df.select(selection.cast(polars.Boolean)).to_series()

    selection = numpy.broadcast_to(selection, len(df))
    return polars.Series(selection, dtype=polars.Boolean)


def _selection_to_expr(selection: AtomSelection) -> polars.Expr:
    if isinstance(selection, polars.Expr):
        return selection.cast(polars.Boolean)
    if isinstance(selection, polars.Series):
        return polars.lit(selection, dtype=polars.Boolean)

    selection = numpy.asanyarray(selection, dtype=numpy.bool_)
    return polars.lit(selection)


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
    - mass: Atomic mass, in g/mol (approx. Da)
    - v_[xyz]: Atom velocities, dimensions of length/time
    - atom_type: Numeric atom type, as used by programs like LAMMPS
    """

    def __new__(cls, data: t.Optional[IntoAtoms] = None, columns: t.Optional[t.Sequence[str]] = None,
                orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None) -> AtomFrame:
        if data is None:
            return super().__new__(cls)
        if isinstance(data, polars.DataFrame):
            obj = data.clone()
        else:
            obj = polars.DataFrame(data, columns, orient=orient)

        missing: t.Tuple[str] = tuple(set(('symbol', 'elem')) - set(obj.columns))  # type: ignore
        if len(missing) > 1:
            raise ValueError("'AtomFrame' missing columns 'elem' and/or 'symbol'.")
        if missing == ('symbol',):
            obj = obj.with_columns(get_sym(obj['elem']))
        elif missing == ('elem',):
            obj = obj.with_columns(get_elem(obj['symbol']))

        # cast to standard dtypes
        obj = obj.with_columns([
            obj.get_column(col).cast(dtype)
            for (col, dtype) in _COLUMN_DTYPES.items() if col in obj
        ])

        obj.__class__ = cls
        return t.cast(AtomFrame, obj)

    def __init__(self, data: IntoAtoms, columns: t.Optional[t.Sequence[str]] = None,
                 orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None):
        self._validate_atoms()
        self._bbox: t.Optional[BBox] = None

    def _validate_atoms(self):
        missing = [col for col in ['x', 'y', 'z', 'elem', 'symbol'] if col not in self]
        if len(missing):
            raise ValueError(f"'AtomFrame' missing column(s) {', '.join(map(repr, missing))}")

    @staticmethod
    def empty() -> AtomFrame:
        data = [
            polars.Series('x', (), dtype=polars.Float64),
            polars.Series('y', (), dtype=polars.Float64),
            polars.Series('z', (), dtype=polars.Float64),
            polars.Series('elem', (), dtype=polars.Int8),
            polars.Series('symbol', (), dtype=polars.Utf8),
        ]
        return AtomFrame(data)

    def with_column(self, column: t.Union[polars.Series, polars.Expr]) -> AtomFrame:
        new = super().with_column(column)
        new.__class__ = type(self)
        return t.cast(AtomFrame, new)

    def with_columns(self, exprs: t.Union[t.Literal[None], polars.Series, polars.Expr, t.Sequence[t.Union[polars.Series, polars.Expr]]],
                     **named_exprs: t.Union[polars.Expr, polars.Series]) -> AtomFrame:
        new = super().with_columns(exprs, **named_exprs)
        new.__class__ = type(self)
        return t.cast(AtomFrame, new)

    def try_get_column(self, name: str) -> t.Optional[polars.Series]:
        try:
            return self.get_column(name)
        except polars.NotFoundError:
            return None

    def coords(self, selection: t.Optional[AtomSelection] = None) -> numpy.ndarray:
        """Returns a (N, 3) ndarray of atom coordinates."""
        # TODO find a way to get a view
        if selection is not None:
            self = self.filter(_selection_to_expr(selection))
        return self.select(('x', 'y', 'z')).to_numpy()

    def velocities(self, selection: t.Optional[AtomSelection] = None) -> t.Optional[numpy.ndarray]:
        """Returns a (N, 3) ndarray of atom velocities."""
        if selection is not None:
            self = self.filter(_selection_to_expr(selection))
        try:
            return self.select(('v_x', 'v_y', 'v_z')).to_numpy()
        except polars.NotFoundError:
            return None

    def types(self) -> t.Optional[polars.Series]:
        return self.try_get_column('type')

    def masses(self) -> t.Optional[polars.Series]:
        return self.try_get_column('mass')

    def with_wobble(self, wobble: t.Optional[polars.Series] = None) -> AtomFrame:
        """
        Return self with the given displacements. If `wobble` is not specified,
        defaults to the already-existing wobbles or 0.
        """
        if wobble is not None:
            return self.with_column(polars.Series('wobble', wobble, dtype=polars.Float64))
        if 'wobble' in self:
            return self

        return self.with_column(polars.lit(0., dtype=polars.Float64).alias('wobble'))

    def with_occupancy(self, frac_occupancy: t.Optional[polars.Series] = None) -> AtomFrame:
        """
        Return self with the given fractional occupancies. If `frac_occupancy` is not specified,
        defaults to the already-existing occupancies or 1.
        """
        if frac_occupancy is not None:
            return self.with_column(polars.Series('frac_occupancy', frac_occupancy, dtype=polars.Float64))
        if 'frac_occupancy' in self:
            return self

        return self.with_column(polars.lit(1., dtype=polars.Float64).alias('frac_occupancy'))

    def with_type(self, types: t.Optional[polars.Series] = None) -> AtomFrame:
        """
        Return `self` with the given atom types in column 'types'.
        If `types` is not specified, use the already existing types or auto-assign them.

        When auto-assigning, each symbol is given a unique value, case-sensitive.
        Values are assigned from lowest atomic number to highest.
        For instance: ["Ag+", "Na", "H", "Ag"] => [3, 11, 1, 2]
        """
        if types is not None:
            return self.with_column(polars.Series('type', types, dtype=polars.Int32))
        if 'type' in self:
            return self

        unique = self.unique(maintain_order=False, subset=['elem', 'symbol']).sort(['elem', 'symbol'])
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

    def with_mass(self, mass: t.Optional[ArrayLike] = None) -> AtomFrame:
        """
        Return `self` with the given atom masses in column 'mass'.
        If `mass` is not specified, use the already existing masses or auto-assign them.
        """
        if mass is not None:
            return self.with_column(polars.Series('mass', mass, dtype=polars.Float32))
        if 'mass' in self:
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

    def with_coords(self, pts: ArrayLike, selection: t.Optional[AtomSelection] = None) -> AtomFrame:
        """
        Return `self` replaced with the given atomic positions.
        """
        pts = numpy.broadcast_to(pts, (len(self), 3))
        if selection is None:
            return self.__class__(self.with_columns((
                polars.Series(pts[:, 0], dtype=polars.Float64).alias('x'),
                polars.Series(pts[:, 1], dtype=polars.Float64).alias('y'),
                polars.Series(pts[:, 2], dtype=polars.Float64).alias('z'),
            )))

        selection = _selection_to_series(self, selection)
        return self.__class__(self.with_columns((
            self['x'].set_at_idx(selection, pts[:, 0]),
            self['y'].set_at_idx(selection, pts[:, 1]),
            self['z'].set_at_idx(selection, pts[:, 2]),
        )))

    def with_velocity(self, pts: t.Optional[ArrayLike] = None, selection: t.Optional[AtomSelection] = None) -> AtomFrame:
        """
        Return `self` replaced with the given atomic velocities.
        If `pts` is not specified, use the already existing velocities or zero.
        """
        if pts is None:
            if all(col in self for col in ('v_x', 'v_y', 'v_z')):
                return self
            pts = numpy.zeros((len(self), 3))
        else:
            pts = numpy.broadcast_to(pts, (len(self), 3))

        if selection is None:
            return self.__class__(self.with_columns((
                polars.Series(pts[:, 0], dtype=polars.Float64).alias('v_x'),
                polars.Series(pts[:, 1], dtype=polars.Float64).alias('v_y'),
                polars.Series(pts[:, 2], dtype=polars.Float64).alias('v_z'),
            )))

        selection = _selection_to_series(self, selection)
        return self.__class__(self.with_columns((
            self['v_x'].set_at_idx(selection, pts[:, 0]),
            self['v_y'].set_at_idx(selection, pts[:, 1]),
            self['v_z'].set_at_idx(selection, pts[:, 2]),
        )))

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self._bbox = BBox.from_pts(self.coords())

        return self._bbox

    def transform(self, transform: IntoTransform, selection: t.Optional[AtomSelection] = None, *, transform_velocities: bool = False) -> AtomFrame:
        transform = Transform.make(transform)
        selection = map_some(lambda s: _selection_to_series(self, s), selection)
        transformed = self.with_coords(Transform.make(transform) @ self.coords(selection), selection)
        # try to transform velocities as well
        if transform_velocities and (velocities := self.velocities(selection)) is not None:
            return transformed.with_velocity(transform.transform_vec(velocities), selection)
        return transformed


AtomSelection = t.Union[polars.Series, polars.Expr, ArrayLike]
"""
Polars expression selecting a subset of atoms from an AtomFrame. Can be used with DataFrame.filter()
"""
