from __future__ import annotations

import typing as t

import numpy
import polars

from .vec import BBox
from .elem import get_elem, get_sym
from .transform import Transform, IntoTransform


IntoAtoms: t.TypeAlias = t.Union[t.Dict[str, t.Sequence[t.Any]], t.Sequence[t.Any], numpy.ndarray, polars.DataFrame]
"""
A type convertible into an `AtomFrame`.
"""


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

    def __new__(cls, data: t.Optional[IntoAtoms] = None, columns: t.Optional[t.Sequence[str]] = None,
                orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None) -> AtomFrame:
        if data is None:
            return super().__new__(cls)
        if isinstance(data, polars.DataFrame):
            obj = data.clone()
        else:
            obj = polars.DataFrame(data, columns, orient=orient)

        missing: t.Tuple[str] = tuple(set(('symbol', 'elem')) - set(obj.columns))  # type: ignore
        if missing == ('symbol',):
            obj = obj.with_columns(get_sym(obj['elem']))
        elif missing == ('elem',):
            obj = obj.with_columns(get_elem(obj['symbol']))

        obj.__class__ = cls
        return t.cast(AtomFrame, obj)

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

    def __init__(self, data: IntoAtoms, columns: t.Optional[t.Sequence[str]] = None,
                 orient: t.Union[t.Literal['row'], t.Literal['col'], None] = None):
        self._validate_atoms()
        self._bbox: t.Optional[BBox] = None

    def _validate_atoms(self):
        missing: t.Set[str] = set(('x', 'y', 'z', 'elem', 'symbol')) - set(self.columns)  # type: ignore
        if len(missing):
            raise ValueError(f"'Atoms' missing column(s) {', '.join(map(repr, missing))}")

    def coords(self) -> numpy.ndarray:
        """Returns a (N, 3) ndarray of atom coordinates."""
        # TODO find a way to get a view
        return self.select(('x', 'y', 'z')).to_numpy()

    def velocities(self) -> t.Optional[numpy.ndarray]:
        """Returns a (N, 3) ndarray of atom velocities."""
        try:
            return self.select(('v_x', 'v_y', 'v_z')).to_numpy()
        except polars.NotFoundError:
            return None

    def with_coords(self, pts) -> AtomFrame:
        assert pts.shape == (len(self), 3)
        return self.__class__(self.with_columns((
            polars.Series(pts[:, 0], dtype=polars.Float64).alias('x'),
            polars.Series(pts[:, 1], dtype=polars.Float64).alias('y'),
            polars.Series(pts[:, 2], dtype=polars.Float64).alias('z'),
        )))

    def with_velocities(self, pts) -> AtomFrame:
        assert pts.shape == (len(self), 3)
        return self.__class__(self.with_columns((
            polars.Series(pts[:, 0], dtype=polars.Float64).alias('v_x'),
            polars.Series(pts[:, 1], dtype=polars.Float64).alias('v_y'),
            polars.Series(pts[:, 2], dtype=polars.Float64).alias('v_z'),
        )))

    @property
    def bbox(self) -> BBox:
        if self._bbox is None:
            self._bbox = BBox.from_pts(self.coords())

        return self._bbox

    def transform(self, transform: IntoTransform, transform_velocities: bool = False) -> AtomFrame:
        transform = Transform.make(transform)
        transformed = self.with_coords(Transform.make(transform) @ self.coords())
        # try to transform velocities as well
        if transform_velocities and (velocities := self.velocities()) is not None:
            return transformed.with_velocities(transform.transform_vec(velocities))
        return transformed



AtomSelection = t.NewType('AtomSelection', polars.Expr)
"""
Polars expression selecting a subset of atoms from an AtomFrame. Can be used with DataFrame.filter()
"""