"""Bounding boxes"""

from __future__ import annotations

import typing as t

import numpy


# pyright: reportImportCycles=false
if t.TYPE_CHECKING:
    from .types import Vec3, VecLike
    from .transform import AffineTransform3D

from .vec import to_vec3


class BBox3D:
    """
    3D axis-aligned bounding box, with corners `min` and `max`.
    """

    def __init__(self, min: VecLike, max: VecLike):
        # shape: (3, 2)
        # self._inner[1, 0]: min of y
        min, max = to_vec3(min), to_vec3(max)
        self.inner: numpy.ndarray = numpy.stack((min, max), axis=-1)

    @classmethod
    def from_pts(cls, pts: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> BBox3D:
        """Construct the minimum bounding box containing the points `pts`."""
        pts = numpy.atleast_2d(pts).reshape(-1, 3)
        return cls(numpy.nanmin(pts, axis=0), numpy.nanmax(pts, axis=0))

    @classmethod
    def unit(cls) -> BBox3D:
        """Return a unit bbox (cube from [0,0,0] to [1,1,1])."""
        return cls([0., 0., 0.], [1., 1., 1.])

    def transform_from_unit(self) -> AffineTransform3D:
        """Return the transform which transforms a unit bbox to `self`."""
        from .transform import AffineTransform3D
        return AffineTransform3D.translate(self.min).scale(self.max - self.min)

    def transform_to_unit(self) -> AffineTransform3D:
        """Return the transform which transforms `self` to a unit bbox."""
        return self.transform_from_unit().inverse()

    @property
    def min(self) -> Vec3:
        """Return the minimum corner `[xmin, ymin, zmin]`."""
        return self.inner[:, 0]

    @property
    def max(self) -> Vec3:
        """Return the minimum corner `[xmax, ymax, zmax]`."""
        return self.inner[:, 1]

    @property
    def x(self) -> numpy.ndarray:
        """Return the interval `[xmin, xmax]`."""
        return self.inner[0]

    @property
    def y(self) -> numpy.ndarray:
        """Return the interval `[ymin, ymax]`."""
        return self.inner[1]

    @property
    def z(self) -> numpy.ndarray:
        """Return the interval `[zmin, zmax]`."""
        return self.inner[2]

    @property
    def size(self) -> Vec3:
        """Return the size `[xsize, ysize, zsize]`."""
        return self.max - self.min

    def volume(self) -> float:
        """Return the volume of the bbox."""
        return float(numpy.prod(self.size))

    def corners(self) -> numpy.ndarray:
        """Return a (8, 3) ndarray containing the corners of the bbox."""
        return numpy.stack(list(map(numpy.ravel, numpy.meshgrid(*self.inner))), axis=-1)

    def pad(self, amount: t.Union[float, VecLike]) -> BBox3D:
        """
        Pad the given bbox by `amount`. If a vector `[x, y, z]` is given, pad each axis by the given amount.
        """
        amount_v = numpy.broadcast_to(amount, 3)

        return type(self)(
            self.min - amount_v,
            self.max + amount_v
        )

    def __or__(self, other: t.Union[Vec3, BBox3D]) -> BBox3D:
        """
        Union this bbox with another point or bbox.
        """
        if isinstance(other, numpy.ndarray):
            return self.from_pts((self.min, self.max, other))

        return type(self)(
            numpy.nanmin(((self.min, other.min)), axis=0),
            numpy.nanmax(((self.max, other.max)), axis=0),
        )

    __ror__ = __or__

    def __and__(self, other: BBox3D) -> BBox3D:
        """
        Intersect this bbox with another point or bbox.

        Undefined if there is no overlap between the two.
        """
        return type(self)(
            numpy.nanmax(((self.min, other.min)), axis=0),
            numpy.nanmin(((self.max, other.max)), axis=0),
        )

    def __repr__(self) -> str:
        return f"BBox({self.min}, {self.max})"