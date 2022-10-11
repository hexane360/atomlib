from __future__ import annotations

import typing as t

import numpy


if t.TYPE_CHECKING:
    from .types import Vec3, VecLike


class BBox:
    """3D Bounding Box"""

    def __init__(self, min: Vec3, max: Vec3):
        # shape: (3, 2)
        # self._inner[1, 0]: min of y
        self.inner: numpy.ndarray = numpy.stack((min, max), axis=-1)

    @property
    def min(self) -> Vec3:
        return self.inner[:, 0]

    @property
    def max(self) -> Vec3:
        return self.inner[:, 1]

    @property
    def x(self) -> numpy.ndarray:
        return self.inner[0]

    @property
    def y(self) -> numpy.ndarray:
        return self.inner[1]

    @property
    def z(self) -> numpy.ndarray:
        return self.inner[2]

    @property
    def size(self) -> Vec3:
        return self.max - self.min

    def volume(self) -> float:
        return numpy.prod(self.size)  # type: ignore

    def corners(self) -> numpy.ndarray:
        """Return a (8, 3) ndarray containing the corners of self."""
        return numpy.stack(list(map(numpy.ravel, numpy.meshgrid(*self.inner))), axis=-1)

    def pad(self, amount: t.Union[float, VecLike]) -> BBox:
        """
        Pad the given BBox by `amount`. If a vector [x, y, z] is given,
        pad each axis by the given amount.
        """

        if isinstance(amount, float):
            amount_v = numpy.full(3, amount)
        else:
            amount_v = numpy.broadcast_to(amount, 3)

        return type(self)(
            self.min - amount_v,
            self.max + amount_v
        )

    @classmethod
    def from_pts(cls, pts: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> BBox:
        """Construct a BBox containing 'pts'."""
        pts = numpy.atleast_2d(pts).reshape(-1, 3)
        return cls(numpy.nanmin(pts, axis=0), numpy.nanmax(pts, axis=0))

    def __or__(self, other: t.Union[Vec3, BBox]) -> BBox:
        if isinstance(other, numpy.ndarray):
            return self.from_pts((self.min, self.max, other))

        return type(self)(
            numpy.nanmin(((self.min, other.min)), axis=0),
            numpy.nanmax(((self.max, other.max)), axis=0),
        )

    __ror__ = __or__

    def __and__(self, other: BBox) -> BBox:
        return type(self)(
            numpy.nanmax(((self.min, other.min)), axis=0),
            numpy.nanmin(((self.max, other.max)), axis=0),
        )

    def __repr__(self) -> str:
        return f"BBox({self.min}, {self.max})"