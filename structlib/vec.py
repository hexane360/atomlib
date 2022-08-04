from __future__ import annotations

import typing as t

import numpy


class Vec3(numpy.ndarray):
    def __array_finalize__(self, obj):
        if not self.shape == (3,):
            raise ValueError(f"Expected array of shape (3,), instead got: {self.shape}")

    @classmethod
    def make(cls: t.Type[Vec3], val: t.Tuple[float, float, float]) -> Vec3:
        return numpy.array(val).view(cls)

    def any(self, *args, **kwargs) -> t.Union[numpy.bool_, numpy.ndarray]:
        return self.view(numpy.ndarray).any(*args, **kwargs)

    def all(self, *args, **kwargs) -> t.Union[numpy.bool_, numpy.ndarray]:
        return self.view(numpy.ndarray).all(*args, **kwargs)

    def sum(self, *args, **kwargs) -> numpy.ndarray:
        return self.view(numpy.ndarray).sum(*args, **kwargs)

    @t.overload
    def __mul__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __mul__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __mul__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__mul__(other)
        return super().__mul__(other).view(Vec3)

    @t.overload
    def __rmul__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __rmul__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __rmul__(self, other) -> numpy.ndarray:
        return self.__mul__(other)

    @t.overload
    def __truediv__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __truediv__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __truediv__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__truediv__(other)
        return super().__truediv__(self).view(Vec3)

    @t.overload
    def __rtruediv__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __rtruediv__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __rtruediv__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__rtruediv__(other)
        return super().__rtruediv__(self).view(Vec3)

    @t.overload
    def __add__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __add__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __add__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__add__(other)
        return super().__add__(other).view(Vec3)

    @t.overload
    def __radd__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __radd__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __radd__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__radd__(other)
        return super().__radd__(other).view(Vec3)

    @t.overload
    def __sub__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __sub__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __sub__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__sub__(other)
        return super().__sub__(other).view(Vec3)

    @t.overload
    def __rsub__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __rsub__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __rsub__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return self.view(numpy.ndarray).__rsub__(other)
        return super().__rsub__(other).view(Vec3)


class BBox:
    """3D Bounding Box"""

    def __init__(self, min: Vec3, max: Vec3):
        # shape: (3, 2)
        # self._inner[1, 0]: min of y
        self._inner: numpy.ndarray = numpy.stack((min, max), axis=-1)

    @property
    def min(self) -> Vec3:
        return self._inner[:, 0].view(Vec3)

    @property
    def max(self) -> Vec3:
        return self._inner[:, 1].view(Vec3)

    @property
    def size(self) -> Vec3:
        return self.max - self.min

    def volume(self) -> float:
        return numpy.prod(self.size)

    def corners(self) -> numpy.ndarray:
        """Return a (8, 3) ndarray containing the corners of self."""
        return numpy.stack(list(map(numpy.ravel, numpy.meshgrid(*self._inner))), axis=-1)

    def pad(self, amount: t.Union[float, Vec3]) -> BBox:
        """
        Pad the given BBox by `amount`. If a vector [x, y, z] is given,
        pad each axis by the given amount.
        """

        if isinstance(amount, float):
            amount_v = numpy.full(3, amount).view(Vec3)
        else:
            amount_v = numpy.broadcast_to(amount, 3).view(Vec3)

        return type(self)(
            self.min - amount_v,
            self.max + amount_v
        )

    @classmethod
    def from_pts(cls, pts: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> BBox:
        """Construct a BBox containing 'pts'."""
        pts = numpy.atleast_2d(pts).reshape(-1, 3)
        return cls(numpy.min(pts, axis=0), numpy.max(pts, axis=0))

    def __or__(self, other: t.Union[Vec3, BBox]) -> BBox:
        if isinstance(other, numpy.ndarray):
            return self.from_pts((self.min, self.max, other))

        return type(self)(
            numpy.min(((self.min, other.min)), axis=0),
            numpy.max(((self.max, other.max)), axis=0),
        )

    __ror__ = __or__

    def __and__(self, other: BBox) -> BBox:
        return type(self)(
            numpy.max(((self.min, other.min)), axis=0),
            numpy.min(((self.max, other.max)), axis=0),
        )

    def __repr__(self) -> str:
        return f"BBox({self.min}, {self.max})"