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

    @t.overload
    def __mul__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __mul__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __mul__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return super().__mul__(other)
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
            return super().__truediv__(other)
        return super().__truediv__(other).view(Vec3)

    @t.overload
    def __rtruediv__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __rtruediv__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __rtruediv__(self, other) -> numpy.ndarray:
        return self.__truediv__(other)

    @t.overload
    def __add__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __add__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __add__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return super().__add__(other)
        return super().__add__(other).view(Vec3)

    @t.overload
    def __radd__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __radd__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __radd__(self, other) -> numpy.ndarray:
        return self.__add__(other)

    @t.overload
    def __sub__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __sub__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __sub__(self, other) -> numpy.ndarray:
        if isinstance(other, numpy.ndarray) and not isinstance(other, Vec3):
            return super().__sub__(other)
        return super().__sub__(other).view(Vec3)

    @t.overload
    def __rsub__(self, other: t.Union[float, int, complex, Vec3]) -> Vec3:
        ...

    @t.overload
    def __rsub__(self, other: numpy.ndarray) -> numpy.ndarray:
        ...

    def __rsub__(self, other) -> numpy.ndarray:
        return self.__sub__(other)