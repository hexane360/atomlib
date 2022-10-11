from __future__ import annotations

import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray


def dot(v1: ArrayLike, v2: ArrayLike, axis: int = -1, keepdims: bool = True) -> NDArray[numpy.floating]:
    return numpy.add.reduce(numpy.atleast_1d(v1) * numpy.atleast_1d(v2), axis=axis, keepdims=keepdims)


def norm(v: ArrayLike) -> numpy.floating:
    return numpy.linalg.norm(v)


def perp(v1: ArrayLike, v2: ArrayLike) -> NDArray[numpy.floating]:
    """Return the component of `v1` perpendicular to `v2`."""
    v1 = numpy.atleast_1d(v1)
    v2 = numpy.atleast_1d(v2)
    v2 /= norm(v2)
    return v1 - v2 * dot(v1, v2)


def para(v1: ArrayLike, v2: ArrayLike) -> NDArray[numpy.floating]:
    """Return the component of `v1` parallel to `v2`."""
    v1 = numpy.atleast_1d(v1)
    v2 = numpy.atleast_1d(v2)
    v2 /= norm(v2)
    return v2 * dot(v1, v2)