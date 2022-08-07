import typing as t

import numpy

from .vec import Vec3, BBox


VecLike = t.Union[t.Sequence[float], t.Sequence[int], Vec3, numpy.ndarray]
"""3d vector-like"""

PtsLike = t.Union[BBox, Vec3, t.Sequence[Vec3], numpy.ndarray]
""""""

Num = t.Union[float, int]
"""Scalar numeric type"""

ElemLike = t.Union[str, int]
"""Element-like"""


def to_vec3(v: VecLike) -> Vec3:
    try:
        v = numpy.broadcast_to(v, (3,)).view(Vec3)
    except (ValueError, TypeError):
        raise TypeError("Expected a vector of 3 elements.") from None
    return v