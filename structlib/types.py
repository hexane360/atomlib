import sys
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

if t.TYPE_CHECKING:
    from .bbox import BBox3D


if sys.version_info < (3, 10):
    import typing_extensions
    ParamSpec = typing_extensions.ParamSpec
    Concatenate = typing_extensions.Concatenate
else:
    ParamSpec = t.ParamSpec
    Concatenate = t.Concatenate


Vec3 = NDArray[numpy.floating[t.Any]]


VecLike = ArrayLike
"""3d vector-like"""

Pts3DLike = t.Union['BBox3D', ArrayLike]
"""Sequence of 3d points-like"""

Num = t.Union[float, int]
"""Scalar numeric type"""

ElemLike = t.Union[str, int]
"""Element-like"""

NumDTypeT = t.TypeVar('NumDTypeT', bound=numpy.number[t.Any])
DTypeT = t.TypeVar('DTypeT', bound=numpy.dtype[t.Any])


@t.overload
def to_vec3(v: VecLike, dtype: None = None) -> NDArray[numpy.float_]:
    ...

@t.overload
def to_vec3(v: VecLike, dtype: t.Type[NumDTypeT]) -> NDArray[NumDTypeT]:
    ...

def to_vec3(v: VecLike, dtype: t.Optional[t.Type[numpy.number[t.Any]]] = None) -> NDArray[numpy.number[t.Any]]:
    try:
        v = numpy.broadcast_to(v, (3,)).astype(dtype or numpy.float_)
    except (ValueError, TypeError):
        raise TypeError("Expected a vector of 3 elements.") from None
    return v
