import typing as t

from typing_extensions import TypeAlias
import numpy
from numpy.typing import NDArray, ArrayLike

# pyright: reportImportCycles=false
if t.TYPE_CHECKING:
    from .bbox import BBox3D


Vec3: TypeAlias = NDArray[numpy.floating[t.Any]]
"""3D float vector, of shape (3,)."""

VecLike: TypeAlias = ArrayLike
"""3d vector-like"""

Pts3DLike: TypeAlias = t.Union['BBox3D', ArrayLike]
"""Sequence of 3d points-like"""

Num: TypeAlias = t.Union[float, int]
"""Scalar numeric type"""

ElemLike: TypeAlias = t.Union[str, int]
"""Element-like"""

ElemsLike: TypeAlias = t.Union[str, int, t.Sequence[t.Union[ElemLike, t.Tuple[ElemLike, float]]]]

NumT = t.TypeVar('NumT', bound=numpy.number)
"""[`numpy.number`][numpy.number]-bound type variable"""

ScalarT = t.TypeVar('ScalarT', bound=numpy.generic)
"""[`numpy.generic`][numpy.generic]-bound type variable"""


@t.overload
def to_vec3(v: VecLike, dtype: None = None) -> NDArray[numpy.float64]:
    ...

@t.overload
def to_vec3(v: VecLike, dtype: t.Type[ScalarT]) -> NDArray[ScalarT]:
    ...

def to_vec3(v: VecLike, dtype: t.Optional[t.Type[numpy.generic]] = None) -> NDArray[numpy.generic]:
    """
    Broadcast and coerce `v` to a [`Vec3`][atomlib.types.Vec3] of type `dtype`.
    """

    try:
        v = numpy.broadcast_to(v, (3,)).astype(dtype or numpy.float64)
    except (ValueError, TypeError):
        raise TypeError("Expected a vector of 3 elements.") from None
    return v

__all__ = [
    'Vec3', 'VecLike', 'Pts3DLike', 'ElemLike',
    'ScalarT', 'NumT', 'Num',
    'to_vec3',
]
