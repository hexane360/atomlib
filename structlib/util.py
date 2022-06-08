
from io import TextIOBase, TextIOWrapper
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray


T = t.TypeVar('T')
U = t.TypeVar('U')
ScalarType = t.TypeVar("ScalarType", bound=numpy.generic, covariant=True)


def map_some(f: t.Callable[[T], U], val: t.Optional[T]) -> t.Optional[U]:
    if val is None:
        return None
    return f(val)


def split_ndarray(a: NDArray[ScalarType], axis: int = 0) -> t.Iterator[NDArray[ScalarType]]:
    return (numpy.squeeze(sub_a, axis) for sub_a in numpy.split(a, a.shape[axis], axis))


FileOrPath = t.Union[str, Path, TextIOBase, t.TextIO]


def open_file(f: FileOrPath,
              mode: t.Union[t.Literal['r'], t.Literal['w']] = 'r',
              newline: t.Optional[str] = None,
              encoding: t.Optional[str] = 'utf-8') -> TextIOBase:
    if isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)
    if not isinstance(f, TextIOBase):
        return open(f, mode, newline=newline, encoding=encoding)
    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)

    if mode == 'r':
        if not f.readable():
            raise RuntimeError("Error: Provided file not readable.")
    elif mode == 'w':
        if not f.writable():
            raise RuntimeError("Error: Provided file not writable.")

    return f


def miller_4_to_3_vec(a: numpy.ndarray) -> numpy.ndarray:
    """Convert a vector in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    assert a.shape[-1] == 4
    U, V, T, W = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-T, U + V, equal_nan=True)
    return numpy.stack((2*U + V, 2*V + U, W), axis=-1)


def miller_3_to_4_vec(a: numpy.ndarray) -> numpy.ndarray:
    """Convert a vector in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    assert a.shape[-1] == 3
    u, v, w = numpy.split(a, 3, axis=-1)
    U = (2*u - v)/3
    V = (2*v - u)/3
    return numpy.stack((U, V, -(U + V), w), axis=-1)


def miller_4_to_3_plane(a: numpy.ndarray) -> numpy.ndarray:
    """Convert a plane in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    assert a.shape[-1] == 4
    h, k, i, l = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-i, h + k, equal_nan=True)
    return numpy.stack((h, k, l), axis=-1)


def miller_3_to_4_plane(a: numpy.ndarray) -> numpy.ndarray:
    """Convert a plane in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    assert a.shape[-1] == 3
    h, k, l = numpy.split(a, 3, axis=-1)
    return numpy.stack((h, k, -(h + k), l), axis=-1)  # type: ignore