
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