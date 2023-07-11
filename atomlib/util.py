
from io import BufferedIOBase, TextIOBase, TextIOWrapper, IOBase, StringIO, BytesIO
from pathlib import Path
from contextlib import nullcontext, AbstractContextManager
import datetime
import time
import typing as t

from .types import ParamSpec, Concatenate


T = t.TypeVar('T')
U = t.TypeVar('U')
P = ParamSpec('P')
U_co = t.TypeVar('U_co', covariant=True)


def map_some(f: t.Callable[[T], U], val: t.Optional[T]) -> t.Optional[U]:
    """
    Map ``f`` over ``val`` if not ``None``.
    """
    return None if val is None else f(val)


FileOrPath = t.Union[str, Path, TextIOBase, t.TextIO]
"""Open text file or path to a file. Use with :func:`open_file`."""
BinaryFileOrPath = t.Union[str, Path, t.TextIO, t.BinaryIO, IOBase]
"""Open binary file or path to a file. Use with :func:`open_file_binary`."""


def _validate_file(f: t.Union[t.IO, IOBase], mode: t.Union[t.Literal['r'], t.Literal['w']]):
    if f.closed:
        raise IOError("Error: Provided file is closed.")

    if mode == 'r':
        if not f.readable():
            raise IOError("Error: Provided file not readable.")
    elif mode == 'w':
        if not f.writable():
            raise IOError("Error: Provided file not writable.")


def open_file(f: FileOrPath,
              mode: t.Union[t.Literal['r'], t.Literal['w']] = 'r',
              newline: t.Optional[str] = None,
              encoding: t.Optional[str] = 'utf-8') -> AbstractContextManager[TextIOBase]:
    """
    Open the given file for text I/O.

    If given a path-like, opens it with the specified settings.
    Otherwise, make an effort to reconfigure the encoding, and
    check that it is readable/writable as specified.
    """
    if not isinstance(f, (IOBase, t.BinaryIO, t.TextIO)):
        return open(f, mode, newline=newline, encoding=encoding)

    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)
    elif isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)
    elif isinstance(f, (BufferedIOBase, t.BinaryIO)):
        f = TextIOWrapper(t.cast(t.BinaryIO, f), newline=newline, encoding=encoding)

    _validate_file(f, mode)
    return nullcontext(f)  # don't close a f we didn't open


def open_file_binary(f: BinaryFileOrPath,
                     mode: t.Union[t.Literal['r'], t.Literal['w']] = 'r') -> AbstractContextManager[IOBase]:
    """
    Open the given file for binary I/O.

    If given a path-like, opens it with the specified settings. If given text I/O,
    reconfigure to binary. Make sure stream is readable/writable, as specified.
    """
    if not isinstance(f, (IOBase, t.BinaryIO, t.TextIO)):
        return t.cast(IOBase, open(f, mode + 'b'))

    if isinstance(f, (TextIOWrapper, t.TextIO)):
        try:
            f = f.buffer
        except AttributeError:
            raise ValueError("Error: Couldn't get raw buffer from text file.")
    elif isinstance(f, StringIO):
        if mode == 'w':
            raise ValueError("Can't write binary stream to StringIO.")
        return BytesIO(f.getvalue().encode('utf-8'))
    elif isinstance(f, TextIOBase):
        raise ValueError(f"Error: Couldn't get binary stream from text stream of type '{type(f)}'.")

    _validate_file(f, mode)
    return nullcontext(t.cast(IOBase, f))  # don't close a file we didn't open


def localtime() -> datetime.datetime:
    """Return the current time in a timezone-aware datetime object."""
    ltime = time.localtime()
    tz = datetime.timezone(datetime.timedelta(seconds=ltime.tm_gmtoff), ltime.tm_zone)
    return datetime.datetime.now(tz)


class opt_classmethod(classmethod, t.Generic[T, P, U_co]):
    """
    Decorates a method that may be called either on an instance or the class.
    If called on the class, a default instance will be constructed before
    calling the wrapped function.
    """

    __func__: t.Callable[Concatenate[T, P], U_co]  # type: ignore
    def __init__(self, f: t.Callable[Concatenate[T, P], U_co]):
        super().__init__(f)

    def __get__(self, obj: t.Optional[T], ty: t.Optional[t.Type[T]] = None) -> t.Callable[P, U_co]:  # type: ignore
        if obj is None:
            if ty is None:
                raise RuntimeError()  # pragma: no cover
            obj = ty()
        return t.cast(
            t.Callable[P, U_co],
            super().__get__(obj, obj)  # type: ignore
        )

__all__ = [
    'open_file', 'open_file_binary', 'FileOrPath', 'BinaryFileOrPath',
    'opt_classmethod', 'localtime', 'map_some',
]
