
from io import TextIOBase, TextIOWrapper
from pathlib import Path
from fractions import Fraction
import typing as t

import numpy
from numpy.typing import NDArray
import polars


T = t.TypeVar('T')
U = t.TypeVar('U')
ScalarType = t.TypeVar("ScalarType", bound=numpy.generic, covariant=True)


def map_some(f: t.Callable[[T], U], val: t.Optional[T]) -> t.Optional[U]:
    if val is None:
        return None
    return f(val)


def split_arr(a: NDArray[ScalarType], axis: int = 0) -> t.Iterator[NDArray[ScalarType]]:
    return (numpy.squeeze(sub_a, axis) for sub_a in numpy.split(a, a.shape[axis], axis))


FileOrPath = t.Union[str, Path, TextIOBase, t.TextIO]


def open_file(f: FileOrPath,
              mode: t.Union[t.Literal['r'], t.Literal['w']] = 'r',
              newline: t.Optional[str] = None,
              encoding: t.Optional[str] = 'utf-8') -> TextIOBase:
    if not isinstance(f, (TextIOBase, t.TextIO)):
        return open(f, mode, newline=newline, encoding=encoding)

    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)
    elif isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)

    if mode == 'r':
        if not f.readable():
            raise RuntimeError("Error: Provided file not readable.")
    elif mode == 'w':
        if not f.writable():
            raise RuntimeError("Error: Provided file not writable.")

    return f


def polygon_winding(poly: numpy.ndarray, pt: numpy.ndarray) -> NDArray[numpy.int_]:
    poly = t.cast(NDArray[numpy.floating], numpy.atleast_2d(poly))
    pt = numpy.atleast_1d(pt)[..., None, :]

    poly = poly - pt
    poly_next = numpy.roll(poly, -1, axis=-2)
    (x, y) = split_arr(poly, axis=-1)
    (xn, yn) = split_arr(poly_next, axis=-1)

    x_pos = x*(yn - y) - y*(xn - x)  # |p1 cross (p2 - p1)| -> (p2 - p1) to right or left of origin
    up_crossing = (y <= 0) & (yn > 0) & (x_pos > 0)
    down_crossing = (y > 0) & (yn <= 0) & (x_pos < 0)

    print(f"y: {y}\nyn: {yn}")

    print(f"up: {up_crossing}")
    print(f"down: {down_crossing}")
    print(f"x_pos: {x_pos}")

    return (numpy.sum(up_crossing.astype(int), axis=-1) - numpy.sum(down_crossing.astype(int), axis=-1))


WindingRule = t.Union[t.Literal['nonzero'], t.Literal['evenodd'], t.Literal['positive'], t.Literal['negative']]


@t.overload
def in_polygon(poly: numpy.ndarray, pt: t.Literal[None] = None, rule: WindingRule = 'evenodd') -> t.Callable[[numpy.ndarray], NDArray[numpy.bool_]]:
    ...


@t.overload
def in_polygon(poly: numpy.ndarray, pt: numpy.ndarray, rule: WindingRule = 'evenodd') -> NDArray[numpy.bool_]:
    ...


def in_polygon(poly: numpy.ndarray, pt: t.Optional[numpy.ndarray] = None,
               rule: WindingRule = 'evenodd') -> t.Union[NDArray[numpy.bool_], t.Callable[[numpy.ndarray], NDArray[numpy.bool_]]]:
    if pt is None:
        return lambda pt: in_polygon(poly, pt, rule)
    winding = polygon_winding(poly, pt)

    if rule == 'nonzero':
        return winding.astype(numpy.bool_)
    elif rule == 'evenodd':
        return (winding & 1) > 0
    elif rule == 'positive':
        return winding > 0
    elif rule == 'negative':
        return winding < 0


def reduce_vec(a: numpy.ndarray, max_denom: int = 10000) -> numpy.ndarray:
    """
    Reduce a crystallographic vector (int or float) to lowest common terms.
    Example: reduce_vec([3, 3, 3]) = [1, 1, 1]
    reduce_vec([0.25, 0.25, 0.25]) = [1, 1, 1]
    """
    a = numpy.atleast_1d(a)
    if not numpy.issubdtype(a.dtype, numpy.floating):
        return a // numpy.gcd.reduce(a, axis=-1, keepdims=True)

    n = numpy.empty(shape=a.shape, dtype=numpy.int64)
    d = numpy.empty(shape=a.shape, dtype=numpy.int64)
    with numpy.nditer([a, n, d], ['refs_ok'], [['readonly'], ['writeonly'], ['writeonly']]) as it:  # type: ignore
        for (v, n_, d_) in it:
            (n_[()], d_[()]) = Fraction(float(v)).limit_denominator(max_denom).as_integer_ratio()

    # reduce to common denominator
    factors = numpy.lcm.reduce(d, axis=-1, keepdims=True) // d
    n *= factors
    # and then reduce numerators
    return n // numpy.gcd.reduce(n, axis=-1, keepdims=True)


def miller_4_to_3_vec(a: numpy.ndarray, reduce: bool = True, max_denom: int = 10000) -> numpy.ndarray:
    """Convert a vector in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 4
    U, V, T, W = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-T, U + V, equal_nan=True)
    u = 2*U + V
    v = 2*V + U
    w = W
    out = numpy.concatenate((2*U + V, 2*V + U, W), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_3_to_4_vec(a: numpy.ndarray, reduce: bool = True, max_denom: int = 10000) -> numpy.ndarray:
    """Convert a vector in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 3
    u, v, w = numpy.split(a, 3, axis=-1)
    U = 2*u - v
    V = 2*v - u
    W = 3*w
    out = numpy.concatenate((U, V, -(U + V), W), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_4_to_3_plane(a: numpy.ndarray, reduce: bool = True, max_denom: int = 10000) -> numpy.ndarray:
    """Convert a plane in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 4
    h, k, i, l = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-i, h + k, equal_nan=True)
    out = numpy.concatenate((h, k, l), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_3_to_4_plane(a: numpy.ndarray, reduce: bool = True, max_denom: int = 10000) -> numpy.ndarray:
    """Convert a plane in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 3
    h, k, l = numpy.split(a, 3, axis=-1)
    out = numpy.concatenate((h, k, -(h + k), l), axis=-1)  # type: ignore
    return reduce_vec(out, max_denom) if reduce else out