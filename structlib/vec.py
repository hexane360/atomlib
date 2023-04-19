"""Helper functions for spatial vectors."""
from __future__ import annotations

from fractions import Fraction
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

# re-export to_vec3
from .types import to_vec3, ScalarT


def dot(v1: ArrayLike, v2: ArrayLike, axis: int = -1, keepdims: bool = True) -> NDArray[numpy.floating]:
    return numpy.add.reduce(numpy.atleast_1d(v1) * numpy.atleast_1d(v2), axis=axis, keepdims=keepdims)


def norm(v: ArrayLike) -> numpy.floating:
    return numpy.linalg.norm(v)


def perp(v1: ArrayLike, v2: ArrayLike) -> NDArray[numpy.floating]:
    """Return the component of ``v1`` perpendicular to ``v2``."""
    v1 = numpy.atleast_1d(v1)
    v2 = numpy.atleast_1d(v2)
    v2 /= norm(v2)
    return v1 - v2 * dot(v1, v2)


def para(v1: ArrayLike, v2: ArrayLike) -> NDArray[numpy.floating]:
    """Return the component of ``v1`` parallel to ``v2``."""
    v1 = numpy.atleast_1d(v1)
    v2 = numpy.atleast_1d(v2)
    v2 /= norm(v2)
    return v2 * dot(v1, v2)


def is_diagonal(matrix: numpy.ndarray, tol: float = 1e-10) -> bool:
    d = matrix.shape[0]
    assert matrix.shape == (d, d)
    p, q = matrix.strides
    offdiag = numpy.lib.stride_tricks.as_strided(matrix[:, 1:], (d-1, d), (p+q, q))
    return bool((numpy.abs(offdiag) < tol).all())


def split_arr(a: NDArray[ScalarT], axis: int = 0) -> t.Iterator[NDArray[ScalarT]]:
    return (numpy.squeeze(sub_a, axis) for sub_a in numpy.split(a, a.shape[axis], axis))


def polygon_solid_angle(poly: ArrayLike, pts: t.Optional[ArrayLike] = None,
                        winding: t.Optional[ArrayLike] = None) -> NDArray[numpy.float_]:
    """
    Return the signed solid angle of the polygon ``poly`` in the xy plane, as viewed from ``pts``.

    ``poly``: ndarray of shape (..., N, 2)
    ``pts``: ndarray of shape (..., 3)

    Returns a ndarray of shape ``broadcast(poly.shape[:-2], pts.shape[:-1])``
    """
    poly = numpy.atleast_2d(poly).astype(numpy.float_)
    pts = (numpy.array([0., 0., 0.]) if pts is None else numpy.atleast_1d(pts)).astype(numpy.float_)

    if poly.shape[-1] == 3:
        raise ValueError("Only 2d polygons are supported.")
    if poly.shape[-1] != 2:
        raise ValueError("`poly` must be a list of 2d points.")
    if winding is None:
        # calculate winding
        winding = polygon_winding(poly)
    else:
        winding = numpy.asarray(winding, dtype=int)
    # extend to 3d
    poly = numpy.concatenate((poly, numpy.zeros_like(poly, shape=(*poly.shape[:-1], 1))), axis=-1)

    if pts.shape[-1] != 3:
        raise ValueError("`pts` must be a list of 3d points.")

    poly = poly - pts[..., None, :]
    # normalize polygon points to unit sphere
    numpy.divide(poly, numpy.linalg.norm(poly, axis=-1, keepdims=True), out=poly)

    def _dot(v1: NDArray[numpy.float_], v2: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
        return numpy.add.reduce(v1 * v2, axis=-1)

    # next and previous points in polygon
    poly_n = numpy.roll(poly, -1, axis=-2)
    poly_p = numpy.roll(poly, 1, axis=-2)

    # spherical angle is 2*pi - sum(atan2(-|v1v2v3|, v1 dot v2 * v2 dot v3 - v1 dot v3))
    angles = numpy.arctan2(_dot(poly_p, numpy.cross(poly, poly_n)), _dot(poly_p, poly) * _dot(poly, poly_n) - _dot(poly_p, poly_n))
    angle = numpy.sum(angles, axis=-1)

    # when winding is nonzero, we have to offset the calculated angle by the angle created by winding.
    numpy.mod(angle, 4*numpy.pi*winding, out=angle, where=(winding != 0))
    return angle - 2*numpy.pi*winding


def polygon_winding(poly: ArrayLike, pt: t.Optional[ArrayLike] = None) -> NDArray[numpy.int_]:
    """
    Return the winding number of the given 2d polygon ``poly`` around the point ``pt``.
    If ``pt`` is not specified, return the polygon's total winding number (turning number).

    Vectorized. CCW winding is defined as positive.
    """
    poly = numpy.atleast_2d(poly)
    if poly.dtype == object:
        raise ValueError("Ragged arrays not supported.")
    poly = poly.astype(numpy.float_)

    if pt is None:
        # return polygon's total winding number (turning number)
        poly_next = numpy.roll(poly, -1, axis=-2)
        # equivalent to the turning number of velocity vectors (difference vectors)
        poly = poly_next - poly
        # about the origin
        pt = numpy.array([0., 0.])

        # remove points at the origin (duplicate points)
        zero_pts = (numpy.isclose(poly[..., 0], 0., atol=1e-10) & 
                    numpy.isclose(poly[..., 1], 0., atol=1e-10))
        poly = poly[~zero_pts]

    pt = numpy.atleast_1d(pt)[..., None, :].astype(numpy.float_)

    # shift the polygon's origin to `pt`.
    poly = poly - pt
    poly_next = numpy.roll(poly, -1, axis=-2)
    (x, y) = split_arr(poly, axis=-1)
    (xn, yn) = split_arr(poly_next, axis=-1)

    # |p1 cross (p2 - p1)| -> (p2 - p1) to right or left of origin
    x_pos = x*(yn - y) - y*(xn - x)  # type: ignore
    # count up crossings and down crossings
    up_crossing = (y <= 0) & (yn > 0) & (x_pos > 0)
    down_crossing = (y > 0) & (yn <= 0) & (x_pos < 0)

    # reduce and return
    return numpy.sum(up_crossing, axis=-1) - numpy.sum(down_crossing, axis=-1)


WindingRule = t.Literal['nonzero', 'evenodd', 'positive', 'negative']


@t.overload
def in_polygon(poly: numpy.ndarray, pt: t.Literal[None] = None, *, rule: WindingRule = 'evenodd') -> t.Callable[[numpy.ndarray], NDArray[numpy.bool_]]:
    ...


@t.overload
def in_polygon(poly: numpy.ndarray, pt: numpy.ndarray, *, rule: WindingRule = 'evenodd') -> NDArray[numpy.bool_]:
    ...


def in_polygon(poly: numpy.ndarray, pt: t.Optional[numpy.ndarray] = None, *,
               rule: WindingRule = 'evenodd') -> t.Union[NDArray[numpy.bool_], t.Callable[[numpy.ndarray], NDArray[numpy.bool_]]]:
    """
    Return whether `pt` is in `poly`, under the given winding rule.
    In the one-argument form, return a closure which tests `poly` for the given point.
    """
    if pt is None:
        return lambda pt: in_polygon(poly, pt, rule=rule)
    winding = polygon_winding(poly, pt)

    rule = t.cast(WindingRule, rule.lower())
    if rule == 'nonzero':
        return winding.astype(numpy.bool_)
    elif rule == 'evenodd':
        return (winding & 1) > 0
    elif rule == 'positive':
        return winding > 0
    elif rule == 'negative':
        return winding < 0
    raise ValueError(f"Unknown winding rule '{rule}'. Expected one of "
                     "'nonzero', 'evenodd', 'positive', or 'negative'.")


def reduce_vec(arr: ArrayLike, max_denom: int = 10000) -> NDArray[numpy.int_]:
    """
    Reduce a crystallographic vector (int or float) to lowest common terms.
    Example: reduce_vec([3, 3, 3]) = [1, 1, 1]
    reduce_vec([0.25, 0.25, 0.25]) = [1, 1, 1]
    """
    a = numpy.atleast_1d(arr)
    if not numpy.issubdtype(a.dtype, numpy.floating):
        return a // numpy.gcd.reduce(a, axis=-1, keepdims=True)

    a = a / numpy.max(numpy.abs(a))

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


def miller_4_to_3_vec(a: NDArray[numpy.number], reduce: bool = True, max_denom: int = 10000) -> NDArray[numpy.number]:
    """Convert a vector in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 4
    U, V, T, W = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-T, U + V, equal_nan=True)
    out = numpy.concatenate((2*U + V, 2*V + U, W), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_3_to_4_vec(a: NDArray[numpy.number], reduce: bool = True, max_denom: int = 10000) -> NDArray[numpy.number]:
    """Convert a vector in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 3
    u, v, w = numpy.split(a, 3, axis=-1)
    U = 2*u - v
    V = 2*v - u
    W = 3*w
    out = numpy.concatenate((U, V, -(U + V), W), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_4_to_3_plane(a: NDArray[numpy.number], reduce: bool = True, max_denom: int = 10000) -> NDArray[numpy.number]:
    """Convert a plane in 4-axis Miller-Bravais notation to 3-axis Miller notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 4
    h, k, i, l = numpy.split(a, 4, axis=-1)
    assert numpy.allclose(-i, h + k, equal_nan=True)
    out = numpy.concatenate((h, k, l), axis=-1)
    return reduce_vec(out, max_denom) if reduce else out


def miller_3_to_4_plane(a: NDArray[numpy.number], reduce: bool = True, max_denom: int = 10000) -> NDArray[numpy.number]:
    """Convert a plane in 3-axis Miller notation to 4-axis Miller-Bravais notation."""
    a = numpy.atleast_1d(a)
    assert a.shape[-1] == 3
    h, k, l = numpy.split(a, 3, axis=-1)
    out = numpy.concatenate((h, k, -(h + k), l), axis=-1)  # type: ignore
    return reduce_vec(out, max_denom) if reduce else out


__all__ = [
    'dot', 'norm', 'perp', 'para', 'reduce_vec', 'split_arr', 'to_vec3',
    'polygon_winding', 'polygon_solid_angle', 'in_polygon',
    'miller_4_to_3_vec', 'miller_3_to_4_vec',
    'miller_4_to_3_plane', 'miller_3_to_4_plane',
]
