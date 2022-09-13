"""
Helper functions for working with crystallographic unit cells
"""

import typing as t

import numpy

from .vec import Vec3
from .transform import LinearTransform
from .types import VecLike, to_vec3


@t.overload
def _validate_cell_size(cell_size: VecLike) -> Vec3:
    ...

@t.overload
def _validate_cell_size(cell_size: t.Literal[None]) -> t.Literal[None]:
    ...

def _validate_cell_size(cell_size: t.Optional[VecLike]) -> t.Optional[Vec3]:
    if cell_size is None:
        return cell_size
    cell_size = to_vec3(cell_size)
    if (numpy.isclose(cell_size, 0)).any():
        raise ValueError(f"Zero cell dimension: {cell_size}")
    return cell_size


def _validate_cell_angle(cell_angle: t.Optional[VecLike]) -> Vec3:
    if cell_angle is None:
        return numpy.pi/2. * numpy.ones((3,)).view(Vec3)
    cell_angle = to_vec3(cell_angle)
    if (cell_angle < 0).any() or (cell_angle > numpy.pi).any() or cell_angle.sum() > 2*numpy.pi:
        raise ValueError(f"Invalid cell angle: {cell_angle}")
    return cell_angle.view(Vec3)


def cell_to_ortho(cell_size: VecLike, cell_angle: t.Optional[VecLike] = None) -> LinearTransform:
    """Get orthogonalization transform from unit cell parameters (which turns fractional cell coordinates into real-space coordinates)."""
    cell_size = _validate_cell_size(cell_size)
    cell_angle = _validate_cell_angle(cell_angle)
    (a, b, c) = cell_size if cell_size is not None else (1., 1., 1.)

    if numpy.allclose(cell_angle.view(numpy.ndarray), numpy.pi/2.):
        return LinearTransform().scale(a, b, c)

    (alpha, beta, gamma) = cell_angle
    alphastar = numpy.cos(beta) * numpy.cos(gamma) - numpy.cos(alpha)
    alphastar /= numpy.sin(beta) * numpy.sin(gamma)
    alphastar = numpy.arccos(alphastar)
    assert not numpy.isnan(alphastar)

    # aligns a axis along x
    # aligns b axis in the x-y plane
    return LinearTransform(numpy.array([
        [a,  b * numpy.cos(gamma),  c * numpy.cos(beta)],
        [0,  b * numpy.sin(gamma), -c * numpy.sin(beta) * numpy.cos(alphastar)],
        [0,  0,                     c * numpy.sin(beta) * numpy.sin(alphastar)],
    ])).round_near_zero()


def ortho_to_cell(ortho: LinearTransform) -> t.Tuple[Vec3, Vec3]:
    """Get unit cell parameters (cell_size, cell_angle) from orthogonalization transform."""
    # TODO suspect
    cell_size = numpy.linalg.norm(ortho.inner, axis=-2).view(Vec3)
    cell_size = _validate_cell_size(cell_size)
    normed = ortho.inner / cell_size
    cosines = numpy.array([
        numpy.dot(normed[..., 1], normed[..., 2]), # alpha
        numpy.dot(normed[..., 2], normed[..., 0]), # beta
        numpy.dot(normed[..., 0], normed[..., 1]), # gamma
    ])
    cell_angle = numpy.arccos(cosines).view(Vec3)
    cell_angle = _validate_cell_angle(cell_angle)

    return (cell_size, cell_angle)
