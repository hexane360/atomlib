"""
Helper functions for working with crystallographic unit cells and coordinate frames.
"""

from dataclasses import dataclass
import typing as t

import numpy

from .transform import LinearTransform, AffineTransform
from .types import VecLike, Vec3, to_vec3
from .vec import reduce_vec


CoordinateFrame = t.Union[
    t.Literal['cell'], t.Literal['cell_frac'], t.Literal['cell_box'],
    t.Literal['ortho'], t.Literal['ortho_frac'], t.Literal['ortho_box'],
    t.Literal['local'], t.Literal['global']
]
"""
A coordinate frame to use.
 - 'cell': Angstroms along crystal axes
 - 'cell_frac': Fraction of unit cells
 - 'cell_box': Fraction of cell box
 - 'ortho': Angstroms along orthogonal cell
 - 'ortho_frac': Fraction of orthogonal cell
 - 'ortho_box': Fraction of orthogonal box
 - 'local': Angstroms in local coordinate system (with affine transformation)
 - 'global': Angstroms in global coordinate system (with all transformations)
"""


@dataclass(kw_only=True)
class CrystalFrame:
    """Internal class for representing the coordinate systems of a crystal."""

    affine: AffineTransform = AffineTransform()
    ortho: LinearTransform = LinearTransform()
    n_cells: LinearTransform = LinearTransform()
    cell_size: LinearTransform = LinearTransform()

    @staticmethod
    def from_cell(cell_size: VecLike, cell_angle: t.Optional[VecLike] = None, n_cells: t.Optional[VecLike] = None):
        return CrystalFrame(
            ortho = cell_to_ortho(cell_size, cell_angle),
            n_cells = LinearTransform() if n_cells is None else LinearTransform.scale(n_cells)
        )

    @staticmethod
    def from_ortho(ortho: AffineTransform, n_cells: t.Optional[VecLike] = None):
        lin = ortho.to_linear()
        # decompose into orthogonal and upper triangular
        q, r = numpy.linalg.qr(lin.inner)

        cell_size = LinearTransform.scale(numpy.linalg.norm(r, axis=-2))
        return CrystalFrame(
            affine=LinearTransform(q).translate(ortho.translation()),
            ortho=LinearTransform(r) @ cell_size.inverse(),
            cell_size=cell_size,
            n_cells = LinearTransform() if n_cells is None else LinearTransform.scale(n_cells)
        )

    def transform_cell(self, transform: LinearTransform) -> 'CrystalFrame':
        """Apply the given transform to the unit cell, and return a new `CrystalFrame`."""
        raise NotImplementedError()

    def _get_transform_inverse(self, frame: CoordinateFrame) -> AffineTransform:
        """Get the transform from 'frame' to local coordinates."""
        frame = t.cast(CoordinateFrame, frame.lower())

        if frame == 'local' or frame == 'global':
            return LinearTransform()

        if frame.startswith('cell'):
            transform = self.ortho
        elif frame.startswith('ortho'):
            transform = LinearTransform()
        else:
            raise ValueError(f"Unknown coordinate frame '{frame}'")

        if '_' not in frame:
            return transform
        end = frame.split('_', 2)[1]
        if end == 'frac':
            return self.n_cells @ self.cell_size @ transform
        if end == 'box':
            return self.cell_size @ transform
        raise ValueError(f"Unknown coordinate frame '{frame}'")

    def get_transform(self, frame_to: CoordinateFrame, frame_from: t.Optional[CoordinateFrame] = None) -> AffineTransform:
        """Get the transform from local coordinates to 'frame'."""
        if frame_from is None:
            return self._get_transform_inverse(frame_to).inverse()
        if frame_to == 'local' or frame_to == 'global':
            return self._get_transform_inverse(frame_from)
        return self._get_transform_inverse(frame_to).inverse() @ self._get_transform_inverse(frame_from)


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
        return numpy.pi/2. * numpy.ones((3,))
    cell_angle = to_vec3(cell_angle)
    if (cell_angle < 0).any() or (cell_angle > numpy.pi).any() or cell_angle.sum() > 2*numpy.pi:
        raise ValueError(f"Invalid cell angle: {cell_angle}")
    return cell_angle


def cell_to_ortho(cell_size: VecLike, cell_angle: t.Optional[VecLike] = None) -> LinearTransform:
    """Get orthogonalization transform from unit cell parameters (which turns fractional cell coordinates into real-space coordinates)."""
    cell_size = _validate_cell_size(cell_size)
    cell_angle = _validate_cell_angle(cell_angle)
    (a, b, c) = cell_size if cell_size is not None else (1., 1., 1.)

    if numpy.allclose(cell_angle.view(numpy.ndarray), numpy.pi/2.):
        return LinearTransform.scale(a, b, c)

    (alpha, beta, gamma) = cell_angle
    alphastar = numpy.cos(beta) * numpy.cos(gamma) - numpy.cos(alpha)
    alphastar /= numpy.sin(beta) * numpy.sin(gamma)
    alphastar = numpy.arccos(alphastar)
    assert not numpy.isnan(alphastar)

    # aligns a axis along x
    # aligns b axis in the x-y plane
    return LinearTransform(numpy.array([
        [a,  b * numpy.cos(gamma),  c * numpy.cos(beta)],
        [0.,  b * numpy.sin(gamma), -c * numpy.sin(beta) * numpy.cos(alphastar)],
        [0.,  0.,                     c * numpy.sin(beta) * numpy.sin(alphastar)],
    ], dtype=float)).round_near_zero()


def ortho_to_cell(ortho: LinearTransform) -> t.Tuple[Vec3, Vec3]:
    """Get unit cell parameters (cell_size, cell_angle) from orthogonalization transform."""
    # TODO suspect
    cell_size = numpy.linalg.norm(ortho.inner, axis=-2)
    cell_size = _validate_cell_size(cell_size)
    normed = ortho.inner / cell_size
    cosines = numpy.array([
        numpy.dot(normed[..., 1], normed[..., 2]), # alpha
        numpy.dot(normed[..., 2], normed[..., 0]), # beta
        numpy.dot(normed[..., 0], normed[..., 1]), # gamma
    ])
    cell_angle = numpy.arccos(cosines)
    cell_angle = _validate_cell_angle(cell_angle)

    return (cell_size, cell_angle)


def plane_to_zone(metric: LinearTransform, plane: VecLike, reduce: bool = True) -> Vec3:
    """
    Return the zone axis associated with a given crystallographic plane.
    If `reduce` is True, call `reduce_vec` before returning. Otherwise,
    return a unit vector.
    """

    plane = to_vec3(plane)
    if metric.is_orthogonal():
        return plane

    # reciprocal lattice is transpose of inverse of real lattice
    # [b1 b2 b3]^T = [a1 a2 a3]^-1
    # so real indices [uvw] = O^-1 O^-1^T (hkl)
    # O^-1 O^-1^T = (O^T O)^-1 = M^-1
    zone = metric.inverse() @ plane

    if reduce:
        return to_vec3(reduce_vec(zone))
    # otherwise reduce to unit vector
    return zone / float(numpy.linalg.norm(zone))


def zone_to_plane(metric: LinearTransform, zone: VecLike, reduce: bool = True) -> Vec3:
    """
    Return the crystallographic plane associated with a given zone axis.
    If `reduce` is True, call `reduce_vec` before returning. Otherwise,
    return a unit vector.
    """

    zone = to_vec3(zone)
    if metric.is_orthogonal():
        return zone

    plane = metric @ zone

    if reduce:
        return to_vec3(reduce_vec(plane))
    # otherwise reduce to unit vector
    return plane / float(numpy.linalg.norm(plane))