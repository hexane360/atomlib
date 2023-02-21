"""
Helper functions for working with crystallographic unit cells and coordinate frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from warnings import warn
import typing as t

import numpy
from numpy.typing import NDArray

from .transform import LinearTransform3D, AffineTransform3D, Transform3D
from .types import VecLike, Vec3, to_vec3
from .vec import reduce_vec
from .bbox import BBox3D


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


@dataclass(frozen=True, init=False)
class Cell:
    """
    Internal class for representing the coordinate systems of a crystal.

    The overall transformation from crystal coordinates to real-space coordinates is
    is split into four transformations, applied from bottom to top. First is ``n_cells``,
    which scales from fractions of a unit cell to fractions of a supercell. Next is
    ``cell_size``, which scales to real-space units. ``ortho`` is an orthogonalization
    matrix, a det = 1 upper-triangular matrix which transforms crystal axes to
    an orthogonal coordinate system. Finally, ``affine`` contains any remaining
    transformations to the local coordinate system, which atoms are stored in.
    """

    affine: AffineTransform3D = AffineTransform3D()
    ortho: LinearTransform3D = LinearTransform3D()
    cell_size: NDArray[numpy.float_]
    cell_angle: NDArray[numpy.float_] = field(default_factory=lambda: numpy.full(3, numpy.pi/2.))
    n_cells: NDArray[numpy.int_] = field(default_factory=lambda: numpy.ones(3, numpy.int_))
    pbc: NDArray[numpy.bool_] = field(default_factory=lambda: numpy.ones(3, numpy.bool_))

    def __init__(self, *,
        affine: t.Optional[AffineTransform3D] = None, ortho: t.Optional[LinearTransform3D] = None,
        cell_size: VecLike, cell_angle: t.Optional[VecLike] = None,
        n_cells: t.Optional[VecLike] = None, pbc: t.Optional[VecLike]):

        object.__setattr__(self, 'affine', AffineTransform3D() if affine is None else affine)
        object.__setattr__(self, 'ortho', LinearTransform3D() if ortho is None else ortho)
        object.__setattr__(self, 'cell_size', to_vec3(cell_size))
        object.__setattr__(self, 'cell_angle', numpy.full(3, numpy.pi/2.) if cell_angle is None else to_vec3(cell_angle))
        object.__setattr__(self, 'n_cells', numpy.ones(3, numpy.int_) if n_cells is None else to_vec3(n_cells, numpy.int_))
        object.__setattr__(self, 'pbc', numpy.ones(3, numpy.bool_) if pbc is None else to_vec3(pbc, numpy.bool_))

    @property
    def ortho_size(self) -> NDArray[numpy.float_]:
        """
        Return size of orthogonal unit cell.

        Equivalent to the diagonal of the orthogonalization matrix.
        """
        return self.cell_size * numpy.diag(self.ortho.inner)

    @property
    def box_size(self) -> NDArray[numpy.float_]:
        """
        Return size of the cell box.

        Equivalent to ``self.n_cells * self.cell_size``.
        """
        return self.n_cells * self.cell_size

    def corners(self, frame: CoordinateFrame = 'local') -> numpy.ndarray:
        corners = numpy.array(list(itertools.product((0., 1.), repeat=3)))
        return self.get_transform(frame, 'cell_box') @ corners

    def bbox(self, frame: CoordinateFrame = 'local') -> BBox3D:
        """Return the bounding box of the cell box in the given coordinate system."""
        return BBox3D.from_pts(self.corners(frame))

    def is_orthogonal(self, tol: float = 1e-8) -> bool:
        """Returns whether this cell is orthogonal (axes are at right angles.)"""
        return self.ortho.is_diagonal(tol=tol)

    def is_orthogonal_in_local(self, tol: float = 1e-8) -> bool:
        """Returns whether this cell is orthogonal and aligned with the local coordinate system."""
        transform = (self.affine @ self.ortho).to_linear()
        if not transform.is_scaled_orthogonal(tol):
            return False
        normed = transform.inner / numpy.linalg.norm(transform.inner, axis=-2, keepdims=True)
        # every row of transform must be a +/- 1 times a basis vector (i, j, or k)
        return all(
            any(numpy.isclose(numpy.abs(numpy.dot(row, v)), 1., atol=tol) for v in numpy.eye(3))
            for row in normed
        )

    @staticmethod
    def from_unit_cell(cell_size: VecLike, cell_angle: t.Optional[VecLike] = None, n_cells: t.Optional[VecLike] = None,
                       pbc: t.Optional[VecLike] = None):
        return Cell(
            ortho=cell_to_ortho([1.]*3, cell_angle),
            n_cells=to_vec3([1]*3 if n_cells is None else n_cells, numpy.int_),
            cell_size=to_vec3(cell_size),
            cell_angle=to_vec3([numpy.pi/2.]*3 if cell_angle is None else cell_angle),
            pbc=pbc
        )

    @staticmethod
    def from_ortho(ortho: AffineTransform3D, n_cells: t.Optional[VecLike] = None, pbc: t.Optional[VecLike] = None):
        lin = ortho.to_linear()
        # decompose into orthogonal and upper triangular
        q, r = numpy.linalg.qr(lin.inner)

        # flip QR decomposition so R has positive diagonals
        signs = numpy.sign(numpy.diagonal(r))
        # multiply flips to columns of Q, rows of R
        q = q * signs; r = r * signs[:, None]
        #numpy.testing.assert_allclose(q @ r, lin.inner)
        if numpy.linalg.det(q) < 0:
            warn("Crystal is left-handed. This is currently unsupported, and may cause errors.")
            # currently, behavior is to leave `ortho` proper, and move the inversion into the affine transform

        cell_size, cell_angle = ortho_to_cell(lin)
        return Cell(
            affine=LinearTransform3D(q).translate(ortho.translation()),
            ortho=LinearTransform3D(r / cell_size).round_near_zero(),
            cell_size=cell_size, cell_angle=cell_angle,
            n_cells=to_vec3([1]*3 if n_cells is None else n_cells, numpy.int_),
            pbc=pbc,
        )

    def to_ortho(self) -> AffineTransform3D:
        return self.get_transform('local', 'cell_frac')

    def transform_cell(self, transform: AffineTransform3D, frame: CoordinateFrame = 'local') -> Cell:
        """
        Apply the given transform to the unit cell, and return a new `Cell`.
        The transform is applied in coordinate frame 'frame'.
        Orthogonal and affine transformations are applied to the affine matrix component,
        while skew and scaling is applied to the orthogonalization matrix/cell_size.
        """
        transform = t.cast(AffineTransform3D, self.change_transform(transform, 'local', frame))
        if not transform.to_linear().is_orthogonal():
            raise NotImplementedError()
        return Cell(
            affine=transform @ self.affine,
            ortho=self.ortho,
            cell_size=self.cell_size,
            cell_angle=self.cell_angle,
            n_cells=self.n_cells,
            pbc=self.pbc,
        )

    def strain_orthogonal(self) -> Cell:
        """
        Orthogonalize the cell using strain.

        Strain is applied such that the x-axis remains fixed, and the y-axis remains in the xy plane.
        For small displacements, no hydrostatic strain is applied (volume is conserved).
        """
        return Cell(
            affine=self.affine,
            ortho=LinearTransform3D(),
            cell_size=self.cell_size,
            n_cells=self.n_cells,
            pbc=self.pbc,
        )

    def repeat(self, n: t.Union[int, VecLike]) -> Cell:
        """Tile the cell by `n` in each dimension."""
        ns = numpy.broadcast_to(n, 3)
        if not numpy.issubdtype(ns.dtype, numpy.integer):
            raise ValueError(f"repeat() argument must be an integer or integer array.")
        return Cell(
            affine=self.affine,
            ortho=self.ortho,
            cell_size=self.cell_size,
            cell_angle=self.cell_angle,
            n_cells=self.n_cells * numpy.broadcast_to(n, 3),
            pbc = self.pbc | (ns > 1)  # assume periodic along repeated directions
        )

    def explode(self) -> Cell:
        return Cell(
            affine=self.affine,
            ortho=self.ortho,
            cell_size=self.cell_size*self.n_cells,
            cell_angle=self.cell_angle,
            pbc=self.pbc,
        )

    def crop(self, x_min: float = -numpy.inf, x_max: float = numpy.inf,
             y_min: float = -numpy.inf, y_max: float = numpy.inf,
             z_min: float = -numpy.inf, z_max: float = numpy.inf, *,
             frame: CoordinateFrame = 'local') -> Cell:
        """
        Crop 'cell' to the given extents. For a non-orthogonal
        cell, this must be specified in cell coordinates. This
        function implicity `explode`s the cell as well.
        """

        if not frame.lower().startswith('cell'):
            if not self.is_orthogonal():
                raise ValueError("Cannot crop a non-orthogonal cell in orthogonal coordinates. Use crop_atoms instead.")

        min = to_vec3([x_min, y_min, z_min])
        max = to_vec3([x_max, y_max, z_max])
        (min, max) = self.get_transform('cell_box').transform([min, max])
        new_box = BBox3D(min, max) & BBox3D.unit()
        cropped = (new_box.min > 0.) | (new_box.max < 1.)

        return Cell(
            affine=self.affine @ AffineTransform3D.translate(-new_box.min),
            ortho=self.ortho,
            cell_size=new_box.size * self.cell_size * self.n_cells,
            cell_angle=self.cell_angle,
            pbc=self.pbc & ~cropped  # remove periodicity along cropped directions
        )

    def _get_transform_to_local(self, frame: CoordinateFrame) -> AffineTransform3D:
        """Get the transform from 'frame' to local coordinates."""
        frame = t.cast(CoordinateFrame, frame.lower())

        if frame == 'local' or frame == 'global':
            return LinearTransform3D()

        if frame.startswith('cell'):
            transform = self.affine @ self.ortho
            cell_size = self.cell_size
        elif frame.startswith('ortho'):
            transform = self.affine
            cell_size = self.ortho_size
        else:
            raise ValueError(f"Unknown coordinate frame '{frame}'")

        if '_' not in frame:
            return transform
        end = frame.split('_', 2)[1]
        if end == 'frac':
            return transform @ LinearTransform3D.scale(cell_size)
        if end == 'box':
            return transform @ LinearTransform3D.scale(cell_size * self.n_cells)
        raise ValueError(f"Unknown coordinate frame '{frame}'")

    def get_transform(self, frame_to: t.Optional[CoordinateFrame] = None, frame_from: t.Optional[CoordinateFrame] = None) -> AffineTransform3D:
        """
        In the two-argument form, get the transform to 'frame_to' from 'frame_from'.
        In the one-argument form, get the transform from local coordinates to 'frame'.
        """
        transform_from = self._get_transform_to_local(frame_from) if frame_from is not None else AffineTransform3D()
        transform_to = self._get_transform_to_local(frame_to) if frame_to is not None else AffineTransform3D()
        if frame_from is not None and frame_to is not None and frame_from.lower() == frame_to.lower():
            return AffineTransform3D()
        return transform_to.inverse() @ transform_from

    def change_transform(self, transform: Transform3D,
                         frame_to: t.Optional[CoordinateFrame] = None,
                         frame_from: t.Optional[CoordinateFrame] = None) -> Transform3D:
        """Coordinate-change a transformation to 'frame_to' from 'frame_from'."""
        if frame_to == frame_from and frame_to is not None:
            return transform
        coord_change = self.get_transform(frame_to, frame_from)
        return coord_change @ transform @ coord_change.inverse()

    def assert_equal(self, other: t.Any):
        assert isinstance(other, Cell)
        numpy.testing.assert_array_almost_equal(self.affine.inner, other.affine.inner, 6)
        numpy.testing.assert_array_almost_equal(self.ortho.inner, other.ortho.inner, 6)
        numpy.testing.assert_array_almost_equal(self.cell_size, other.cell_size, 6)
        numpy.testing.assert_array_equal(self.n_cells, other.n_cells)
        numpy.testing.assert_array_equal(self.pbc, other.pbc)


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


def cell_to_ortho(cell_size: VecLike, cell_angle: t.Optional[VecLike] = None) -> LinearTransform3D:
    """
    Get orthogonalization transform from unit cell parameters (which turns fractional cell coordinates into real-space coordinates).
    ."""
    cell_size = _validate_cell_size(cell_size)
    cell_angle = _validate_cell_angle(cell_angle)
    (a, b, c) = cell_size if cell_size is not None else (1., 1., 1.)

    if numpy.allclose(cell_angle.view(numpy.ndarray), numpy.pi/2.):
        return LinearTransform3D.scale(a, b, c)

    (alpha, beta, gamma) = cell_angle
    alphastar = numpy.cos(beta) * numpy.cos(gamma) - numpy.cos(alpha)
    alphastar /= numpy.sin(beta) * numpy.sin(gamma)
    alphastar = numpy.arccos(alphastar)
    assert not numpy.isnan(alphastar)

    # aligns a axis along x
    # aligns b axis in the x-y plane
    return LinearTransform3D(numpy.array([
        [a,  b * numpy.cos(gamma),  c * numpy.cos(beta)],
        [0.,  b * numpy.sin(gamma), -c * numpy.sin(beta) * numpy.cos(alphastar)],
        [0.,  0.,                     c * numpy.sin(beta) * numpy.sin(alphastar)],
    ], dtype=float)).round_near_zero()


def ortho_to_cell(ortho: LinearTransform3D) -> t.Tuple[Vec3, Vec3]:
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


def plane_to_zone(metric: LinearTransform3D, plane: VecLike, reduce: bool = True) -> Vec3:
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


def zone_to_plane(metric: LinearTransform3D, zone: VecLike, reduce: bool = True) -> Vec3:
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
