from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import scipy.linalg

from .types import VecLike, Pts3DLike, Num, to_vec3
from .vec import perp, reduce_vec, is_diagonal
from .bbox import BBox3D
from .util import opt_classmethod


Transform3DT = t.TypeVar('Transform3DT', bound='Transform3D')
NumT = t.TypeVar('NumT', bound=t.Union[float, int])

Affine3DSelf = t.TypeVar('Affine3DSelf', bound='AffineTransform3D')
IntoTransform3D = t.Union['Transform3D', t.Callable[[NDArray[numpy.floating]], numpy.ndarray], numpy.ndarray]


class Transform3D(ABC):
    @staticmethod
    @abstractmethod
    def identity() -> Transform3D:
        """Return an identity transformation."""
        ...

    @staticmethod
    def make(data: IntoTransform3D) -> Transform3D:
        """Make a transformation from a function or numpy array."""
        if isinstance(data, Transform3D):
            return data
        if not isinstance(data, numpy.ndarray) and hasattr(data, '__call__'):
            return FuncTransform3D(data)
        data = numpy.array(data)
        if data.shape == (3, 3):
            return LinearTransform3D(data)
        if data.shape == (4, 4):
            return AffineTransform3D(data)
        raise ValueError(f"Transform3D of invalid shape {data.shape}")

    @abstractmethod
    def compose(self, other: Transform3D) -> Transform3D:
        """Compose this transformation with another."""
        ...

    @t.overload
    @abstractmethod
    def transform(self, points: BBox3D) -> BBox3D:
        ...
    
    @t.overload
    @abstractmethod
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    @abstractmethod
    def transform(self, points: Pts3DLike) -> t.Union[BBox3D, NDArray[numpy.floating]]:
        """Transform points according to the given transformation."""
        ...

    __call__ = transform

    def transform_vec(self, vecs: ArrayLike) -> NDArray[numpy.floating]:
        """Transform vector quantities. This excludes translation, as would be expected when transforming vectors."""
        a = numpy.atleast_1d(vecs)
        return self.transform(a) - self.transform(numpy.zeros_like(a))
    
    @t.overload
    def __matmul__(self, other: Transform3D) -> Transform3D:
        ...

    @t.overload
    def __matmul__(self, other: BBox3D) -> BBox3D:
        ...
    
    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[Transform3D, Pts3DLike]) -> t.Union[Transform3D, BBox3D, NDArray[numpy.floating]]:
        """Compose this transformation, or apply it to a given set of points."""
        if isinstance(other, Transform3D):
            return other.compose(self)
        return self.transform(other)

    def __rmatmul__(self, other: t.Any):
        raise ValueError("Transform must be applied to points, not the other way around.")


class FuncTransform3D(Transform3D):
    """Transformation which applies a function to the given points."""

    def __init__(self, f: t.Callable[[numpy.ndarray], numpy.ndarray]):
        self.f: t.Callable[[numpy.ndarray], numpy.ndarray] = f

    @staticmethod
    def identity() -> FuncTransform3D:
        return FuncTransform3D(lambda pts: pts)

    @t.overload
    def transform(self, points: BBox3D) -> BBox3D:
        ...
    
    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: Pts3DLike) -> t.Union[BBox3D, NDArray[numpy.floating]]:
        if isinstance(points, BBox3D):
            return points.from_pts(self.transform(points.corners()))

        return self.f(numpy.atleast_1d(points))

    def compose(self, other: Transform3D) -> FuncTransform3D:
        return FuncTransform3D(lambda pts: other.transform(self.f(pts)))

    def _rcompose(self, after: Transform3D) -> FuncTransform3D:
        return FuncTransform3D(lambda pts: self.f(after.transform(pts)))

    __call__ = transform


class AffineTransform3D(Transform3D):
    __array_ufunc__ = None

    def __init__(self, array: t.Optional[ArrayLike] = None):
        if array is None:
            array = numpy.eye(4)
        self.inner = numpy.broadcast_to(array, (4, 4))

    @property
    def __array_interface__(self):
        return self.inner.__array_interface__

    def __repr__(self) -> str:
        return f"AffineTransform3D(\n{self.inner!r}\n)"

    @staticmethod
    def identity() -> AffineTransform3D:
        return AffineTransform3D()

    def round_near_zero(self: Affine3DSelf) -> Affine3DSelf:
        """Round near-zero matrix elements in self."""
        return type(self)(
            numpy.where(numpy.abs(self.inner) < 1e-15, 0., self.inner)
        )

    @staticmethod
    def from_linear(linear: LinearTransform3D) -> AffineTransform3D:
        """Make an affine transformation from a linear transformation."""
        dtype = linear.inner.dtype
        return AffineTransform3D(numpy.block([
            [linear.inner, numpy.zeros((3, 1), dtype=dtype)],
            [numpy.zeros((1, 3), dtype=dtype), numpy.ones((), dtype=dtype)]
        ]))  # type: ignore

    def to_linear(self) -> LinearTransform3D:
        """Return the linear part of an affine transformation."""
        return LinearTransform3D(self.inner[:3, :3])

    def to_translation(self) -> AffineTransform3D:
        return AffineTransform3D.translate(self.translation())

    def det(self) -> float:
        """Return the determinant of an affine transformation."""
        return numpy.linalg.det(self.inner[:3, :3])

    def translation(self) -> numpy.ndarray:
        return self.inner[:3, -1]

    def inverse(self) -> AffineTransform3D:
        """Return the inverse of an affine transformation."""
        linear_inv = LinearTransform3D(self.inner[:3, :3]).inverse()
        # first undo translation, then undo linear transformation
        return linear_inv @ AffineTransform3D.translate(*-self.translation())

    @t.overload
    @classmethod
    def translate(cls, x: VecLike, /) -> AffineTransform3D:
        ...

    @t.overload
    @classmethod
    def translate(cls, x: Num = 0., y: Num = 0., z: Num = 0.) -> AffineTransform3D:
        ...

    @opt_classmethod
    def translate(self, x: t.Union[Num, VecLike] = 0., y: Num = 0., z: Num = 0.) -> AffineTransform3D:
        """Create or append an affine translation"""
        if isinstance(x, t.Sized) and len(x) > 1:
            try:
                (x, y, z) = to_vec3(x)
            except ValueError:
                raise ValueError("translate() must be called with a sequence or three numbers.")

        if isinstance(self, LinearTransform3D):
            self = AffineTransform3D.from_linear(self)

        a = self.inner.copy()
        a[:3, -1] += [x, y, z]
        return AffineTransform3D(a)

    @t.overload
    @classmethod
    def scale(cls, x: VecLike, /) -> AffineTransform3D:
        ...

    @t.overload
    @classmethod
    def scale(cls, x: Num = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> AffineTransform3D:
        ...

    @opt_classmethod
    def scale(self, x: t.Union[Num, VecLike] = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> AffineTransform3D:
        """Create or append a scaling transformation"""
        return self.compose(LinearTransform3D.scale(x, y, z, all=all))

    @opt_classmethod
    def rotate(self, v: VecLike, theta: Num) -> AffineTransform3D:
        """
        Create or append a rotation transformation of `theta`
        radians CCW around the given vector `v`
        """
        return self.compose(LinearTransform3D.rotate(v, theta))

    @opt_classmethod
    def rotate_euler(self, x: Num = 0., y: Num = 0., z: Num = 0.) -> AffineTransform3D:
        """
        Create or append a Euler rotation transformation.
        Rotation is performed on the x axis first, then y axis and z axis.
        Values are specified in radians.
        """
        return self.compose(LinearTransform3D.rotate_euler(x, y, z))

    @t.overload
    @classmethod
    def mirror(cls, a: VecLike, /) -> AffineTransform3D:
        ...
    
    @t.overload
    @classmethod
    def mirror(cls, a: Num, b: Num, c: Num) -> AffineTransform3D:
        ...

    @opt_classmethod
    def mirror(self, a: t.Union[Num, VecLike],
               b: t.Optional[Num] = None,
               c: t.Optional[Num] = None) -> AffineTransform3D:
        """
        Create or append a mirror transformation across the given plane.
        """
        return self.compose(LinearTransform3D.mirror(a, b, c))

    @opt_classmethod
    def strain(self, strain: float, v: VecLike = (0, 0, 1), poisson: float = 0.) -> AffineTransform3D:
        """
        Apply a strain of ``strain`` in direction ``v``, assuming an elastically isotropic material.

        Strain is applied relative to the origin.

        With ``poisson=0`` (default), a uniaxial strain is applied.
        With ``poisson=-1``, hydrostatic strain is applied.
        Otherwise, a uniaxial stress is applied, which results in shrinkage
        perpendicular to the direction strain is applied.
        """
        return self.compose(LinearTransform3D.strain(strain, v, poisson))

    @t.overload
    def transform(self, points: BBox3D) -> BBox3D:
        ...
    
    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: Pts3DLike) -> t.Union[BBox3D, NDArray[numpy.floating]]:
        if isinstance(points, BBox3D):
            return points.from_pts(self.transform(points.corners()))

        points = numpy.atleast_1d(points)
        pts = numpy.concatenate((points, numpy.broadcast_to(1., (*points.shape[:-1], 1))), axis=-1)
        # carefully handle inf and nan. this is probably slow
        isnan = numpy.bitwise_or.reduce(numpy.isnan(pts), axis=-1)
        with numpy.errstate(invalid='ignore'):
            prod = self.inner * pts[..., None, :]
        prod[numpy.isnan(prod)] = 0.
        prod[isnan, :] = numpy.nan
        return prod.sum(axis=-1)[..., :3]

    __call__ = transform

    def transform_vec(self, vecs: ArrayLike) -> NDArray[numpy.floating]:
        return self.to_linear().transform(vecs)

    @t.overload
    def compose(self, other: AffineTransform3D) -> AffineTransform3D:
        ...

    @t.overload
    def compose(self, other: Transform3DT) -> Transform3DT:
        ...

    def compose(self, other: Transform3D) -> Transform3D:
        if not isinstance(other, Transform3D):
            raise TypeError(f"Expected a Transform3D, got {type(other)}")
        if isinstance(other, LinearTransform3D):
            return self.compose(AffineTransform3D.from_linear(other))
        if isinstance(other, AffineTransform3D):
            return AffineTransform3D(other.inner @ self.inner)
        elif hasattr(other, '_rcompose'):
            return other._rcompose(self)  # type: ignore
        else:
            raise NotImplementedError()

    @t.overload
    def conjugate(self, transform: AffineTransform3D) -> AffineTransform3D:
        ...

    @t.overload
    def conjugate(self, transform: Transform3DT) -> Transform3DT:
        ...

    def conjugate(self, transform: Transform3D) -> Transform3D:
        """
        Apply ``transform`` in the coordinate frame of ``self``.

        Equivalent to an (inverse) conjugation in group theory, or :math:`T^-1 A T`
        """
        return self.inverse() @ transform @ self

    @t.overload
    def __matmul__(self, other: AffineTransform3D) -> AffineTransform3D:
        ...

    @t.overload
    def __matmul__(self, other: Transform3D) -> Transform3D:
        ...

    @t.overload
    def __matmul__(self, other: BBox3D) -> BBox3D:
        ...
    
    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[Transform3D, ArrayLike, BBox3D]):
        if isinstance(other, Transform3D):
            return other.compose(self)
        return self.transform(other)


class LinearTransform3D(AffineTransform3D):
    def __init__(self, array: t.Optional[ArrayLike] = None):
        if array is None:
            array = numpy.eye(3, dtype=numpy.float_)
        self.inner = numpy.broadcast_to(array, (3, 3))

    @property
    def T(self):
        return LinearTransform3D(self.inner.T)

    def __repr__(self) -> str:
        return f"LinearTransform3D(\n{self.inner!r}\n)"

    def translation(self):
        return numpy.zeros(3, dtype=self.inner.dtype)

    @staticmethod
    def identity() -> LinearTransform3D:
        return LinearTransform3D()

    def det(self) -> float:
        return numpy.linalg.det(self.inner)

    def inverse(self) -> LinearTransform3D:
        return LinearTransform3D(numpy.linalg.inv(self.inner))

    def to_linear(self) -> LinearTransform3D:
        return self

    def is_diagonal(self, tol: float = 1e-10) -> bool:
        d = self.inner.shape[0]
        p, q = self.inner.strides
        offdiag = numpy.lib.stride_tricks.as_strided(self.inner[:, 1:], (d-1, d), (p+q, q))
        return bool((numpy.abs(offdiag) < tol).all())

    def is_normal(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Returns `True` if `self` is a normal matrix."""
        return bool(numpy.allclose(
            self.inner.T @ self.inner, self.inner @ self.inner.T,
            rtol=rtol, atol=atol
        ))

    def is_orthogonal(self, tol: float = 1e-8) -> bool:
        """
        Returns `True` if `self` is an orthogonal matrix (i.e. a pure rotation or roto-reflection).
        """
        return numpy.allclose(self.inner @ self.inner.T, numpy.eye(3), atol=tol)

    def is_scaled_orthogonal(self, tol: float = 1e-8) -> bool:
        """
        Returns `True` if `self` is a scaled orthogonal matrix (composed of orthogonal
        basis vectors, i.e. a scaling + a rotation or roto-reflection)
        """
        return is_diagonal(self.inner @ self.inner.T, tol=tol)

    @t.overload
    @classmethod
    def mirror(cls, a: VecLike, /) -> LinearTransform3D:
        ...

    @t.overload
    @classmethod
    def mirror(cls, a: Num, b: Num, c: Num) -> LinearTransform3D:
        ...

    @opt_classmethod
    def mirror(self, a: t.Union[Num, VecLike],
               b: t.Optional[Num] = None,
               c: t.Optional[Num] = None) -> LinearTransform3D:
        if isinstance(a, t.Sized):
            v = numpy.array(numpy.broadcast_to(a, 3), dtype=numpy.float_)
            if b is not None or c is not None:
                raise ValueError("mirror() must be passed a sequence or three numbers.")
        else:
            v = numpy.array([a, b, c], dtype=numpy.float_)
        v /= numpy.linalg.norm(v)
        mirror = numpy.eye(3) - 2 * numpy.outer(v, v)
        return LinearTransform3D(mirror @ self.inner)

    @opt_classmethod
    def strain(self, strain: float, v: VecLike = (0, 0, 1), poisson: float = 0.) -> LinearTransform3D:
        """
        Apply a strain of ``strain`` in direction ``v``, assuming an elastically isotropic material.

        Strain is applied relative to the origin.

        With ``poisson=0`` (default), a uniaxial strain is applied.
        With ``poisson=-1``, hydrostatic strain is applied.
        Otherwise, a uniaxial stress is applied, which results in shrinkage
        perpendicular to the direction strain is applied.
        """
        shrink = (1 + strain) ** -poisson
        return self.compose(LinearTransform3D.align(v).conjugate(
                            LinearTransform3D.scale([shrink, shrink, 1. + strain])))

    @opt_classmethod
    def rotate(self, v: VecLike, theta: Num) -> LinearTransform3D:
        theta = float(theta)
        v = numpy.array(numpy.broadcast_to(v, (3,)), dtype=numpy.float_)
        l = numpy.linalg.norm(v)
        if numpy.isclose(l, 0.):
            if numpy.isclose(theta, 0.):
                # null rotation
                return self
            raise ValueError("rotate() about the zero vector is undefined.")
        v /= l

        # Rodrigues rotation formula
        w = numpy.array([[  0., -v[2],  v[1]],
                         [ v[2],   0., -v[0]],
                         [-v[1], v[0],   0.]], dtype=numpy.float_)
        # I + sin(t) W + (1 - cos(t)) W^2 = I + sin(t) W + 2*sin^2(t/2) W^2
        a = numpy.eye(3) + numpy.sin(theta) * w + 2 * (numpy.sin(theta / 2)**2) * w @ w
        return LinearTransform3D(a @ self.inner)

    @opt_classmethod
    def rotate_euler(self, x: Num = 0., y: Num = 0., z: Num = 0.) -> LinearTransform3D:
        angles = numpy.array([x, y, z], dtype=numpy.float_)
        c, s = numpy.cos(angles), numpy.sin(angles)
        a = numpy.array([
            [c[1]*c[2], s[0]*s[1]*c[2] - c[0]*s[2], c[0]*s[1]*c[2] + s[0]*s[2]],
            [c[1]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], c[0]*s[1]*s[2] - s[0]*c[2]],
            [-s[1],     s[0]*c[1],                  c[0]*c[1]],
        ], dtype=numpy.float_)
        return LinearTransform3D(a @ self.inner)

    @opt_classmethod
    def align(self, v1: VecLike, horz: t.Optional[VecLike] = None) -> LinearTransform3D:
        """
        Create a transformation which transforms `v1` to align with [0, 0, 1].
        If `horz` is specified, it will be aligned in the direction of [1, 0, 0].
        """
        v1 = numpy.broadcast_to(v1, 3)
        v1 = v1 / numpy.linalg.norm(v1)
        if horz is None:
            if numpy.isclose(v1[0], 1.):
                # zone is [1., 0., 0.], choose a different direction
                horz = numpy.array([0., 1., 0.])
            else:
                horz = numpy.array([1., 0., 0.])
        else:
            horz = numpy.broadcast_to(horz, 3)

        return self.align_to(v1, [0., 0., 1.], horz, [1., 0., 0.])

    @t.overload
    def align_to(self, v1: VecLike, v2: VecLike, p1: t.Literal[None] = None, p2: t.Literal[None] = None) -> LinearTransform3D:
        ...

    @t.overload
    def align_to(self, v1: VecLike, v2: VecLike, p1: VecLike, p2: VecLike) -> LinearTransform3D:
        ...

    @opt_classmethod
    def align_to(self, v1: VecLike, v2: VecLike,
                 p1: t.Optional[VecLike] = None, p2: t.Optional[VecLike] = None) -> LinearTransform3D:
        """
        Create a transformation which transforms `v1` to align with `v2`.
        If specified, additionally ensure that `p1` aligns with `p2` in the plane of `v2`.
        """
        v1 = numpy.broadcast_to(v1, 3)
        v1 = v1 / numpy.linalg.norm(v1)
        v2 = numpy.broadcast_to(v2, 3)
        v2 = v2 / numpy.linalg.norm(v2)

        v3 = numpy.cross(v1, v2)
        # rotate along v1 x v2 (geodesic rotation)
        theta = numpy.arctan2(numpy.linalg.norm(v3), numpy.dot(v1, v2))
        if numpy.isclose(numpy.linalg.norm(v3), 0.):
            # any non-v1/v2 vector works. We choose the unit vector with largest cross product
            v3 = numpy.zeros_like(v3)
            v3[numpy.argmin(numpy.abs(v1))] = 1.

        aligned = self.rotate(v3, theta)

        if p1 is None and p2 is None:
            return aligned.round_near_zero()
        if p1 is None:
            raise ValueError("If `p2` is specified, `p1` must also be specified.")
        if p2 is None:
            raise ValueError("If `p1` is specified, `p2` must also be specified.")

        p1_align = aligned.transform(numpy.broadcast_to(p1, 3))
        p2 = numpy.broadcast_to(p2, 3)
        # components perpendicular to v2
        p2_perp = perp(p2, v2)
        p1_perp = perp(p1_align, v2)
        # now rotate along v2
        theta = numpy.arctan2(numpy.dot(v2, numpy.cross(p1_perp, p2_perp)), numpy.dot(p1_perp, p2_perp))
        #theta = numpy.arctan2(numpy.linalg.norm(numpy.cross(p1_perp, p2_perp)), numpy.dot(p1_perp, p2_perp))
        return aligned.rotate(v2, theta).round_near_zero()

    def align_standard(self) -> LinearTransform3D:
        """
        Align `self` so `v1` is in the x-axis and `v2` is in the xy-plane.
        """
        assert self.det() > 0  # only works on right handed crystal systems
        _q, r = t.cast(t.Tuple[numpy.ndarray, numpy.ndarray], scipy.linalg.qr(self.inner))
        # qr unique up to the sign of the digonal
        r = r * numpy.sign(r.diagonal())
        assert numpy.linalg.det(r) > 0
        return LinearTransform3D(r).round_near_zero()

    def _orthogonal_axes(self, max_denom: int = 1000) -> NDArray[numpy.int_]:
        """
        Given a linear transformation A, compute an optimal linear
        combination of basis vectors to form an orthogonal basis.

        More formally, returns a small integer matrix M such that A@M is normal.
        """
        inv = self.inverse().inner
        r, _q = scipy.linalg.rq(inv)
        # rq unique up to the sign of the digonal
        r = r * numpy.sign(r.diagonal())

        int_r = numpy.array([reduce_vec(v, max_denom) for v in r.T]).T
        return int_r

    @t.overload
    @classmethod
    def scale(cls, x: VecLike, /) -> LinearTransform3D:
        ...

    @t.overload
    @classmethod
    def scale(cls, x: Num = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> LinearTransform3D:
        ...

    @opt_classmethod
    def scale(self, x: t.Union[Num, VecLike] = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> LinearTransform3D:
        if isinstance(x, t.Sized):
            v = numpy.broadcast_to(x, 3)
            if y != 1. or z != 1.:
                raise ValueError("scale() must be passed a sequence or three numbers.")
        else:
            v = numpy.array([x, y, z])

        a = numpy.zeros((3, 3), dtype=self.inner.dtype)
        a[numpy.diag_indices(3)] = all * v
        return LinearTransform3D(a @ self.inner)

    def conjugate(self, transform: Transform3DT) -> Transform3DT:
        """
        Apply ``transform`` in the coordinate frame of ``self``.

        Equivalent to an (inverse) conjugation in group theory, or :math:`T^-1 A T`
        """
        return self.inverse() @ self.compose(transform)

    def compose(self, other: Transform3DT) -> Transform3DT:
        if isinstance(other, LinearTransform3D):
            return other.__class__(other.inner @ self.inner)
        if isinstance(other, AffineTransform3D):
            return AffineTransform3D.from_linear(self).compose(other)
        if not isinstance(other, Transform3D):
            raise TypeError(f"Expected a Transform3D, got {type(other)}")
        elif hasattr(other, '_rcompose'):
            return other._rcompose(self)  # type: ignore
        raise NotImplementedError()

    @t.overload
    def transform(self, points: BBox3D) -> BBox3D:
        ...

    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: Pts3DLike) -> t.Union[BBox3D, NDArray[numpy.floating]]:
        if isinstance(points, BBox3D):
            return points.from_pts(self.transform(points.corners()))

        points = numpy.atleast_1d(points)
        if points.shape[-1] != 3:
            raise ValueError(f"{self.__class__} works on 3d points only.")

        # carefully handle inf and nan. this is probably slow
        isnan = numpy.bitwise_or.reduce(numpy.isnan(points), axis=-1)
        with numpy.errstate(invalid='ignore'):
            prod = self.inner * points[..., None, :]
        prod[numpy.isnan(prod)] = 0.
        prod[isnan, :] = numpy.nan
        return prod.sum(axis=-1)

    @t.overload
    def __matmul__(self, other: Transform3DT) -> Transform3DT:
        ...

    @t.overload
    def __matmul__(self, other: BBox3D) -> BBox3D:
        ...

    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[Transform3DT, ArrayLike, BBox3D]) -> t.Union[Transform3DT, NDArray[numpy.floating], BBox3D]:
        if isinstance(other, Transform3D):
            return other.compose(self)
        return self.transform(other)


__ALL__ = [
    'Transform3D', 'FuncTransform3D', 'AffineTransform3D', 'LinearTransform3D',
    'IntoTransform3D',
]
