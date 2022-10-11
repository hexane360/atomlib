from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from .types import VecLike, PtsLike, Num, to_vec3
from .bbox import BBox

TransformT = t.TypeVar('TransformT', bound='Transform')
PtsT = t.TypeVar('PtsT', bound=PtsLike)
NumT = t.TypeVar('NumT', bound=t.Union[float, int])
P = t.ParamSpec('P')
T = t.TypeVar('T')
U = t.TypeVar('U')

AffineSelf = t.TypeVar('AffineSelf', bound='AffineTransform')
IntoTransform = t.Union['Transform', t.Callable[[numpy.ndarray], numpy.ndarray], numpy.ndarray]


class opt_classmethod(classmethod, t.Generic[T, P, U]):
    """
    Method that may be called either on an instance or on the class.
    If called on the class, a default instance will be constructed.
    """

    __func__: t.Callable[t.Concatenate[T, P], U]
    def __init__(self, f: t.Callable[t.Concatenate[T, P], U]):
        super().__init__(f)

    def __get__(self, obj: t.Optional[T], ty: t.Optional[t.Type[T]] = None) -> t.Callable[P, U]:
        if obj is None:
            if ty is None:
                raise RuntimeError()
            obj = ty()
        return t.cast(
            t.Callable[P, U],
            super().__get__(obj, obj)  # type: ignore
        )


class Transform(ABC):
    @staticmethod
    @abstractmethod
    def identity() -> Transform:
        """Return an identity transformation."""
        ...

    @staticmethod
    def make(data: IntoTransform) -> Transform:
        """Make a transformation from a function or numpy array."""
        if isinstance(data, Transform):
            return data
        if not isinstance(data, numpy.ndarray) and hasattr(data, '__call__'):
            return FuncTransform(data)
        data = numpy.array(data)
        if data.shape == (3, 3):
            return LinearTransform(data)
        if data.shape == (4, 4):
            return AffineTransform(data)
        raise ValueError(f"Transform of invalid shape {data.shape}")

    @abstractmethod
    def compose(self, other: Transform) -> Transform:
        """Compose this transformation with another."""
        ...

    @t.overload
    @abstractmethod
    def transform(self, points: BBox) -> BBox:
        ...
    
    @t.overload
    @abstractmethod
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    @abstractmethod
    def transform(self, points: PtsLike) -> t.Union[BBox, NDArray[numpy.floating]]:
        """Transform points according to the given transformation."""
        ...

    __call__ = transform

    def transform_vec(self, vecs: ArrayLike) -> NDArray[numpy.floating]:
        """Transform vector quantities. This excludes translation, as would be expected when transforming vectors."""
        a = numpy.atleast_1d(vecs)
        return self.transform(a) - self.transform(numpy.zeros_like(a))
    
    @t.overload
    def __matmul__(self, other: Transform) -> Transform:
        ...

    @t.overload
    def __matmul__(self, other: BBox) -> BBox:
        ...
    
    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[Transform, PtsLike]) -> t.Union[Transform, BBox, NDArray[numpy.floating]]:
        """Compose this transformation, or apply it to a given set of points."""
        if isinstance(other, Transform):
            return other.compose(self)
        return self.transform(other)

    def __rmatmul__(self, other):
        raise ValueError("Transform must be applied to points, not the other way around.")


class FuncTransform(Transform):
    """Transformation which applies a function to the given points."""

    def __init__(self, f: t.Callable[[numpy.ndarray], numpy.ndarray]):
        self.f: t.Callable[[numpy.ndarray], numpy.ndarray] = f

    @classmethod
    def identity(cls) -> FuncTransform:
        return cls(lambda pts: pts)

    @t.overload
    def transform(self, points: BBox) -> BBox:
        ...
    
    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: PtsLike) -> t.Union[BBox, NDArray[numpy.floating]]:
        if isinstance(points, BBox):
            return points.from_pts(self.transform(points.corners()))

        return self.f(numpy.atleast_1d(points))

    def compose(self, other: Transform) -> FuncTransform:
        return FuncTransform(lambda pts: other.transform(self.f(pts)))

    def _rcompose(self, after: Transform) -> FuncTransform:
        return FuncTransform(lambda pts: self.f(after.transform(pts)))

    __call__ = transform


class AffineTransform(Transform):
    __array_ufunc__ = None

    def __init__(self, array=None):
        if array is None:
            array = numpy.eye(4)
        self.inner = numpy.broadcast_to(array, (4, 4))

    @property
    def __array_interface__(self):
        return self.inner.__array_interface__

    def __repr__(self) -> str:
        return f"AffineTransform(\n{self.inner!r}\n)"

    @classmethod
    def identity(cls: t.Type[TransformT]) -> TransformT:
        return cls()

    def round_near_zero(self: AffineSelf) -> AffineSelf:
        """Round near-zero matrix elements in self."""
        return type(self)(
            numpy.where(numpy.abs(self.inner) < 1e-15, 0., self.inner)
        )

    @staticmethod
    def from_linear(linear: LinearTransform) -> AffineTransform:
        """Make an affine transformation from a linear transformation."""
        dtype = linear.inner.dtype
        return AffineTransform(numpy.block([
            [linear.inner, numpy.zeros((3, 1), dtype=dtype)],
            [numpy.zeros((1, 3), dtype=dtype), numpy.ones((), dtype=dtype)]
        ]))  # type: ignore

    def to_linear(self) -> LinearTransform:
        """Return the linear part of an affine transformation."""
        return LinearTransform(self.inner[:3, :3])

    def det(self) -> float:
        """Return the determinant of an affine transformation."""
        return numpy.linalg.det(self.inner[:3, :3])

    def _translation(self) -> numpy.ndarray:
        return self.inner[:3, -1]

    def inverse(self) -> AffineTransform:
        """Return the inverse of an affine transformation."""
        linear_inv = LinearTransform(self.inner[:3, :3]).inverse()
        # first undo translation, then undo linear transformation
        return linear_inv @ AffineTransform.translate(*-self._translation())

    @t.overload
    @classmethod
    def translate(cls, x: VecLike, /) -> AffineTransform:
        ...

    @t.overload
    @classmethod
    def translate(cls, x: Num = 0., y: Num = 0., z: Num = 0.) -> AffineTransform:
        ...

    @opt_classmethod
    def translate(self, x: t.Union[Num, VecLike] = 0., y: Num = 0., z: Num = 0.) -> AffineTransform:
        """Create or append an affine translation"""
        if isinstance(x, t.Sized) and len(x) > 1:
            try:
                (x, y, z) = to_vec3(x)
            except ValueError:
                raise ValueError("translate() must be called with a sequence or three numbers.")

        if isinstance(self, LinearTransform):
            self = AffineTransform.from_linear(self)

        a = self.inner.copy()
        a[:3, -1] += [x, y, z]
        return AffineTransform(a)

    @t.overload
    @classmethod
    def scale(cls, x: VecLike, /) -> AffineTransform:
        ...

    @t.overload
    @classmethod
    def scale(cls, x: Num = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> AffineTransform:
        ...

    @opt_classmethod
    def scale(self, x: t.Union[Num, VecLike] = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> AffineTransform:
        """Create or append a scaling transformation"""
        return self.compose(LinearTransform.scale(x, y, z, all=all))  # type: ignore

    @opt_classmethod
    def rotate(self, v: VecLike, theta: Num) -> AffineTransform:
        """
        Create or append a rotation transformation of `theta`
        radians CCW around the given vector `v`
        """
        return self.compose(LinearTransform.rotate(v, theta))

    @opt_classmethod
    def rotate_euler(cls, x: Num = 0., y: Num = 0., z: Num = 0.) -> AffineTransform:
        """
        Create or append a Euler rotation transformation.
        Rotation is performed on the x axis first, then y axis and z axis.
        Values are specified in radians.
        """
        self = cls() if isinstance(cls, type) else cls
        return self.compose(LinearTransform.rotate_euler(x, y, z))

    @t.overload
    @classmethod
    def mirror(cls, a: VecLike, /) -> AffineTransform:
        ...
    
    @t.overload
    @classmethod
    def mirror(cls, a: Num, b: Num, c: Num) -> AffineTransform:
        ...

    @opt_classmethod
    def mirror(self, a: t.Union[Num, VecLike],
               b: t.Optional[Num] = None,
               c: t.Optional[Num] = None) -> AffineTransform:
        """
        Create or append a mirror transformation across the given plane.
        """
        return self.compose(LinearTransform.mirror(a, b, c))  # type: ignore

    @t.overload
    def transform(self, points: BBox) -> BBox:
        ...
    
    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: PtsLike) -> t.Union[BBox, NDArray[numpy.floating]]:
        if isinstance(points, BBox):
            return points.from_pts(self.transform(points.corners()))

        points = numpy.atleast_1d(points)
        pts = numpy.concatenate((points, numpy.broadcast_to(1., (*points.shape[:-1], 1))), axis=-1)
        return (self.inner @ pts.T)[:3].T

    __call__ = transform

    def transform_vec(self, vecs: ArrayLike) -> NDArray[numpy.floating]:
        return self.to_linear().transform(vecs)

    @t.overload
    def compose(self, other: AffineTransform) -> AffineTransform:
        ...

    @t.overload
    def compose(self, other: Transform) -> Transform:
        ...

    def compose(self, other: Transform) -> Transform:
        if not isinstance(other, Transform):
            raise TypeError(f"Expected a Transform, got {type(other)}")
        if isinstance(other, LinearTransform):
            return self.compose(AffineTransform.from_linear(other))
        if isinstance(other, AffineTransform):
            return AffineTransform(other.inner @ self.inner)
        elif hasattr(other, '_rcompose'):
            return other._rcompose(self)  # type: ignore
        else:
            raise NotImplementedError()

    @t.overload
    def __matmul__(self, other: AffineTransform) -> AffineTransform:
        ...

    @t.overload
    def __matmul__(self, other: Transform) -> Transform:
        ...

    @t.overload
    def __matmul__(self, other: BBox) -> BBox:
        ...
    
    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[Transform, ArrayLike, BBox]):
        if isinstance(other, Transform):
            return other.compose(self)
        return self.transform(other)


class LinearTransform(AffineTransform):
    def __init__(self, array=None):
        if array is None:
            array = numpy.eye(3)
        self.inner = numpy.broadcast_to(array, (3, 3))

    @property
    def T(self):
        return LinearTransform(self.inner.T)

    def __repr__(self) -> str:
        return f"LinearTransform(\n{self.inner!r}\n)"

    def _translation(self):
        # not defined for LinearTransform
        raise NotImplementedError()

    @staticmethod
    def identity() -> LinearTransform:
        return LinearTransform()

    def det(self) -> float:
        return numpy.linalg.det(self.inner)

    def inverse(self) -> LinearTransform:
        return LinearTransform(numpy.linalg.inv(self.inner))

    def to_linear(self) -> LinearTransform:
        return self

    def is_orthogonal(self, tol: float = 1e-10) -> bool:
        d = self.inner.shape[0]
        p, q = self.inner.strides
        offdiag = numpy.lib.stride_tricks.as_strided(self.inner[:, 1:], (d-1, d), (p+q, q))
        return bool((numpy.abs(offdiag) < tol).all())
    
    @t.overload
    @classmethod
    def mirror(cls, a: VecLike, /) -> LinearTransform:
        ...
    
    @t.overload
    @classmethod
    def mirror(cls, a: Num, b: Num, c: Num) -> LinearTransform:
        ...

    @opt_classmethod
    def mirror(self, a: t.Union[Num, VecLike],
               b: t.Optional[Num] = None,
               c: t.Optional[Num] = None) -> LinearTransform:
        if isinstance(a, t.Sized):
            v = numpy.array(numpy.broadcast_to(a, 3), dtype=float)
            if b is not None or c is not None:
                raise ValueError("mirror() must be passed a sequence or three numbers.")
        else:
            v = numpy.array([a, b, c], dtype=float)
        v /= numpy.linalg.norm(v)
        mirror = numpy.eye(3) - 2 * numpy.outer(v, v)
        return LinearTransform(mirror @ self.inner)

    @opt_classmethod
    def rotate(self, v: VecLike, theta: Num) -> LinearTransform:
        theta = float(theta)
        v = numpy.array(numpy.broadcast_to(v, (3,)), dtype=float)
        l = numpy.linalg.norm(v)
        if numpy.isclose(l, 0.):
            if numpy.isclose(theta, 0.):
                # null rotation
                return self
            raise ValueError("rotate() about the zero vector is undefined.")
        v /= l

        # Rodrigues rotation formula
        w = numpy.array([[   0, -v[2],  v[1]],
                         [ v[2],    0, -v[0]],
                         [-v[1], v[0],    0]])
        # I + sin(t) W + (1 - cos(t)) W^2 = I + sin(t) W + 2*sin^2(t/2) W^2
        a = numpy.eye(3) + numpy.sin(theta) * w + 2 * (numpy.sin(theta / 2)**2) * w @ w
        return LinearTransform(a @ self.inner)

    @opt_classmethod
    def rotate_euler(self, x: Num = 0., y: Num = 0., z: Num = 0.) -> LinearTransform:
        angles = numpy.array([x, y, z], dtype=float)
        c, s = numpy.cos(angles), numpy.sin(angles)
        a = numpy.array([
            [c[1]*c[2], s[0]*s[1]*c[2] - c[0]*s[2], c[0]*s[1]*c[2] + s[0]*s[2]],
            [c[1]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], c[0]*s[1]*s[2] - s[0]*c[2]],
            [-s[1],     s[0]*c[1],                  c[0]*c[1]],
        ])
        return LinearTransform(a @ self.inner)

    @opt_classmethod
    def align(self, v1: VecLike, horz: t.Optional[VecLike] = None) -> LinearTransform:
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
    def align_to(self, v1: VecLike, v2: VecLike, p1: t.Literal[None] = None, p2: t.Literal[None] = None) -> LinearTransform:
        ...

    @t.overload
    def align_to(self, v1: VecLike, v2: VecLike, p1: VecLike, p2: VecLike) -> LinearTransform:
        ...

    @opt_classmethod
    def align_to(self, v1: VecLike, v2: VecLike,
                 p1: t.Optional[VecLike] = None, p2: t.Optional[VecLike] = None) -> LinearTransform:
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
        p2_perp = p2 - v2 * numpy.dot(p2, v2)
        p1_perp = p1_align - v2 * numpy.dot(p1_align, v2)
        # now rotate along v2
        theta = numpy.arctan2(numpy.dot(v2, numpy.cross(p1_perp, p2_perp)), numpy.dot(p1_perp, p2_perp))
        #theta = numpy.arctan2(numpy.linalg.norm(numpy.cross(p1_perp, p2_perp)), numpy.dot(p1_perp, p2_perp))
        return aligned.rotate(v2, theta).round_near_zero()

    @t.overload
    @classmethod
    def scale(cls, x: VecLike, /) -> LinearTransform:
        ...

    @t.overload
    @classmethod
    def scale(cls, x: Num = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> LinearTransform:
        ...

    @opt_classmethod
    def scale(self, x: t.Union[Num, VecLike] = 1., y: Num = 1., z: Num = 1., *,
              all: Num = 1.) -> LinearTransform:
        if isinstance(x, t.Sized):
            v = numpy.broadcast_to(x, 3)
            if y != 1. or z != 1.:
                raise ValueError("scale() must be passed a sequence or three numbers.")
        else:
            v = numpy.array([x, y, z])

        a = numpy.zeros((3, 3))
        a[numpy.diag_indices(3)] = all * v
        return LinearTransform(a @ self.inner)
    
    def compose(self, other: TransformT) -> TransformT:
        if isinstance(other, LinearTransform):
            return other.__class__(other.inner @ self.inner)
        if isinstance(other, AffineTransform):
            return AffineTransform.from_linear(self).compose(other)
        if not isinstance(other, Transform):
            raise TypeError(f"Expected a Transform, got {type(other)}")
        elif hasattr(other, '_rcompose'):
            return other._rcompose(self)  # type: ignore
        raise NotImplementedError()

    @t.overload
    def transform(self, points: BBox) -> BBox:
        ...
    
    @t.overload
    def transform(self, points: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def transform(self, points: PtsLike) -> t.Union[BBox, NDArray[numpy.floating]]:
        if isinstance(points, BBox):
            return points.from_pts(self.transform(points.corners()))

        points = numpy.atleast_1d(points)
        if points.shape[-1] != 3:
            raise ValueError(f"{self.__class__} works on 3d points only.")

        return (self.inner @ points.T).T

    @t.overload
    def __matmul__(self, other: TransformT) -> TransformT:
        ...

    @t.overload
    def __matmul__(self, other: BBox) -> BBox:
        ...

    @t.overload
    def __matmul__(self, other: ArrayLike) -> NDArray[numpy.floating]:
        ...

    def __matmul__(self, other: t.Union[TransformT, ArrayLike, BBox]) -> t.Union[TransformT, NDArray[numpy.floating], BBox]:
        if isinstance(other, Transform):
            return other.compose(self)
        return self.transform(other)


__ALL__ = [
    'Transform', 'FuncTransform', 'AffineTransform', 'LinearTransform',
]
