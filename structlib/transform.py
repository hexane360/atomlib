from __future__ import annotations

from abc import ABC, abstractmethod
import typing as t

import numpy

from .vec import Vec3, BBox


class Transform(ABC):
	@abstractmethod
	def transform(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		...

	@staticmethod
	@abstractmethod
	def identity() -> Transform:
		...

	@abstractmethod
	def compose(self, other: Transform) -> Transform:
		...

	def __call__(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		return self.transform(points)
	
	@t.overload
	def __matmul__(self, other: Transform) -> Transform:
		...
	
	@t.overload
	def __matmul__(self, other: numpy.ndarray) -> numpy.ndarray:
		...

	@t.overload
	def __matmul__(self, other: Vec3) -> Vec3:
		...

	def __matmul__(self, other: t.Union[Transform, numpy.ndarray]) -> t.Union[Transform, numpy.ndarray]:
		if isinstance(other, Transform):
			return other.compose(self)
		return self.transform(other)

	def __rmatmul__(self, other):
		raise ValueError("Transform() must be applied to points, not the other way around.")


class LinearTransform(Transform):
	def __init__(self, array=None):
		if array is None:
			array = numpy.eye(3)
		self.inner = numpy.broadcast_to(array, (3, 3))

	@property
	def T(self):
		return LinearTransform(self.inner.T)

	@property
	def __array_interface__(self):
		return self.inner.__array_interface__

	def __repr__(self) -> str:
		return f"LinearTransform(\n{self.inner!r}\n)"

	@staticmethod
	def identity() -> LinearTransform:
		return LinearTransform()

	def det(self) -> float:
		return numpy.linalg.det(self.inner)

	def inverse(self) -> LinearTransform:
		return LinearTransform(numpy.linalg.inv(self.inner))
	
	@t.overload
	def mirror(self, a: t.Sequence[float]) -> LinearTransform:
		...
	
	@t.overload
	def mirror(self, a: float, b: float, c: float) -> LinearTransform:
		...

	def mirror(self, a: t.Union[float, t.Sequence[float]],
	           b: t.Optional[float] = None,
	           c: t.Optional[float] = None) -> LinearTransform:
		if c is None:
			if b is not None:
				raise ValueError("mirror() must be passed a sequence or three floats.")
			(a, b, c) = a  # type: ignore
		v = numpy.array([a, b, c], dtype=float)
		v /= numpy.linalg.norm(v)
		mirror = numpy.eye(3) - 2 * numpy.outer(v, v)
		return LinearTransform(mirror @ self.inner)

	def rotate(self, v, theta: float) -> LinearTransform:
		v = numpy.array(numpy.broadcast_to(v, (3,)), dtype=float)
		v /= numpy.linalg.norm(v)

		# Rodrigues rotation formula
		w = numpy.array([[   0, -v[2],  v[1]],
		                 [ v[2],    0, -v[0]],
		                 [-v[1], v[0],    0]])
		# I + sin(t) W + (1 - cos(t)) W^2 = I + sin(t) W + 2*sin^2(t/2) W^2
		a = numpy.eye(3) + numpy.sin(theta) * w + 2 * (numpy.sin(theta / 2)**2) * w @ w
		return LinearTransform(a @ self.inner)

	def rotate_euler(self, x: float = 0., y: float = 0., z: float = 0.) -> LinearTransform:
		"""
		Rotate by the given Euler angles (in radians). Rotation is performed on the x axis
		first, then y axis and z axis.
		"""

		angles = numpy.array([x, y, z])
		c, s = numpy.cos(angles), numpy.sin(angles)
		a = numpy.array([
			[c[1]*c[2], s[0]*s[1]*c[2] - c[0]*s[2], c[0]*s[1]*c[2] + s[0]*s[2]],
			[c[1]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], c[0]*s[1]*s[2] - s[0]*c[2]],
			[-s[1],     s[0]*c[1],                  c[0]*c[1]],
		])
		return LinearTransform(a @ self.inner)

	def scale(self, x: float = 1., y: float = 1., z: float = 1.) -> LinearTransform:
		a = numpy.zeros((3, 3))
		a[numpy.diag_indices(3)] = [x, y, z]
		return LinearTransform(a @ self.inner)
	
	@t.overload
	def compose(self, other: LinearTransform) -> LinearTransform:
		...
	
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
			return LinearTransform(other.inner @ self.inner)
		elif isinstance(other, AffineTransform):
			return AffineTransform.from_linear(self).compose(other)
		else:
			raise NotImplementedError()

	@t.overload
	def transform(self, points: Vec3) -> Vec3:
		...

	@t.overload
	def transform(self, points: BBox) -> BBox:
		...
	
	@t.overload
	def transform(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		...

	def transform(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3], Vec3, BBox]):
		if isinstance(points, BBox):
			return points.from_pts(self.transform(points.corners()))

		result = (self.inner @ numpy.atleast_1d(points).T).T
		return result.view(Vec3) if isinstance(points, Vec3) else result

	@t.overload
	def __matmul__(self, other: LinearTransform) -> LinearTransform:
		...

	@t.overload
	def __matmul__(self, other: AffineTransform) -> AffineTransform:
		...

	@t.overload
	def __matmul__(self, other: Transform) -> Transform:
		...

	@t.overload
	def __matmul__(self, other: Vec3) -> Vec3:
		...

	@t.overload
	def __matmul__(self, other: BBox) -> BBox:
		...

	@t.overload
	def __matmul__(self, other: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		...

	def __matmul__(self, other: t.Union[Transform, numpy.ndarray, t.Sequence[Vec3], Vec3, BBox]) -> t.Union[Transform, numpy.ndarray, Vec3, BBox]:
		if isinstance(other, Transform):
			return self.compose(other)
		return self.transform(other)


class AffineTransform(Transform):
	def __init__(self, array=None):
		if array is None:
			array = numpy.eye(4)
		self.inner = numpy.broadcast_to(array, (4, 4))

	@property
	def __array_interface__(self):
		return self.inner.__array_interface__

	def __repr__(self) -> str:
		return f"AffineTransform(\n{self.inner!r}\n)"

	@staticmethod
	def identity() -> AffineTransform:
		return AffineTransform()

	@staticmethod
	def from_linear(linear: LinearTransform) -> AffineTransform:
		return AffineTransform(numpy.block([
			[linear.inner, numpy.zeros((3, 1))],
			[numpy.zeros((1, 3)), 1]
		]))  # type: ignore

	def det(self) -> float:
		return numpy.linalg.det(self.inner[:3, :3])

	def _translation(self) -> numpy.ndarray:
		return self.inner[:3, -1]

	def inverse(self) -> AffineTransform:
		linear_inv = LinearTransform(self.inner[:3, :3]).inverse()
		return linear_inv.compose(AffineTransform().translate(*(linear_inv @ -self._translation())))

	@t.overload
	def translate(self, x: t.Sequence[float]) -> AffineTransform:
		...

	@t.overload
	def translate(self, x: float = 0., y: float = 0., z: float = 0.) -> AffineTransform:
		...

	def translate(self, x: t.Union[float, t.Sequence[float]] = 0., y: float = 0., z: float = 0.) -> AffineTransform:
		if isinstance(x, t.Sized) and len(x) > 1:
			if not y == 0. or not z == 0.:
				raise ValueError("translate() must be called with a sequence or three floats.")
			(x, y, z) = map(float, x)

		a = self.inner.copy()
		a[:, -1] += [x, y, z, 0]
		return AffineTransform(a)

	def scale(self, x: float = 1., y: float = 1., z: float = 1.) -> AffineTransform:
		return self.compose(LinearTransform().scale(x, y, z))

	def rotate(self, v, theta: float) -> AffineTransform:
		return self.compose(LinearTransform().rotate(v, theta))

	def rotate_euler(self, x: float = 0., y: float = 0., z: float = 0.) -> AffineTransform:
		return self.compose(LinearTransform().rotate_euler(x, y, z))

	@t.overload
	def mirror(self, a: t.Sequence[float]) -> AffineTransform:
		...
	
	@t.overload
	def mirror(self, a: float, b: float, c: float) -> AffineTransform:
		...

	def mirror(self, a: t.Union[float, t.Sequence[float]],
	           b: t.Optional[float] = None,
	           c: t.Optional[float] = None) -> AffineTransform:
		return self.compose(LinearTransform().mirror(a, b, c))  # type: ignore

	@t.overload
	def transform(self, points: Vec3) -> Vec3:
		...

	@t.overload
	def transform(self, points: BBox) -> BBox:
		...
	
	@t.overload
	def transform(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		...

	def transform(self, points: t.Union[numpy.ndarray, t.Sequence[Vec3], Vec3, BBox]):
		if isinstance(points, BBox):
			return points.from_pts(self.transform(points.corners()))

		points = numpy.atleast_1d(points)
		pts = numpy.concatenate((points, numpy.broadcast_to(1., (*points.shape[:-1], 1))), axis=-1)
		result = (self.inner @ pts.T)[:3].T
		return result.view(Vec3) if isinstance(points, Vec3) else result

	@t.overload
	def compose(self, other: t.Union[LinearTransform, AffineTransform]) -> AffineTransform:
		...

	@t.overload
	def compose(self, other: Transform) -> Transform:
		...

	def compose(self, other: Transform) -> Transform:
		if not isinstance(other, Transform):
			raise TypeError(f"Expected a Transform, got {type(other)}")
		if isinstance(other, AffineTransform):
			return AffineTransform(other.inner @ self.inner)
		elif isinstance(other, LinearTransform):
			return self.compose(AffineTransform.from_linear(other))
		else:
			raise NotImplementedError()

	@t.overload
	def __matmul__(self, other: t.Union[LinearTransform, AffineTransform]) -> AffineTransform:
		...

	@t.overload
	def __matmul__(self, other: Transform) -> Transform:
		...

	@t.overload
	def __matmul__(self, other: numpy.ndarray) -> numpy.ndarray:
		...

	@t.overload
	def __matmul__(self, other: Vec3) -> Vec3:
		...

	@t.overload
	def __matmul__(self, other: BBox) -> BBox:
		...

	@t.overload
	def __matmul__(self, other: t.Union[numpy.ndarray, t.Sequence[Vec3]]) -> numpy.ndarray:
		...

	def __matmul__(self, other: t.Union[Transform, numpy.ndarray, t.Sequence[Vec3], Vec3, BBox]):
		if isinstance(other, Transform):
			return self.compose(other)
		return self.transform(other)
