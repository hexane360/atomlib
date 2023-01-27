import typing as t

import numpy
import pytest
from numpy.testing import assert_allclose

from .transform import LinearTransform3D, AffineTransform3D


def test_linear_transform_constructors():
    identity = numpy.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])

    t = LinearTransform3D()
    assert_allclose(t.inner, identity)

    t = LinearTransform3D.identity()
    assert_allclose(t.inner, identity)

    a = numpy.linspace(1.5, 20., 9).reshape((3, 3))
    t = LinearTransform3D(a)
    assert_allclose(t.inner, a)


def test_affine_transform_constructors():
    identity = numpy.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

    t = AffineTransform3D()
    assert_allclose(t.inner, identity)

    t = AffineTransform3D.identity()
    assert_allclose(t.inner, identity)

    a = numpy.linspace(1.5, 20., 16).reshape((4, 4))
    t = AffineTransform3D(a)
    assert_allclose(t.inner, a)


def test_affine_linear_conversion():
    a = numpy.linspace(1.5, 20., 9).reshape((3, 3))
    t = AffineTransform3D.from_linear(LinearTransform3D(a))

    assert_allclose(t.inner[:3, :3], a)
    assert_allclose(t.to_linear().inner, a)


def test_affine_transform_compose():
    pts = numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])

    expected = numpy.array([[1., 0., 0.], [3., 0., 0.], [1., 2., 0.], [3., 2., 0.]])

    assert_allclose(LinearTransform3D.scale(all=2).translate([1., 0., 0.]) @ pts, expected)
    assert_allclose(AffineTransform3D.scale(all=2).translate([1., 0., 0.]) @ pts, expected)
    assert_allclose((AffineTransform3D.translate([1., 0., 0.]) @ AffineTransform3D.scale(all=2)) @ pts, expected)
    assert_allclose((LinearTransform3D.translate([1., 0., 0.]) @ AffineTransform3D.scale(all=2)) @ pts, expected)

    t = LinearTransform3D.scale(all=2).translate([1., 0., 0.])

    assert_allclose((t @ t.inverse()).inner, numpy.eye(4))
    assert_allclose((t.inverse() @ t).inner, numpy.eye(4))


@pytest.mark.parametrize('transform', (LinearTransform3D, AffineTransform3D, LinearTransform3D(), AffineTransform3D()))
def test_transform_ops(transform: t.Union[AffineTransform3D, t.Type[AffineTransform3D]]):
    # rotate
    assert_allclose(transform.rotate([0., 0., 1.], numpy.pi/4).to_linear().inner, numpy.array([
        [1./numpy.sqrt(2), -1./numpy.sqrt(2), 0.],
        [1./numpy.sqrt(2),  1./numpy.sqrt(2), 0.],
        [0.,  0., 1.]
    ]), atol=1e-12)

    # euler rotation 
    assert_allclose(
        transform.rotate_euler(x=numpy.pi/6, y=numpy.pi, z=numpy.pi/3).to_linear().inner,
        numpy.array([
            [-0.5000000, -0.7500000,  0.4330127],
            [-0.8660254,  0.4330127, -0.2500000],
            [ 0.0000000, -0.5000000, -0.8660254],
        ]), atol=1e-12
    )

    # scale
    assert_allclose(transform.scale(2.0, 1.5).to_linear().inner, numpy.array([
        [2.,  0., 0.],
        [0., 1.5, 0.],
        [0.,  0., 1.]
    ]), atol=1e-12)

    # uniform scale
    assert_allclose(transform.scale(all=4.).to_linear().inner, numpy.array([
        [4., 0., 0.],
        [0., 4., 0.],
        [0., 0., 4.],
    ]), atol=1e-12)

    # translation
    assert_allclose(transform.translate([4., 2., -1.]).inner, numpy.array([
        [1., 0., 0.,  4.],
        [0., 1., 0.,  2.],
        [0., 0., 1., -1.],
        [0., 0., 0.,  1.],
    ]))

    # chaining
    assert_allclose(transform.scale(2.0, 1.5).rotate([0, 0, 1], numpy.pi/2)
                    .to_linear().inner, numpy.array([
        [0.0, -1.5, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]), atol=1e-12)


def test_transform_apply():
    pts = numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])

    t = LinearTransform3D().scale(2., 2., 1.)
    t2 = LinearTransform3D().scale(all=2.)

    expected = numpy.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.]])
    assert_allclose(t.transform(pts), expected)
    assert_allclose(t2.transform(pts), expected)
    assert_allclose(t @ pts, expected)
    assert_allclose(t2 @ pts, expected)

    t = AffineTransform3D().translate(1., 0., 0.)
    expected = numpy.array([[1., 0., 0.], [2., 0., 0.], [1., 1., 0.], [2., 1., 0.]])
    assert_allclose(t @ pts, expected)

    with pytest.raises(ValueError, match="Transform must be applied to points"):
        assert pts @ t


def test_transform_compose():
    t1 = LinearTransform3D().scale(2., 1., 1.)
    t2 = LinearTransform3D().rotate([0., 0., 1.], numpy.pi/2)

    # scale and then rotate
    expected = numpy.array([
        [0., -1., 0.],
        [2., 0., 0.],
        [0., 0., 1.],
    ])
    assert_allclose(t1.compose(t2).inner, expected, atol=1e-12)
    assert_allclose((t2 @ t1).inner, expected, atol=1e-12)
    assert_allclose(t1.compose(t2) @ [1, -1, 0.], [1., 2., 0.], atol=1e-12)

    # rotate and then scale
    expected = numpy.array([
        [0., -2., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
    ])
    assert_allclose(t2.compose(t1).inner, expected, atol=1e-12)
    assert_allclose((t1 @ t2).inner, expected, atol=1e-12)
    assert_allclose(t2.compose(t1) @ [1, -1, 0.], [2., 1., 0.], atol=1e-12)

    affine = AffineTransform3D().translate([1., 0., 0.])

    # scale then translate
    expected = numpy.array([
        [2., 0., 0., 1.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])
    assert_allclose((affine @ t1).inner, expected, atol=1e-12)
    assert_allclose(t1.compose(affine).inner, expected, atol=1e-12)

    # translate then scale
    expected = numpy.array([
        [2., 0., 0., 2.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

    assert_allclose((t1 @ affine).inner, expected, atol=1e-12)
    assert_allclose(affine.compose(t1).inner, expected, atol=1e-12)


@pytest.mark.parametrize(('v1', 'v2', 'p1', 'p2'), [
    ((1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 0)),
    ((1, 1, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, 1), (1, 1, 1), None, None),
])
def test_transform_align_to(v1, v2, p1, p2):
    _test_transform_align_to(v1, v2, p1, p2)


def test_transform_align_to_rand():
    rng = numpy.random.default_rng(1024)
    for _ in range(30):
        (v1, v2, p1, p2) = rng.standard_normal((4, 3))
        _test_transform_align_to(v1, v2, p1, p2)


def _test_transform_align_to(v1, v2, p1, p2):
    print(f"Aligning {v1} to {v2}, {p1} to {p2}")
    transform = LinearTransform3D.align_to(v1, v2, p1, p2)

    v1_t = transform @ (v1 / numpy.linalg.norm(v1))
    assert numpy.linalg.norm(v1_t) == pytest.approx(1.)
    v2 = v2 / numpy.linalg.norm(v2)
    assert_allclose(v1_t, v2, atol=1e-10)

    if p1 is None or p2 is None:
        return

    p1_t = transform @ (p1 / numpy.linalg.norm(p1))
    assert numpy.linalg.norm(p1_t) == pytest.approx(1.)
    p2 = p2 / numpy.linalg.norm(p2)

    p1_perp = p1_t - v2 * numpy.dot(p1_t, v2)
    p2_perp = p2 - v2 * numpy.dot(p2, v2)
    assert_allclose(p1_perp / numpy.linalg.norm(p1_perp), p2_perp / numpy.linalg.norm(p2_perp), atol=1e-10)