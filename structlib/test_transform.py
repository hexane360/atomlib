import typing as t

import numpy
import pytest
from numpy.testing import assert_allclose

from .transform import LinearTransform, AffineTransform


def test_linear_transform_constructors():
    identity = numpy.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])

    t = LinearTransform()
    assert_allclose(t.inner, identity)

    t = LinearTransform.identity()
    assert_allclose(t.inner, identity)

    a = numpy.linspace(1.5, 20., 9).reshape((3, 3))
    t = LinearTransform(a)
    assert_allclose(t.inner, a)


def test_affine_transform_constructors():
    identity = numpy.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

    t = AffineTransform()
    assert_allclose(t.inner, identity)

    t = AffineTransform.identity()
    assert_allclose(t.inner, identity)

    a = numpy.linspace(1.5, 20., 16).reshape((4, 4))
    t = AffineTransform(a)
    assert_allclose(t.inner, a)


def test_affine_linear_conversion():
    a = numpy.linspace(1.5, 20., 9).reshape((3, 3))
    t = AffineTransform.from_linear(LinearTransform(a))

    assert_allclose(t.inner[:3, :3], a)
    assert_allclose(t.to_linear().inner, a)


def test_affine_transform_compose():
    pts = numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])

    expected = numpy.array([[1., 0., 0.], [3., 0., 0.], [1., 2., 0.], [3., 2., 0.]])

    assert_allclose(LinearTransform.scale(all=2).translate([1., 0., 0.]) @ pts, expected)
    assert_allclose(AffineTransform.scale(all=2).translate([1., 0., 0.]) @ pts, expected)
    assert_allclose((AffineTransform.translate([1., 0., 0.]) @ AffineTransform.scale(all=2)) @ pts, expected)
    assert_allclose((LinearTransform.translate([1., 0., 0.]) @ AffineTransform.scale(all=2)) @ pts, expected)

    t = LinearTransform.scale(all=2).translate([1., 0., 0.])

    assert_allclose((t @ t.inverse()).inner, numpy.eye(4))
    assert_allclose((t.inverse() @ t).inner, numpy.eye(4))


@pytest.mark.parametrize('transform', (LinearTransform, AffineTransform, LinearTransform(), AffineTransform()))
def test_transform_ops(transform: t.Union[AffineTransform, t.Type[AffineTransform]]):
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

    t = LinearTransform().scale(2., 2., 1.)
    t2 = LinearTransform().scale(all=2.)

    expected = numpy.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [2., 2., 0.]])
    assert_allclose(t.transform(pts), expected)
    assert_allclose(t2.transform(pts), expected)
    assert_allclose(t @ pts, expected)
    assert_allclose(t2 @ pts, expected)

    t = AffineTransform().translate(1., 0., 0.)
    expected = numpy.array([[1., 0., 0.], [2., 0., 0.], [1., 1., 0.], [2., 1., 0.]])
    assert_allclose(t @ pts, expected)

    with pytest.raises(ValueError, match="Transform must be applied to points"):
        assert pts @ t


def test_transform_compose():
    t1 = LinearTransform().scale(2., 1., 1.)
    t2 = LinearTransform().rotate([0., 0., 1.], numpy.pi/2)

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

    affine = AffineTransform().translate([1., 0., 0.])

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