
import warnings
import numpy
import numpy.testing
import pytest
from matplotlib import pyplot

from .vec import reduce_vec, miller_3_to_4_plane, miller_3_to_4_vec, miller_4_to_3_plane, miller_4_to_3_vec
from .vec import polygon_winding, polygon_solid_angle, in_polygon, split_arr


square = [[-1, -1.], [1., -1.], [1., 1.], [-1., 1.]]
hourglass = [[-1, -1.], [1., -1.], [-1., 1.], [1., 1.]]


@pytest.mark.parametrize(['input', 'output'], [
    ([1/3, 1/3, 2/3], [1, 1, 2]),
    ([1/12, 0, 0], [1, 0, 0]),
    ([0.1, 0.9, 0.], [1, 9, 0]),
    ([[1/3, 1/3, 1/2], [1/9, 2/9, 3/9]], [[2, 2, 3], [1, 2, 3]]),
])
def test_reduce_vec(input, output):
    input = numpy.asanyarray(input)
    output = numpy.asanyarray(output)

    print(input.shape)
    result = reduce_vec(input)

    assert numpy.issubdtype(result.dtype, numpy.integer)
    numpy.testing.assert_array_equal(result, output)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in (true_)?divide')
        factors = input / result

    (factors, expected) = map(numpy.array, numpy.broadcast_arrays(factors.T, factors.T[0]))
    expected[numpy.isnan(factors)] = factors[numpy.isnan(factors)]

    assert numpy.allclose(factors.T, expected.T, equal_nan=True)


@pytest.mark.parametrize(['input', 'output'], [
    ([1, 0, 0], [1, 0, -1, 0]),
    ([0, 1, 0], [0, 1, -1, 0]),
    ([0, 0, 1], [0, 0, 0, 1]),
    ([1, 1, 0], [1, 1, -2, 0]),
    ([[1, 0, 1], [0, 1, -1]], [[1, 0, -1, 1], [0, 1, -1, -1]])
])
def test_miller_plane(input, output):
    assert numpy.array_equal(miller_3_to_4_plane(input), output)
    assert numpy.array_equal(miller_4_to_3_plane(output), input)


@pytest.mark.parametrize(['input', 'output'], [
    ([1, 0, 0], [2, -1, -1, 0]),
    ([0, 1, 0], [-1, 2, -1, 0]),
    ([0, 0, 1], [0, 0, 0, 1]),
    ([1, 1, 0], [1, 1, -2, 0]),
])
def test_miller_vec(input, output):
    assert numpy.array_equal(miller_3_to_4_vec(input), output)
    assert numpy.array_equal(miller_4_to_3_vec(output), input)
    assert numpy.allclose(0., numpy.sum(miller_3_to_4_vec(input)[..., :3], axis=-1))


@pytest.mark.parametrize(['poly', 'pts', 'windings'], [
    (square,
     [[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.], [-1.5, 0.]], [1, 0, 1, 0, 1, 0]),
    (hourglass,
     [[0., 0.1], [0.2, 0.], [-0.2, 0.], [0.9, 0.95], [0.9, -0.95]], [-1, 0, 0, -1, 1]),
    ([square[0], *square], # square w/ duplicate point
     [[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.], [-1.5, 0.]], [1, 0, 1, 0, 1, 0]),
])
def test_polygon_winding(poly, pts, windings):
    assert numpy.array_equal(polygon_winding(poly, pts), windings)


@pytest.mark.parametrize(['poly', 'turning'], [
    (square, 1), # square
    (hourglass, 0),
    ([square[0], *square], 1), # square w/ duplicate point
    ([[1., 0.], [-0.809, 0.588], [0.309, -0.951], [.309, 0.951], [-0.809, -0.588]], 2),  # 5-point star
])
def test_polygon_turning(poly, turning):
    assert numpy.array_equal(polygon_winding(poly), turning)
    assert numpy.array_equal(polygon_winding(numpy.flip(poly, axis=-2)), -turning)


def plot_polygon_winding(poly):
    poly = numpy.atleast_2d(poly)
    xs = numpy.linspace(-4, 4, 50)
    xx, yy = numpy.meshgrid(xs, xs)
    pts = numpy.stack((xx, yy), axis=-1)

    pyplot.pcolormesh(xx, yy, polygon_winding(poly, pts))
    pyplot.scatter(
        *split_arr(poly, axis=-1), s=10,
        c=numpy.arange(poly.shape[-2])  # type: ignore
    )

@pytest.mark.parametrize(['poly', 'pts', 'expected'], [
    (square, [[0., 0., 0.01], [0., 0., -0.01], [0., 0., 0.05], [0., 0., 10.]], [6.22662, -6.22662, 6.000637, 0.0396046]),
    (square, [[3., 1., 0.5], [-3., 1., 0.5], [3., -1., 0.5]], [0.0705337] * 3),
    (hourglass,
    [[0., 0., 0.5], [0., 1.5, 0.5], [0., 0.5, 0.5], [0., -0.5, 0.5], [0., -0.5, -0.5]], [0., -0.439289, -1.637512, 1.637512, -1.637512]),
    ([*square, *square],
    [[0., 0.5, 0.2], [0., 0.5, -0.2], [0., 1.5, 0.5], [0., 1.5, -0.5], [0., 1.5, 0.01]], [9.911478, -9.911478, 1.539936, -1.539936, 0.0463514]),
    (list(reversed(square)), [[0., 0., 0.01], [0., 0., -0.01], [0., 0., 0.05], [0., 0., 10.]], [-6.22662, 6.22662, -6.000637, -0.0396046])
])
def test_solid_angle(poly, pts, expected):
    numpy.testing.assert_array_almost_equal(polygon_solid_angle(poly, pts), expected)