
import warnings
import numpy
import numpy.testing
import pytest
from matplotlib import pyplot

from .util import reduce_vec, miller_3_to_4_plane, miller_3_to_4_vec, miller_4_to_3_plane, miller_4_to_3_vec
from .util import polygon_winding, in_polygon, split_arr


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
        warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
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
    ([[-1, -1.], [1., -1.], [1., 1.], [-1., 1.]], # square
     [[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.], [-1.5, 0.]], [1, 0, 1, 0, 1, 0]),
    ([[-1, -1.], [1., -1.], [-1., 1.], [1., 1.]],  # hourglass
     [[0., 0.1], [0.2, 0.], [-0.2, 0.], [0.9, 0.95], [0.9, -0.95]], [-1, 0, 0, -1, 1]),
]
)
def test_polygon_winding(poly, pts, windings):
    assert numpy.array_equal(polygon_winding(poly, pts), windings)


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