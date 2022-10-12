
import numpy
import pytest

from . import fcc, wurtzite, graphite
from .. import AtomCell, AtomFrame
from ..transform import LinearTransform


def test_fcc():
    cell = fcc('Al', 2., cell='cOnV')  # type: ignore
    expected = AtomCell(AtomFrame({
        'x': [0., 0., 1.0, 1.0],
        'y': [0., 1.0, 0., 1.0],
        'z': [0., 1.0, 1.0, 0.],
        'symbol': ['Al', 'Al', 'Al', 'Al'],
    }), ortho=LinearTransform.scale(all=2.))

    expected.assert_equal(cell)

    cell = fcc('Al', 2., cell='prim')

    ortho = LinearTransform([
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
    ])
    expected = AtomCell(AtomFrame({
        'x': [0.], 'y': [0.], 'z': [0.],
        'symbol': ['Al'],
    }), ortho=ortho)

    expected.assert_equal(cell)

    cell = fcc('Al', 2., cell='ortho')

    ortho = LinearTransform.scale(numpy.sqrt(2), numpy.sqrt(2), 2.)
    expected = AtomCell(AtomFrame({
        'x': [0.0, 1/numpy.sqrt(2)],
        'y': [0.0, 1/numpy.sqrt(2)],
        'z': [0.0, 1.],
        'symbol': ['Al', 'Al'],
    }), ortho=ortho)

    expected.assert_equal(cell)


def test_wurtzite():
    a = 3.13
    c = 5.02
    cell = wurtzite('AlN', a, c, cell='prim')

    ortho = LinearTransform([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])
    print(cell)
    expected = AtomCell(AtomFrame({
        'symbol': ['Al', 'N', 'Al', 'N'],
        'x': [   1.565,    1.565,      0.0,      0.0],
        'y': [0.903553, 0.903553, 1.807106, 1.807106],
        'z': [    2.51,  4.41552,      0.0,  1.90552],
    }), ortho=ortho)

    expected.assert_equal(cell)


def test_graphite():
    cell = graphite(cell='prim')

    a = 2.47
    c = 8.69
    ortho = LinearTransform([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])

    expected = AtomCell(AtomFrame({
        'symbol': ['C'] * 4,
        'x': [0.0, 2/3, 0.0, 1/3],
        'y': [0.0, 1/3, 0.0, 2/3],
        'z': [0.0, 0.0, 1/2, 1/2]
    }), ortho=ortho, frac=True)

    expected.assert_equal(cell)