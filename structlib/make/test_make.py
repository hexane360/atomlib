
import numpy

from . import fcc, wurtzite, graphite, zincblende, fluorite, slab
from .. import AtomCell, Atoms
from ..transform import LinearTransform3D
from tests.util import check_equals_structure


def test_fcc():
    cell = fcc('Al', 2., cell='cOnV')  # type: ignore
    expected = AtomCell.from_ortho(Atoms({
        'x': [0., 0., 1.0, 1.0],
        'y': [0., 1.0, 0., 1.0],
        'z': [0., 1.0, 1.0, 0.],
        'symbol': ['Al', 'Al', 'Al', 'Al'],
    }), LinearTransform3D.scale(all=2.), frame='local')

    expected.assert_equal(cell)

    cell = fcc('Al', 2., cell='prim')

    ortho = LinearTransform3D([
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
    ])
    expected = AtomCell.from_ortho(Atoms({
        'x': [0.], 'y': [0.], 'z': [0.],
        'symbol': ['Al'],
    }), ortho, frame='local')

    expected.assert_equal(cell)

    cell = fcc('Al', 2., cell='ortho')

    ortho = LinearTransform3D.scale(numpy.sqrt(2), numpy.sqrt(2), 2.)
    expected = AtomCell.from_ortho(Atoms({
        'x': [0.0, 1/numpy.sqrt(2)],
        'y': [0.0, 1/numpy.sqrt(2)],
        'z': [0.0, 1.],
        'symbol': ['Al', 'Al'],
    }), ortho, frame='local')

    expected.assert_equal(cell)


def test_wurtzite():
    a = 3.13
    c = 5.02
    cell = wurtzite('AlN', a, c, cell='prim')

    ortho = LinearTransform3D([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])
    print(cell)
    expected = AtomCell.from_ortho(Atoms({
        'symbol': ['Al', 'N', 'Al', 'N'],
        'x': [   1.565,    1.565,      0.0,      0.0],
        'y': [0.903553, 0.903553, 1.807106, 1.807106],
        'z': [    2.51,  4.41552,      0.0,  1.90552],
    }), ortho, frame='local')

    expected.assert_equal(cell)


@check_equals_structure('AlN_ortho.xsf')
def test_wurtzite_ortho():
    return wurtzite('AlN', 3.13, 5.02, 0.38, cell='ortho')


@check_equals_structure('CeO2_ortho.xsf')
def test_ceo2_ortho():
    return fluorite('CeO2', 5.47, cell='ortho')


@check_equals_structure('CeO2_conv.xsf')
def test_ceo2_conv():
    return fluorite('CeO2', 5.47)


@check_equals_structure('CeO2_prim.xsf')
def test_ceo2_prim():
    return fluorite('CeO2', 5.47, cell='prim')


@check_equals_structure('ZnSe_conv.xsf')
def test_znse_conv():
    return zincblende('ZnSe', 5.66)


@check_equals_structure('ZnSe_ortho.xsf')
def test_znse_ortho():
    return zincblende('ZnSe', 5.66, cell='ortho')


@check_equals_structure('ZnSe_prim.xsf')
def test_znse_prim():
    return zincblende('ZnSe', 5.66, cell='prim')


@check_equals_structure('CeO2_112_slab.xsf')
def test_slab_ceo2_112():
    cell = fluorite('CeO2', 5.47, cell='prim')
    return slab(cell, [1, 1, 2], [0, 0, 1])


@check_equals_structure('CeO2_100_slab.xsf')
def test_slab_ceo2_100():
    cell = fluorite('CeO2', 5.47, cell='conv')
    return slab(cell, [1, 0, 0], [0, 0, 1])


def test_graphite():
    cell = graphite(cell='prim')

    a = 2.47
    c = 8.69
    ortho = LinearTransform3D([
        [a, -a*numpy.cos(numpy.pi/3), 0.],
        [0., a*numpy.sin(numpy.pi/3), 0.],
        [0., 0., c],
    ])

    expected = AtomCell.from_ortho(Atoms({
        'symbol': ['C'] * 4,
        'x': [0.0, 2/3, 0.0, 1/3],
        'y': [0.0, 1/3, 0.0, 2/3],
        'z': [0.0, 0.0, 1/2, 1/2]
    }), ortho, frame='cell_frac')

    expected.assert_equal(cell)