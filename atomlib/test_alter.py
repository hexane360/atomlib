
from numpy.testing import assert_array_equal
import pytest

from .testing import check_equals_structure

from . import make, alter, AtomCell
from .transform import AffineTransform3D


@pytest.fixture(scope='module')
def znse_supercell():
    return make.zincblende('ZnSe', 5.667).repeat(6)


@check_equals_structure('unbunched.xyz')
def test_unbunch():
    cell = make.random([100., 100., 5.], elems='C', density=1., seed='test_unbunch')
    return alter.unbunch(cell, threshold=3.0, max_iter=30)


@check_equals_structure('ZnSe_contaminated.xsf')
def test_contaminated(znse_supercell: AtomCell):
    cell = alter.contaminate(znse_supercell, (20., 10.), seed='test_znse_cont')
    assert_array_equal(cell.n_cells, [6, 6, 1])
    assert_array_equal(cell.affine.inner, AffineTransform3D.translate([0., 0., -10.]).inner)
    return cell.explode().transform(cell.get_cell().affine.inverse())