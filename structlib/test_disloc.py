
import numpy
import pytest

from tests.util import check_equals_structure
from . import make
from .disloc import disloc_loop_z, disloc_edge, disloc_screw


@check_equals_structure('Al_disloc_loop_extrinsic.xsf')
def test_disloc_loop_z_extrinsic():
    cell = make.fcc('Al', 4.05, cell='conv')
    cell = cell.repeat(8).explode()

    return disloc_loop_z(cell, center=4.05*4.01*numpy.ones(3), b=4.05/2.*numpy.array([0,1,1]), loop_r=10, poisson=0.32)


@check_equals_structure('Al_disloc_loop_intrinsic.xsf')
def test_disloc_loop_z_intrinsic():
    cell = make.fcc('Al', 4.05, cell='conv')
    cell = cell.repeat(8).explode()

    return disloc_loop_z(cell, center=4.05*4.01*numpy.ones(3), b=-4.05/2.*numpy.array([0,1,1]), loop_r=10, poisson=0.32)


@check_equals_structure('disloc_AlN_edge_shift.xsf')
def test_disloc_edge_shift(aln_cell):
    b = aln_cell.cell.cell_size[1] / 5.
    # shockley partial dislocation, shifted half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., b / 3., 0.], t=[0., 0., 1.], cut='shift')


@check_equals_structure('disloc_AlN_edge_add.xsf')
def test_disloc_edge_add(aln_cell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, added half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut='add')


@check_equals_structure('disloc_AlN_edge_rm.xsf')
def test_disloc_edge_rm(aln_cell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, removed half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut='rm')


@check_equals_structure('disloc_AlN_edge_cut011.xsf')
def test_disloc_edge_cut011(aln_cell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, weird cut plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut=[0., 1., 1.])


@check_equals_structure('disloc_AlN_screw_cut010.xsf')
def test_disloc_screw_cut010(aln_cell):
    c = aln_cell.cell.cell_size[2] / 5.
    # perfect basal screw dislocation
    return disloc_screw(aln_cell, [12., 12., 12.], [0., 0., c], cut=[0., 1., 0.])


@check_equals_structure('disloc_AlN_screw_cut010_neg.xsf')
def test_disloc_screw_neg(aln_cell):
    c = aln_cell.cell.cell_size[2] / 5.
    return disloc_screw(aln_cell, [12., 12., 12.], [0., 0., c], cut=[0., 1., 0.], sign=False)


@pytest.fixture
def aln_cell():
    cell = make.wurtzite('AlN', 3.13, 5.02, 0.38, cell='ortho')
    return cell.repeat((8, 5, 5)).explode()