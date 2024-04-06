
import numpy
import pytest

from .testing import check_equals_structure
from . import make, AtomCell
from .defect import disloc_loop_z, disloc_edge, disloc_screw, stacking_fault


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
def test_disloc_edge_shift(aln_cell: AtomCell):
    b = aln_cell.cell.cell_size[1] / 5.
    # shockley partial dislocation, shifted half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., b / 3., 0.], t=[0., 0., 1.], cut='shift')


@check_equals_structure('disloc_AlN_edge_add.xsf')
def test_disloc_edge_add(aln_cell: AtomCell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, added half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut='add')


@check_equals_structure('disloc_AlN_edge_rm.xsf')
def test_disloc_edge_rm(aln_cell: AtomCell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, removed half plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut='rm')


@check_equals_structure('disloc_AlN_edge_cut011.xsf')
def test_disloc_edge_cut011(aln_cell: AtomCell):
    c = aln_cell.cell.cell_size[2] / 5.
    # basal partial dislocation, weird cut plane
    return disloc_edge(aln_cell, [12., 12., 12.], [0., 0., c/2.], t=[1., 0., 0.], cut=[0., 1., 1.])


@check_equals_structure('disloc_AlN_screw_cut010.xsf')
def test_disloc_screw_cut010(aln_cell: AtomCell):
    c = aln_cell.cell.cell_size[2] / 5.
    # perfect basal screw dislocation
    return disloc_screw(aln_cell, [12., 12., 12.], [0., 0., c], cut=[0., 1., 0.])


@check_equals_structure('disloc_AlN_screw_cut010_neg.xsf')
def test_disloc_screw_neg(aln_cell: AtomCell):
    c = aln_cell.cell.cell_size[2] / 5.
    return disloc_screw(aln_cell, [12., 12., 12.], [0., 0., c], cut=[0., 1., 0.], sign=False)


@check_equals_structure('stacking_fault_znse_shift.xsf')
def test_stacking_fault_shift(znse_cell: AtomCell):
    return stacking_fault(
        znse_cell,
        znse_cell.box_size*0.5 + znse_cell.cell_size * 0.3,
        znse_cell.cell_size * [1, 1, -2] / 6.,
        [1, 1, 1],
    ).crop_atoms(0.01, 0.99, 0.01, 0.99, 0.01, 0.99, frame='cell_box').explode()


@check_equals_structure('stacking_fault_znse_add.xsf')
def test_stacking_fault_add(znse_cell: AtomCell):
    return stacking_fault(
        znse_cell,
        znse_cell.box_size*0.5 + znse_cell.cell_size * 0.3,
        znse_cell.cell_size * [1, 1, 1] / 3.,
        [1, 1, 1],
    ).crop_atoms(0.01, 0.99, 0.01, 0.99, 0.01, 0.99, frame='cell_box').explode()


@check_equals_structure('stacking_fault_znse_rm.xsf')
def test_stacking_fault_rm(znse_cell: AtomCell):
    return stacking_fault(
        znse_cell,
        znse_cell.box_size*0.5 + znse_cell.cell_size * 0.3,
        -znse_cell.cell_size * [1, 1, 1] / 3.,
        [1, 1, 1],
    ).crop_atoms(0.01, 0.99, 0.01, 0.99, 0.01, 0.99, frame='cell_box').explode()


@pytest.fixture
def aln_cell() -> AtomCell:
    cell = make.wurtzite('AlN', 3.13, 5.02, 0.38, cell='ortho')
    return cell.repeat((8, 5, 5)).explode()


@pytest.fixture
def znse_cell() -> AtomCell:
    cell = make.zincblende('ZnSe', 5.667, cell='conv')
    return cell.repeat((6, 6, 6))