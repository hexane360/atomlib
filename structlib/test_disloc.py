
import numpy

from .tests.util import check_structure_equal
from . import make
from .disloc import disloc_loop_z


@check_structure_equal('Al_disloc_loop_extrinsic.xsf')
def test_disloc_loop_z_extrinsic():
    cell = make.fcc('Al', 4.05, cell='conv')
    cell = cell.repeat(8, explode=True)

    return disloc_loop_z(cell, center=4.05*4.01*numpy.ones(3), b=4.05/2.*numpy.array([0,1,1]), loop_r=10, poisson=0.32)


@check_structure_equal('Al_disloc_loop_intrinsic.xsf')
def test_disloc_loop_z_intrinsic():
    cell = make.fcc('Al', 4.05, cell='conv')
    cell = cell.repeat(8, explode=True)

    return disloc_loop_z(cell, center=4.05*4.01*numpy.ones(3), b=-4.05/2.*numpy.array([0,1,1]), loop_r=10, poisson=0.32)