import pytest
import numpy
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib import pyplot

from ..visualize import show_atoms_2d, show_atoms_mpl_2d, get_elem_color
from .. import make, AtomCell


@pytest.fixture
def aln_cell():
    cell = make.wurtzite('AlN', 3.13, 5.02)
    cell = cell.repeat((2, 2, 2)).explode()
    return cell


@check_figures_equal(extensions=('pdf',))
def test_show_atoms_mpl_2d(fig_test, fig_ref, aln_cell: AtomCell):
    cell = aln_cell

    assert show_atoms_mpl_2d(cell, fig=fig_test, zone=[0, 0, 1], horz=[1, 0, 0], s=20.) is fig_test

    rect = [0.05, 0.05, 0.95, 0.95]
    ax = fig_ref.add_axes(rect)
    ax.set_xbound(*cell.cell.bbox().x)
    ax.set_ybound(*cell.cell.bbox().y)
    ax.set_aspect('equal')

    coords = cell.atoms.coords()

    colors = numpy.array(list(map(get_elem_color, cell.atoms['elem']))) / 255.
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=1, s=20.)