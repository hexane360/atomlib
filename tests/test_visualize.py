import pytest
from matplotlib import pyplot

from atomlib.visualize import show_atoms_mpl_2d
from atomlib import make, AtomCell

from atomlib.testing import check_figure_draw


@pytest.fixture
def aln_cell():
    cell = make.wurtzite('AlN', 3.13, 5.02)
    cell = cell.repeat((2, 2, 2)).explode()
    return cell


@check_figure_draw('mpl_aln_2d.png')
def test_show_atoms_mpl_2d(aln_cell: AtomCell):
    fig = pyplot.figure()
    assert show_atoms_mpl_2d(aln_cell, fig=fig, zone=[0, 0, 1], horz=[1, 0, 0], s=20.) is fig


"""
@check_figures_equal()
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
"""
