

from io import StringIO

from .mslice import write_mslice
from ..tests.util import check_file_equal
from ..make import fcc


@check_file_equal('Al_from_template.mslice')
def test_mslice_default_template() -> str:
    cell = fcc('Al', 4.05, cell='conv')
    #cell = cell.repeat((2, 3, 4), explode=False)
    frame = cell.get_atoms('local').with_wobble(0.030)
    cell = cell._replace_atoms(frame, 'local')

    s = StringIO()
    write_mslice(cell, s, slice_thickness=2.025)
    return s.getvalue()
