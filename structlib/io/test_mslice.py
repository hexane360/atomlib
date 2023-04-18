

from io import StringIO

from .mslice import write_mslice
from tests.util import check_equals_file
from ..make import fcc


@check_equals_file('Al_from_template.mslice')
def test_mslice_default_template() -> str:
    cell = fcc('Al', 4.05, cell='conv').with_wobble(0.030)

    s = StringIO()
    write_mslice(cell, s, slice_thickness=2.025)
    return s.getvalue()
