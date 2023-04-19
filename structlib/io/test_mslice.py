

from io import StringIO

from .mslice import write_mslice
from ..testing import check_equals_file
from ..make import fcc


@check_equals_file('Al_from_template.mslice')
def test_mslice_default_template(buf: StringIO):
    cell = fcc('Al', 4.05, cell='conv').with_wobble(0.030)
    write_mslice(cell, buf, slice_thickness=2.025)
