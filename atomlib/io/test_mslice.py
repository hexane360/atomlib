
from io import StringIO

import pytest
import numpy

from .mslice import write_mslice
from ..testing import check_equals_file, INPUT_PATH, OUTPUT_PATH
from ..make import fcc, fluorite
from ..io import read
from ..atomcell import HasAtomCell
from ..transform import AffineTransform3D


@pytest.fixture(scope='module')
def ceo2_ortho_cell():
    return fluorite('CeO2', 5.47, cell='ortho') \
        .transform(AffineTransform3D.rotate_euler(x=numpy.pi/2.).translate(y=10.))


@check_equals_file('Al_from_template.mslice')
def test_mslice_default_template(buf: StringIO):
    cell = fcc('Al', 4.05, cell='conv').with_wobble(0.030)
    write_mslice(cell, buf, slice_thickness=2.025)


@check_equals_file('CeO2_ortho_rotated.mslice')
def test_mslice_custom_template(buf: StringIO, ceo2_ortho_cell):
    write_mslice(ceo2_ortho_cell, buf, template=INPUT_PATH / 'bare_template.mslice')


@check_equals_file('Al_roundtrip.mslice')
def test_mslice_roundtrip(buf: StringIO):
    cell = read(OUTPUT_PATH / 'Al_roundtrip.mslice')
    assert isinstance(cell, HasAtomCell)
    write_mslice(cell, buf, slice_thickness=2.025)
