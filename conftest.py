import pytest

from structlib import AtomCollection
from structlib.io import read


@pytest.fixture(scope='function')
def expected_structure(request) -> AtomCollection:
    marker = request.node.get_closest_marker('expected_structure_filename')
    name = str(marker.args[0])
    return read(name)