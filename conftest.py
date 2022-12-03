from pathlib import Path

import pytest

pytest.register_assert_rewrite("structlib.core", "structlib.frame", "structlib.tests.util")

from structlib import AtomCollection
from structlib.io import read
from structlib.tests.util import STRUCTURE_PATH


@pytest.fixture(scope='function')
def expected_structure(request) -> AtomCollection:
    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    return read(STRUCTURE_PATH / name)


@pytest.fixture(scope='function')
def expected_contents(request) -> str:
    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    with open(STRUCTURE_PATH / name, 'r') as f:
        return f.read()