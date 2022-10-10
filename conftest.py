from pathlib import Path

import pytest

pytest.register_assert_rewrite("structlib.core", "structlib.frame", "structlib.tests.util")

from structlib import AtomCollection
from structlib.io import read

STRUCTURE_PATH = Path(__file__).parent / 'result_structures'
assert STRUCTURE_PATH.exists()


@pytest.fixture(scope='function')
def expected_structure(request) -> AtomCollection:
    marker = request.node.get_closest_marker('expected_structure_filename')
    name = str(marker.args[0])
    return read(STRUCTURE_PATH / name)