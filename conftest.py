import typing as t

import pytest
pytest.register_assert_rewrite("structlib.core", "structlib.atoms", "structlib.tests.util")

from tests.util import OUTPUT_PATH

if t.TYPE_CHECKING:
    from structlib import AtomCollection


@pytest.fixture(scope='function')
def expected_structure(request) -> 'AtomCollection':
    from structlib.io import read

    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    return read(OUTPUT_PATH / name)


@pytest.fixture(scope='function')
def expected_contents(request) -> str:
    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    with open(OUTPUT_PATH / name, 'r') as f:
        return f.read()