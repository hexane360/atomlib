import typing as t

import pytest
pytest.register_assert_rewrite("atomlib.atoms", "atomlib.atomcell", "atomlib.testing")


if t.TYPE_CHECKING:
    from atomlib import HasAtoms


@pytest.fixture(scope='function')
def expected_structure(request) -> 'HasAtoms':
    from atomlib.io import read
    from atomlib.testing import OUTPUT_PATH

    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    return read(OUTPUT_PATH / name)


@pytest.fixture(scope='function')
def expected_contents(request) -> str:
    from atomlib.testing import OUTPUT_PATH

    marker = request.node.get_closest_marker('expected_filename')
    name = str(marker.args[0])
    with open(OUTPUT_PATH / name, 'r') as f:
        return f.read()
