
from pathlib import Path
import typing as t

import pytest

from structlib import AtomCollection


def check_structure_equal(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., AtomCollection]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., AtomCollection]):
        @pytest.mark.expected_structure_filename(name)
        def wrapper(expected_structure: AtomCollection, *args, **kwargs):
            result = f(*args, **kwargs)
            assert expected_structure == result

        wrapper.pytestmark += getattr(f, "pytestmark", [])
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        return wrapper

    return decorator