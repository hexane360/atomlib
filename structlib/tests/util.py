
from pathlib import Path
import inspect
import typing as t

import pytest

from structlib import AtomCollection


def check_structure_equal(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., AtomCollection]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., AtomCollection]):

        @pytest.mark.expected_structure_filename(name)
        def wrapper(expected_structure: AtomCollection, *args, **kwargs):
            result = f(*args, **kwargs)
            if hasattr(result, 'assert_equal'):
                result.assert_equal(expected_structure)  # type: ignore
            else:
                assert expected_structure == result

        # hacks to pass pytest fixtures through to wrapper
        old_sig = inspect.signature(f)
        params = list(old_sig.parameters.values())
        params.insert(0, inspect.Parameter('expected_structure', inspect.Parameter.POSITIONAL_OR_KEYWORD))
        new_sig = old_sig.replace(parameters=params)

        wrapper.pytestmark = getattr(f, "pytestmark", []) + wrapper.pytestmark
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        wrapper.__signature__ = new_sig
        return wrapper

    return decorator