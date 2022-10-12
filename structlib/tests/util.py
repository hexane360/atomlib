
from pathlib import Path
import inspect
import typing as t

import pytest

from structlib import AtomCollection


CallableT = t.TypeVar('CallableT', bound=t.Callable)


def _wrap_pytest(wrapper: CallableT, wrapped: t.Callable,
                 prepend_params: t.Sequence[inspect.Parameter] = (),
                 append_params: t.Sequence[inspect.Parameter] = ()) -> CallableT:
    # hacks to allow pytest to find fixtures in wrapped functions
    old_sig = inspect.signature(wrapped)
    params = [*prepend_params, *old_sig.parameters.values(), *append_params]
    new_sig = old_sig.replace(parameters=params)

    wrapper.pytestmark = getattr(wrapped, "pytestmark", []) + wrapper.pytestmark
    wrapper.__name__ = wrapped.__name__
    wrapper.__doc__ = wrapped.__doc__
    wrapper.__signature__ = new_sig
    return wrapper


def check_file_equal(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., str]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., str]):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_contents: str, *args, **kwargs):
            result = f(*args, **kwargs)
            assert result == expected_contents

        return _wrap_pytest(wrapper, f, [inspect.Parameter('expected_contents', inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    return decorator


def check_structure_equal(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., AtomCollection]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., AtomCollection]):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_structure: AtomCollection, *args, **kwargs):
            result = f(*args, **kwargs)
            if hasattr(result, 'assert_equal'):
                result.assert_equal(expected_structure)  # type: ignore
            else:
                assert expected_structure == result

        return _wrap_pytest(wrapper, f, [inspect.Parameter('expected_structure', inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    return decorator