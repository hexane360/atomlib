from __future__ import annotations

from pathlib import Path
import inspect
from io import StringIO
import re
import typing as t

import pytest

if t.TYPE_CHECKING:
    from atomlib import HasAtoms
    from atomlib.mixins import AtomsIOMixin

CallableT = t.TypeVar('CallableT', bound=t.Callable)

OUTPUT_PATH = Path(__file__).parents[2] / 'tests/baseline_files'
assert OUTPUT_PATH.exists()

INPUT_PATH = Path(__file__).parents[2] / 'tests/input_files'
assert INPUT_PATH.exists()


def _wrap_pytest(wrapper: CallableT, wrapped: t.Callable,
                 mod_params: t.Optional[t.Callable[[t.Sequence[inspect.Parameter]], t.Sequence[inspect.Parameter]]] = None
) -> CallableT:
    # hacks to allow pytest to find fixtures in wrapped functions
    old_sig = inspect.signature(wrapped)
    params = tuple(old_sig.parameters.values())
    if mod_params is not None:
        params = mod_params(params)
    new_sig = old_sig.replace(parameters=params)

    testmark = getattr(wrapped, "pytestmark", []) + getattr(wrapper, "pytestmark", [])
    if len(testmark) > 0:
        wrapper.pytestmark = testmark  # type: ignore
    wrapper.__name__ = wrapped.__name__
    wrapper.__doc__ = wrapped.__doc__
    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


def assert_files_equal(expected_path: t.Union[str, Path], actual_path: t.Union[str, Path]):
    with open(OUTPUT_PATH / expected_path, 'r') as f:
        expected = re.sub('\r\n', '\n', f.read())
    with open(actual_path, 'r') as f:
        actual = re.sub('\r\n', '\n', f.read())

    assert expected == actual


def check_equals_file(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., t.Any]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., str]):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_contents: str, *args, **kwargs):  # type: ignore
            buf = StringIO()
            f(buf, *args, **kwargs)
            assert buf.getvalue() == expected_contents

        return _wrap_pytest(wrapper, f, lambda params: [inspect.Parameter('expected_contents', inspect.Parameter.POSITIONAL_OR_KEYWORD), *params[1:]])

    return decorator


def assert_structure_equal(expected_path: t.Union[str, Path], actual: t.Union[str, Path, AtomsIOMixin]):
    from atomlib.io import read

    expected = read(OUTPUT_PATH / expected_path)

    try:
        if isinstance(actual, (str, Path)):
            actual = t.cast('AtomsIOMixin', read(actual))
    except Exception:
        print("Failed to load structure under test.")
        raise

    try:
        if hasattr(actual, 'assert_equal'):
            actual.assert_equal(expected)  # type: ignore
        else:
            assert actual == expected
    except AssertionError:
        try:
            actual_path = Path(expected_path).with_stem(Path(expected_path).stem + '_actual').name
            print(f"Saving result structure to '{actual_path}'")
            actual.write(OUTPUT_PATH / actual_path)
        except Exception:
            print("Failed to save result structure.")
        raise


def check_equals_structure(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., AtomsIOMixin]], t.Callable[..., None]]:
    """Test that the wrapped function returns the same structure as contained in `name`."""
    def decorator(f: t.Callable[..., 'AtomsIOMixin']):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_structure: 'HasAtoms', *args, **kwargs):  # type: ignore
            result = f(*args, **kwargs)
            try:
                if hasattr(result, 'assert_equal'):
                    result.assert_equal(expected_structure)  # type: ignore
                else:
                    assert result == expected_structure
            except AssertionError:
                try:
                    actual_path = Path(name).with_stem(Path(name).stem + '_actual').name
                    print(f"Saving result structure to '{actual_path}'")
                    result.write(OUTPUT_PATH / actual_path)
                except Exception:
                    print("Failed to save result structure.")
                raise

        return _wrap_pytest(wrapper, f, lambda params: [inspect.Parameter('expected_structure', inspect.Parameter.POSITIONAL_OR_KEYWORD), *params])

    return decorator


def check_parse_structure(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., HasAtoms]], t.Callable[..., None]]:
    """Test that `name` parses to the same structure as given in the function body."""
    def decorator(f: t.Callable[..., 'HasAtoms']):
        def wrapper(*args, **kwargs):  # type: ignore
            expected = f(*args, **kwargs)

            from atomlib.io import read
            result = read(INPUT_PATH / name)

            if hasattr(result, 'assert_equal'):
                result.assert_equal(expected)  # type: ignore
            else:
                assert result == expected

        return _wrap_pytest(wrapper, f)
    return decorator


def check_figure_draw(name: t.Union[str, Path, t.Sequence[t.Union[str, Path]]],
                      savefig_kwarg=None) -> t.Callable[[t.Callable[..., None]], t.Callable[..., None]]:
    """Test that the wrapped function draws an identical figure to `name` in `baseline_images`."""

    if isinstance(name, (str, Path)):
        names = (str(name),)
    else:
        names = tuple(map(str, name))

    def decorator(f: t.Callable[..., None]):
        from matplotlib.testing.decorators import image_comparison
        return image_comparison(names, savefig_kwarg=savefig_kwarg)(f)

    return decorator
