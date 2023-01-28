
from pathlib import Path
import inspect
import typing as t

import pytest

if t.TYPE_CHECKING:
    from structlib import AtomCollection

CallableT = t.TypeVar('CallableT', bound=t.Callable)

OUTPUT_PATH = Path(__file__).parent / 'baseline_files'
assert OUTPUT_PATH.exists()

INPUT_PATH = Path(__file__).parent / 'input_files'
assert INPUT_PATH.exists()


def _wrap_pytest(wrapper: CallableT, wrapped: t.Callable,
                 prepend_params: t.Sequence[inspect.Parameter] = (),
                 append_params: t.Sequence[inspect.Parameter] = ()) -> CallableT:
    # hacks to allow pytest to find fixtures in wrapped functions
    old_sig = inspect.signature(wrapped)
    params = [*prepend_params, *old_sig.parameters.values(), *append_params]
    new_sig = old_sig.replace(parameters=params)

    testmark = getattr(wrapped, "pytestmark", []) + getattr(wrapper, "pytestmark", [])
    if len(testmark) > 0:
        wrapper.pytestmark = testmark  # type: ignore
    wrapper.__name__ = wrapped.__name__
    wrapper.__doc__ = wrapped.__doc__
    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


def check_equals_file(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., str]], t.Callable[..., None]]:
    def decorator(f: t.Callable[..., str]):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_contents: str, *args, **kwargs):  # type: ignore
            result = f(*args, **kwargs)
            assert result == expected_contents

        return _wrap_pytest(wrapper, f, [inspect.Parameter('expected_contents', inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    return decorator


def check_equals_structure(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., 'AtomCollection']], t.Callable[..., None]]:
    """Test that the wrapped function returns the same structure as contained in `name`."""
    def decorator(f: t.Callable[..., 'AtomCollection']):
        @pytest.mark.expected_filename(name)
        def wrapper(expected_structure: 'AtomCollection', *args, **kwargs):  # type: ignore
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

        return _wrap_pytest(wrapper, f, [inspect.Parameter('expected_structure', inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    return decorator


def check_parse_structure(name: t.Union[str, Path]) -> t.Callable[[t.Callable[..., 'AtomCollection']], t.Callable[..., None]]:
    """Test that `name` parses to the same structure as given in the function body."""
    def decorator(f: t.Callable[..., 'AtomCollection']):
        def wrapper(*args, **kwargs):  # type: ignore
            expected = f(*args, **kwargs)

            from structlib.io import read
            result = read(INPUT_PATH / name)

            if hasattr(result, 'assert_equal'):
                result.assert_equal(expected)  # type: ignore
            else:
                assert result == expected

        return _wrap_pytest(wrapper, f)
    return decorator


def check_figure_draw(name: t.Union[str, Path, t.Sequence[t.Union[str, Path]]], savefig_kwarg=None) -> t.Callable[[t.Callable[..., None]], t.Callable[..., None]]:
    """Test that the wrapped function draws an identical figure to `name` in `baseline_images`."""

    if isinstance(name, (str, Path)):
        names = (str(name),)
    else:
        names = tuple(map(str, name))

    def decorator(f: t.Callable[..., None]):
        from matplotlib.testing.decorators import image_comparison
        return image_comparison(names, savefig_kwarg=savefig_kwarg)(f)

    return decorator