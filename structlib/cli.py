from __future__ import annotations
import sys
from functools import update_wrapper
from dataclasses import dataclass, field
from pathlib import Path
import typing as t
import logging

import click

from . import CoordinateFrame, HasAtoms, AtomSelection
from .types import ParamSpec, Concatenate
from .transform import LinearTransform3D, AffineTransform3D


frame_type = click.Choice(('global', 'local', 'frac'), case_sensitive=False)
P = ParamSpec('P')


@dataclass
class State:
    structure: HasAtoms
    """Current structure"""
    indices: t.List[int] = field(default_factory=list)
    """Loop indices"""
    outputted: t.Dict[Path, int] = field(default_factory=dict)
    """Number of files outputted to each nominal pathname."""

    selection: t.Optional[AtomSelection] = None
    """Atom selection for use in manipulation commands"""

    def add_index(self):
        self.indices.append(0)

    def pop_index(self):
        self.indices.pop()

    def map_struct(self, f: t.Callable[[HasAtoms], HasAtoms]) -> State:
        self.structure = f(self.structure)
        return self

    def deduplicated_output_path(self, path: t.Union[str, Path]) -> Path:
        """Return a version of `path` deduplicated to avoid overwriting files from ourselves."""
        path = Path(path).resolve(strict=False)
        if path not in self.outputted:
            self.outputted[path] = 0
            return path
        i = self.outputted[path]
        self.outputted[path] = i + 1
        return path.parent / (path.name + str(self.outputted[path]) + path.suffix)


CmdType = t.Callable[[t.Iterable[State]], t.Iterable[State]]


def init_logging(verbose: int = 0):
    if verbose == 0:
        log_fmt = "{message}"
        log_level = logging.INFO
    else:
        log_fmt = "{asctime}: {levelname} {message}"
        if verbose > 1:
            log_fmt += "\n  at {funcName} in {filename}:{lineno}"
        log_level = logging.DEBUG
    logging.basicConfig(format=log_fmt, style="{", level=log_level, datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stderr)


@click.group(chain=True)
@click.option('-v', '--verbose', count=True)
def cli(verbose: int = 0):
    init_logging(verbose)


@cli.result_callback()
def run_chain(cmds: t.Sequence[CmdType], verbose: int = 0):
    states: t.Iterable[State] = ()
    for cmd in cmds:
        if cmd is None:
            raise RuntimeError("'cmd' is None. Did a command forget to return a wrapper function?")
        states = cmd(states)
    for _ in states:
        pass


def lazy(f: t.Callable[Concatenate[t.Iterable[State], P], t.Iterable[State]]) -> t.Callable[P, CmdType]:
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        return lambda states: f(states, *args, **kwargs)

    return update_wrapper(wrapped, f)


def lazy_append(f: t.Callable[P, t.Iterable[HasAtoms]]) -> t.Callable[P, CmdType]:
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        def inner(states: t.Iterable[State]) -> t.Iterable[State]:
            state = None
            for state in states:
                yield state
            if state is None:
                indices = []
                outputted = {}
            else:
                indices = state.indices
                outputted = state.outputted

            for struct in f(*args, **kwargs):
                if len(indices) == 0:
                    indices.append(0)
                else:
                    indices[-1] += 1
                yield State(struct, indices, outputted)

        return inner

    return update_wrapper(wrapped, f)


def lazy_map(f: t.Callable[Concatenate[State, P], t.Iterable[State]]) -> t.Callable[P, CmdType]:
    def wrapped(*args: P.args, **kwargs: P.kwargs):
        def inner(states: t.Iterable[State]) -> t.Iterable[State]:
            for state in states:
                yield from f(state, *args, **kwargs)
        return inner

    return update_wrapper(wrapped, f)


file_type = click.Path(exists=True, dir_okay=False, allow_dash=True, path_type=Path)
file_type_no_stdin = click.Path(exists=True, dir_okay=False, allow_dash=False, path_type=Path)
out_file_type = click.Path(allow_dash=True, path_type=Path)
out_file_type_no_stdout = click.Path(allow_dash=False, path_type=Path)


@cli.command('in')
@click.argument('file', type=file_type_no_stdin)
@lazy_append
def input(file: Path):
    """Input a crystal structure from `file`. Type will be detected via file extension."""
    yield HasAtoms.read(file)


@cli.command('in_cif')
@click.argument('file', type=file_type, required=False)
@lazy_append
def input_cif(file: t.Optional[Path] = None):
    """Input a CIF structure. If `file` is not specified, use stdin."""
    yield HasAtoms.read_cif(file or sys.stdin)


@cli.command('in_xyz')
@click.argument('file', type=file_type, required=False)
@lazy_append
def input_xyz(file: t.Optional[Path] = None):
    """Input an XYZ structure. If `file` is not specified, use stdin."""
    yield HasAtoms.read_xyz(file or sys.stdin)


@cli.command('in_xsf')
@click.argument('file', type=file_type, required=False)
@lazy_append
def input_xsf(file: t.Optional[Path] = None):
    """Input an XSF structure. If `file` is not specified, use stdin."""
    yield HasAtoms.read_xsf(file or sys.stdin)


@cli.command('loop')
@click.argument('n', type=click.IntRange(min=0), required=True)
@lazy_map
def loop(state: State, n: int) -> t.Iterable[State]:
    """
    Create a loop of `n` structures. Loops may be nested, and 'closed' with collecting
    functions like 'union'.
    """
    state.add_index()
    for i in range(n):
        state.indices[-1] = i
        yield state
    state.pop_index()


@cli.command('union')
@lazy
def union(states: t.Iterable[State]) -> t.Iterable[State]:
    """Combine structures. Symmetry is discarded, but """
    last_index = None
    collect: t.List[HasAtoms] = []
    state = None
    for state in states:
        if last_index is None:
            last_index = state.indices[-1]
        elif last_index != state.indices[-1]:
            state.structure = HasAtoms.union(collect)
            state.indices.pop()
            yield state
            last_index = state.indices[-1]
        collect.append(state.structure)
    if state is not None:
        state.structure = HasAtoms.union(collect)
        state.indices.pop()
        yield state


@cli.command('crop')
@click.option('--x_min', '--x-min', type=float)
@click.option('--x_max', '--x-max', type=float)
@click.option('--y_min', '--y-min', type=float)
@click.option('--y_max', '--y-max', type=float)
@click.option('--z_min', '--z-min', type=float)
@click.option('--z_max', '--z-max', type=float)
@click.option('-f', '--frame', type=frame_type, default='global')
@lazy_map
def crop(state: State,
         x_min: t.Optional[float] = None, x_max: t.Optional[float] = None,
         y_min: t.Optional[float] = None, y_max: t.Optional[float] = None,
         z_min: t.Optional[float] = None, z_max: t.Optional[float] = None,
         frame: CoordinateFrame = 'global'):
    """
    Crop the structure box to the given coordinates.
    If none are specified, refers to the global (cartesian) coordinate system.
    Currently, does not update the structure box (but probably should)
    """
    state.structure = state.structure.crop(x_min, x_max, y_min, y_max, z_min, z_max, frame=frame)
    yield state


@cli.command('rotate')
@click.option('-x', type=float, default=0.)
@click.option('-y', type=float, default=0.)
@click.option('-z', type=float, default=0.)
@click.option('-t', '--theta', type=float)
@click.option('-f', '--frame', type=frame_type, default='global',
              help="Frame of reference to transform in")
@lazy_map
def rotate(state: State,
           x: float = 0., y: float = 0., z: float = 0.,
           theta: t.Optional[float] = None,
           frame: CoordinateFrame = 'global'):
    if theta is None:
        transform = LinearTransform3D().rotate_euler(x, y, z)
    else:
        transform = LinearTransform3D().rotate([x, y, z], theta)
    state.structure = state.structure.transform_atoms(transform, frame=frame)
    yield state


@cli.command('translate')
@click.option('-x', type=float, default=0.)
@click.option('-y', type=float, default=0.)
@click.option('-z', type=float, default=0.)
#@click.option('-f', '--frame', type=frame_type, default='global',
#              help="Frame of reference to transform in")
@lazy_map
def translate(state: State, x: float = 0., y: float = 0., z: float = 0.,
              frame: CoordinateFrame = 'global'):
    state.structure = state.structure.transform(AffineTransform3D().translate(x, y, z))
    yield state


@cli.command('print')
@lazy_map
def print_(state: State):
    print(f"index: {tuple(state.indices)}")
    print(state.structure)
    yield state


@cli.command('out')
@click.argument('file', type=out_file_type_no_stdout, required=True)
@lazy_map
def out(state: State, file: Path):
    path = state.deduplicated_output_path(file)
    state.structure.write(path)
    yield state


@cli.command('out_mslice')
@click.argument('file', type=out_file_type, required=False)
@lazy_map
def out_mslice(state: State, file: t.Optional[Path] = None):
    state.structure.write_mslice(sys.stdout if file is None or file == '-' else file)
    yield state


@cli.command('out_xyz')
@click.argument('file', type=out_file_type, required=False)
@click.option('-f', '--frame', type=frame_type, default='global',
              help="Frame of reference to output coordinates in.")
@click.option('--ext/--no-ext', default=True, help="Write extended format")
@click.option('-c', '--comment', type=str)
@lazy_map
def out_xyz(state: State, file: t.Optional[Path] = None, frame: CoordinateFrame = 'global', ext: bool = True, comment: t.Optional[str] = None):
    if file is None or file == '-':
        state.structure.write_xyz(sys.stdout, frame=frame, ext=ext, comment=comment)
    else:
        with open(state.deduplicated_output_path(file), 'w') as f:
            state.structure.write_xyz(f, frame=frame, ext=ext, comment=comment)
    yield state
