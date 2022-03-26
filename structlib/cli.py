from functools import update_wrapper
import sys
import typing as t
import logging

import click

from . import CoordinateFrame, Structure
from .transform import LinearTransform, AffineTransform


frame_type = click.Choice(('global', 'local', 'frac'), case_sensitive=False)


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
def run_chain(cmds, verbose: int = 0):
    states: t.Iterable = ()
    for cmd in cmds:
        if cmd is None:
            raise RuntimeError("'cmd' is None. Did a command forget to return a wrapper function?")
        states = cmd(states)
    for _ in states:
        pass


def lazy(f):
    def wrapped(*args, **kwargs):
        return lambda states: f(states, *args, **kwargs)

    return update_wrapper(wrapped, f)


def lazy_map(f):
    def wrapped(*args, **kwargs):
        def inner(states):
            for state in states:
                yield from f(state, *args, **kwargs)
        return inner

    return update_wrapper(wrapped, f)


@cli.command('in')
@click.argument('file', type=click.Path(exists=True, dir_okay=False))
@lazy
def input(states, file: str):
    yield Structure.from_file(file)


@cli.command('in_cif')
@click.argument('file', type=click.File('r'), required=False)
def input_cif(states, file: t.Optional[t.TextIO]):
    file = sys.stdin if file is None else file
    yield Structure.from_cif(file)


@cli.command('in_xyz')
@click.argument('file', type=click.File('r'), required=False)
def input_xyz(states, file: t.Optional[t.TextIO]):
    file = sys.stdin if file is None else file
    yield Structure.from_xyz(file)


@cli.command('trim')
@click.option('--x_min', '--x-min', type=float)
@click.option('--x_max', '--x-max', type=float)
@click.option('--y_min', '--y-min', type=float)
@click.option('--y_max', '--y-max', type=float)
@click.option('--z_min', '--z-min', type=float)
@click.option('--z_max', '--z-max', type=float)
@click.option('-f', '--frame', type=frame_type, default='global')
@lazy_map
def trim(state: Structure,
         x_min: t.Optional[float] = None, x_max: t.Optional[float] = None,
         y_min: t.Optional[float] = None, y_max: t.Optional[float] = None,
         z_min: t.Optional[float] = None, z_max: t.Optional[float] = None,
         frame: CoordinateFrame = 'global'):
    yield state.trim(x_min, x_max, y_min, y_max, z_min, z_max, frame=frame)


@cli.command('rotate')
@click.option('-x', type=float, default=0.)
@click.option('-y', type=float, default=0.)
@click.option('-z', type=float, default=0.)
@click.option('-t', '--theta', type=float)
@click.option('-f', '--frame', type=frame_type, default='global',
              help="Frame of reference to transform in")
@lazy_map
def rotate(state: Structure,
           x: float = 0., y: float = 0., z: float = 0.,
           theta: t.Optional[float] = None,
           frame: CoordinateFrame = 'global'):
    if theta is None:
        transform = LinearTransform().rotate_euler(x, y, z)
    else:
        transform = LinearTransform().rotate([x, y, z], theta)
    yield state.transform_atoms(transform, frame=frame)


@cli.command('translate')
@click.option('-x', type=float, default=0.)
@click.option('-y', type=float, default=0.)
@click.option('-z', type=float, default=0.)
#@click.option('-f', '--frame', type=frame_type, default='global',
#              help="Frame of reference to transform in")
@lazy_map
def translate(state: Structure, x: float = 0., y: float = 0., z: float = 0.,
              frame: CoordinateFrame = 'global'):
    yield state.transform(AffineTransform().translate(x, y, z))


@cli.command('print')
@lazy_map
def print_(state: Structure):
    print(f"n_cells: {state.n_cells}")
    print(f"cell_size: {state.cell_size}")
    print(state.atoms)
    yield state


@cli.command('out')
@click.argument('file', type=click.Path(allow_dash=False), required=True)
def out(file: str):
    def wrapped(states: t.Sequence[Structure]):
        for i, state in enumerate(states):
            state.write(file)
            yield state
    return wrapped


@cli.command('out_mslice')
@click.argument('file', type=click.Path(allow_dash=True), required=False)
def out_mslice(file):
    def wrapped(states: t.Sequence[Structure]):
        for i, state in enumerate(states):
            f = open(file, 'w') if file is not None and file != '-' else sys.stdout
            state.write_mslice(f)
            yield state
    return wrapped


@cli.command('out_xyz')
@click.argument('file', type=click.Path(allow_dash=True), required=False)
@click.option('-f', '--frame', type=frame_type, default='global',
              help="Frame of reference to output coordinates in.")
@click.option('--ext/--no-ext', default=True, help="Write extended format")
@click.option('-c', '--comment', type=str)
def out_xyz(file, frame: CoordinateFrame = 'global', ext: bool = True, comment: t.Optional[str] = None):
    def wrapped(states: t.Sequence[Structure]):
        for i, state in enumerate(states):
            f = open(file, 'w') if file is not None and file != '-' else sys.stdout
            state.write_xyz(f, frame=frame, ext=ext, comment=comment)
            yield state
    return wrapped