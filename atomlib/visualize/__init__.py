"""Visualization of atomic structures. Useful for debugging."""
from __future__ import annotations

from abc import abstractmethod, ABC
import typing as t

import numpy
from numpy.typing import NDArray
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.collections import EllipseCollection
#from matplotlib.colors import Colormap
#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import PathPatch3D

from ..atoms import HasAtoms
from ..cell import HasCell
from ..atomcell import AtomCell
from ..transform import LinearTransform3D
from ..util import FileOrPath
from ..types import VecLike, to_vec3
from ..vec import split_arr
from ..elem import get_radius


BackendName = t.Literal['mpl', 'ase']
AtomStyle = t.Literal['spacefill', 'ballstick', 'small']


class AtomImage(ABC):
    @abstractmethod
    def save(self, f: FileOrPath):
        ...


def show_atoms_3d(atoms: HasAtoms, *,
                  zone: t.Optional[VecLike] = None,
                  plane: t.Optional[VecLike] = None,
                  backend: BackendName = 'mpl',
                  style: AtomStyle = 'small', **kwargs: t.Any) -> AtomImage:
    backend = t.cast(BackendName, backend.lower())
    if backend == 'mpl':
        return show_atoms_mpl_3d(atoms, zone=zone, plane=plane, style=style, **kwargs)
    elif backend == 'ase':
        raise NotImplementedError()

    raise ValueError(f"Unknown backend '{backend}'")


def show_atoms_2d(atoms: HasAtoms, *,
                  zone: t.Optional[VecLike] = None,
                  plane: t.Optional[VecLike] = None,
                  horz: t.Optional[VecLike] = None,
                  backend: BackendName = 'mpl',
                  style: AtomStyle = 'small', **kwargs: t.Any) -> AtomImage:
    backend = t.cast(BackendName, backend.lower())
    if backend == 'mpl':
        return show_atoms_mpl_2d(atoms, zone=zone, plane=plane, horz=horz, style=style, **kwargs)
    elif backend == 'ase':
        raise NotImplementedError()

    raise ValueError(f"Unknown backend '{backend}'")


class AtomImageMpl(Figure, AtomImage):
    def __new__(cls, fig: Figure):
        fig.__class__ = cls
        return fig

    def __init__(self, fig: Figure):
        ...

    def save(self, f: FileOrPath):
        return self.savefig(f)  # type: ignore


_ELEM_MAP = {
    7: [0, 0, 255],    # N
    8: [255, 0, 0],    # O
    13: [255, 215, 0], # Al
    16: [253, 218, 13], # S
    58: [0, 0, 255], # Ce
    74: [52, 152, 219], # W
    68: [0, 255, 255], # Er
}


def get_elem_color(elem: int) -> t.List[int]:
    # grey fallback
    return _ELEM_MAP.get(elem, [80, 80, 80])


"""
class AtomPatch3D(PathPatch3D):
    def __init__(self, elem: int, fc=None, s: float = 2., **kwargs):
        ...
        self.elem = elem
        if fc is None:
            fc = get_elem_color(elem)

        self.s = s

        super().__init__(fc=fc, **kwargs)
"""


def get_zone(atoms: HasAtoms, zone: t.Optional[VecLike] = None,
             plane: t.Optional[VecLike] = None,
             default: t.Optional[VecLike] = None) -> NDArray[numpy.float64]:
    if zone is not None and plane is not None:
        raise ValueError("'zone' and 'plane' can't both be specified.")
    if plane is not None:
        if isinstance(atoms, AtomCell) and not atoms.is_orthogonal():
            # convert plane into zone
            raise NotImplementedError()
        zone = plane
    if zone is not None:
        return numpy.broadcast_to(zone, 3).astype(numpy.float64)
    if default is not None:
        return numpy.broadcast_to(default, 3).astype(numpy.float64)
    return numpy.array([0., 0., 1.], dtype=numpy.float64)


def get_plot_radii(atoms: HasAtoms, min_r: t.Optional[float] = 1.0, style: AtomStyle = 'small') -> NDArray[numpy.float64]:
    radii = get_radius(atoms['elem']).to_numpy()
    if style == 'small':
        radii = radii * 0.6
    elif style == 'ballstick':
        radii = radii * 0.5
    elif style == 'spacefill':
        radii = radii * 1.0
    else:
        raise ValueError(f"Unknown atom style '{style}'. Expected 'spacefill', 'ballstick', or 'small'.")
    if min_r is not None:
        return numpy.maximum(min_r, radii)
    return radii


def get_azim_elev(zone: VecLike) -> t.Tuple[float, float]:
    (a, b, c) = -to_vec3(zone)  # look down zone
    l = numpy.sqrt(a**2 + b**2)
    # todo: aren't these just arctan2s?
    return (numpy.angle(a + b*1.j, deg=True), numpy.angle(l + c*1.j, deg=True))  # type: ignore


def show_atoms_mpl_3d(atoms: HasAtoms, *, fig: t.Optional[Figure] = None,
                      zone: t.Optional[VecLike] = None,
                      plane: t.Optional[VecLike] = None,
                      min_r: t.Optional[float] = 1.0,
                      style: AtomStyle = 'small') -> AtomImageMpl:
    fig = AtomImageMpl(fig or pyplot.figure())  # type: ignore

    zone = get_zone(atoms, zone, plane, [1., 2., 4.])
    (azim, elev) = get_azim_elev(zone)

    rect = (0., 0., 1., 1.)
    ax: Axes3D = fig.add_axes(rect, axes_class=Axes3D, proj_type='ortho', azim=azim, elev=elev)  # type: ignore
    ax.grid(False)

    bbox = atoms.bbox().pad(0.2)
    ax.set_xlim3d(bbox.x)  # type: ignore
    ax.set_ylim3d(bbox.y)  # type: ignore
    ax.set_zlim3d(bbox.z)  # type: ignore
    ax.set_box_aspect(bbox.size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    frame = atoms.get_atoms('local')
    #radii = get_plot_radii(atoms, min_r=min_r, style=style)
    coords = frame.coords()
    elem_colors = numpy.array(list(map(get_elem_color, frame['elem']))) / 255.
    s = 100

    if isinstance(atoms, HasCell):
        # plot cell corners
        corners = atoms.corners('global')
        faces = [
            numpy.array([
                corners[(val*2**axis + v1*2**((axis+1) % 3) + v2*2**((axis+2) % 3))]
                for (v1, v2) in [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
            ])
            for axis in (0, 1, 2)
            for val in (0, 1)
        ]
        for face in faces:
            ax.plot3D(*split_arr(face, axis=-1), '.-k', alpha=1, markersize=8)

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=elem_colors, alpha=1, s=s)  # type: ignore

    return t.cast(AtomImageMpl, fig)


def show_atoms_mpl_2d(atoms: HasAtoms, *, fig: t.Optional[Figure] = None,
                      zone: t.Optional[VecLike] = None,
                      plane: t.Optional[VecLike] = None,
                      horz: t.Optional[VecLike] = None,
                      min_r: t.Optional[float] = 1.0,
                      style: AtomStyle = 'small') -> AtomImageMpl:
    zone = get_zone(atoms, zone, plane, [0., 0., 1.])
    fig = AtomImageMpl(fig or pyplot.figure())  # type: ignore

    rect = (0.05, 0.05, 0.95, 0.95)
    ax: Axes = fig.add_axes(rect)
    ax.set_aspect('equal')

    frame = atoms.get_atoms('local')
    coords = frame.coords()
    elem_colors = numpy.array(list(map(get_elem_color, frame['elem']))) / 255.
    radii = get_plot_radii(frame, min_r=min_r, style=style)

    # look down zone
    transform = LinearTransform3D.align_to(zone, [0, 0, -1], horz, [1, 0, 0] if horz is not None else None)
    bbox_2d = transform @ atoms.bbox()
    coords = transform @ coords
    # sort by z-order
    sort = numpy.argsort(coords[..., 2])
    coords = coords[sort]
    elem_colors = elem_colors[sort]
    radii = radii[sort]

    ax.set_xbound(*bbox_2d.x)
    ax.set_ybound(*bbox_2d.y)

    # old plotting method
    # ax.scatter(coords[:, 0], coords[:, 1], c=elem_colors, alpha=1., s=s)

    ax.add_collection(EllipseCollection(
        radii, radii, numpy.zeros_like(radii), units='xy', facecolors=elem_colors, ec='black',
        offsets=coords[..., :2], offset_transform=ax.transData,
    ))  # type: ignore (bad api typing)

    return t.cast(AtomImageMpl, fig)
