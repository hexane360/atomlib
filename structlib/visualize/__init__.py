"""Visualization of atomic structures. Useful for debugging."""

from abc import abstractmethod, ABC
from re import A
import typing as t

import numpy
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import PathPatch3D

from ..core import AtomCollection, AtomCell
from ..transform import LinearTransform
from ..util import FileOrPath, BinaryFileOrPath, split_arr
from ..types import VecLike


BackendName = t.Union[t.Literal['mpl'], t.Literal['ase']]


class AtomImage(ABC):
    @abstractmethod
    def save(self, f: FileOrPath):
        ...


def show_atoms_3d(atoms: AtomCollection, *,
                  zone: t.Optional[VecLike] = None,
                  plane: t.Optional[VecLike] = None,
                  backend: BackendName = 'mpl') -> AtomImage:
    backend = t.cast(BackendName, backend.lower())
    if backend == 'mpl':
        return show_atoms_mpl_3d(atoms, zone=zone, plane=plane)
    elif backend == 'ase':
        raise NotImplementedError()

    raise ValueError(f"Unknown backend '{backend}'")


def show_atoms_2d(atoms: AtomCollection, *,
                  zone: t.Optional[VecLike] = None,
                  plane: t.Optional[VecLike] = None,
                  horz: t.Optional[VecLike] = None,
                  backend: BackendName = 'mpl') -> AtomImage:
    backend = t.cast(BackendName, backend.lower())
    if backend == 'mpl':
        return show_atoms_mpl_2d(atoms, zone=zone, plane=plane, horz=horz)
    elif backend == 'ase':
        raise NotImplementedError()

    raise ValueError(f"Unknown backend '{backend}'")


class AtomImageMpl(AtomImage, Figure):
    def __new__(cls, *args, fig: t.Optional[Figure] = None, **kwargs):
        if fig is not None:
            fig.__class__ = cls
            return fig
        return super().__new__(cls)

    def __init__(self, *args, fig: t.Optional[Figure] = None, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, f: FileOrPath):
        return self.savefig(f)  # type: ignore


_ELEM_MAP = {
    7: [0, 0, 255],    # N
    8: [255, 0, 0],    # O
    13: [255, 215, 0], # Al
    16: [253, 218, 13], # S
    74: [52, 152, 219], # W
}


def get_elem_color(elem: int) -> t.List[int]:
    # grey fallback
    return _ELEM_MAP.get(elem, [80, 80, 80])


class AtomPatch3D(PathPatch3D):
    def __init__(self, elem: int, fc=None, s: float = 2., **kwargs):
        ...
        self.elem = elem
        if fc is None:
            fc = get_elem_color(elem)

        self.s = s

        super().__init__(fc=fc, **kwargs)



def show_atoms_mpl_3d(atoms: AtomCollection, *, fig: t.Optional[Figure] = None,
                      zone: t.Optional[VecLike] = None, plane: t.Optional[VecLike] = None) -> AtomImageMpl:
    if fig is not None:
        fig = AtomImageMpl(fig=fig)
    else:
        fig = t.cast(AtomImageMpl, pyplot.figure(FigureClass=AtomImageMpl))

    (azim, elev) = get_azim_elev(zone, plane)

    rect = [0., 0., 1., 1.]
    ax: Axes3D = fig.add_axes(rect, axes_class=Axes3D, proj_type='ortho', azim=azim, elev=elev)
    ax.grid(False)

    bbox = atoms.bbox().pad(0.2)
    ax.set_xlim3d(bbox.x)
    ax.set_ylim3d(bbox.y)
    ax.set_zlim3d(bbox.z)
    ax.set_box_aspect(bbox.size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    frame = atoms.get_atoms('global')

    coords = frame.coords()
    elem_colors = numpy.array(list(map(get_elem_color, frame['elem']))) / 255.
    s = 100

    if isinstance(atoms, AtomCell):  # TODO raise this API to AtomCollection
        # plot cell corners
        corners = atoms.cell_corners('global')
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

    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=elem_colors, alpha=1, s=s)

    return fig


def show_atoms_mpl_2d(atoms: AtomCollection, *, fig: t.Optional[Figure] = None,
                      zone: t.Optional[VecLike] = None,
                      plane: t.Optional[VecLike] = None,
                      horz: t.Optional[VecLike] = None) -> AtomImageMpl:
    if plane is not None:
        if isinstance(atoms, AtomCell) and not atoms.is_orthogonal:
            # convert plane into zone
            raise NotImplementedError()
        zone = plane
    elif zone is None:
        zone = [0., 0., 1.]

    zone = numpy.broadcast_to(zone, 3)

    if fig is not None:
        fig = AtomImageMpl(fig=fig)
    else:
        fig = t.cast(AtomImageMpl, pyplot.figure(FigureClass=AtomImageMpl))

    rect = [0.05, 0.05, 0.95, 0.95]
    ax: Axes = fig.add_axes(rect)
    ax.set_aspect('equal')

    frame = atoms.get_atoms('global')
    coords = frame.coords()
    elem_colors = numpy.array(list(map(get_elem_color, frame['elem']))) / 255.

    transform = LinearTransform.align(zone, horz)
    bbox_2d = transform @ atoms.bbox()
    coords_2d = (transform @ coords)[..., :2]

    s = 3.
    ax.set_xbound(*bbox_2d.x)
    ax.set_ybound(*bbox_2d.y)
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=elem_colors, alpha=1, s=s)

    return fig


def get_azim_elev(zone: t.Optional[VecLike] = None, plane: t.Optional[VecLike] = None) -> t.Tuple[float, float]:
    if zone is not None and plane is not None:
        raise ValueError("'zone' and 'plane' can't both be specified.")
    if plane is None:
        if zone is None:
            zone = [1., 2., 4.]

        (a, b, c) = zone
        l = numpy.sqrt(a**2 + b**2)
        return (numpy.angle(a + b*1.j, deg=True), numpy.angle(l + c*1.j, deg=True))  # type: ignore

    raise NotImplementedError()
        