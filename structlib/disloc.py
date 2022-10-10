import logging
import warnings
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike
import polars
from scipy.special import ellipe, ellipk, elliprf, elliprj

from .core import AtomCollectionT, VecLike, to_vec3
from .transform import AffineTransform, FuncTransform
from .util import split_arr, polygon_solid_angle, polygon_winding
from .frame import AtomFrame, _selection_to_expr


def ellip_pi(n, m):
    """Complete elliptic integral of the third kind"""
    y = 1 - m
    assert numpy.all(y > 0)

    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3


def disloc_loop_z(atoms: AtomCollectionT, center: VecLike, b: VecLike,
                  loop_r: float, *, poisson: float = 0.25) -> AtomCollectionT:
    """
    Displace the structure, adding a circular dislocation loop of radius `loop_r` centered at `center`.

    The dislocation's sign is defined such that traveling upwards through the loop results in a displacement of `b`.
    `poisson` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of `b dot nA`, where `nA` is the loop's normal times its area.
    Conequently, this function will add or remove atoms to compensate for this volume change.
    """
    center = to_vec3(center)
    b_vec = to_vec3(b)

    atoms = atoms.transform_atoms(AffineTransform.translate(center).inverse())
    frame = atoms.get_atoms('local')
    branch = None

    d = numpy.dot(b_vec.view(numpy.ndarray), [0, 0, 1])
    if -d > 1e-8:
        logging.info("Non-conservative dislocation. Removing atoms.")
        frame = frame.filter(~(
            (polars.col('x')**2 + polars.col('y')**2 < loop_r**2)
            & (polars.col('z') >= d/2.) & (polars.col('z') <= -d/2.)
        ))

    if d > 1e-8:
        logging.info("Non-conservative dislocation. Duplicating atoms.")
        duplicate = frame.filter(
            (polars.col('x')**2 + polars.col('y')**2 < loop_r**2)
            & (polars.col('z') >= -d/2.) & (polars.col('z') <= d/2.)
        )
        logging.info(f"Adding {len(duplicate)} atoms")

        frame = AtomFrame(polars.concat([frame, duplicate]))  # type: ignore
        #atoms = atoms._replace_atoms(frame)
        branch = numpy.sign(frame['z'].to_numpy())
        if len(duplicate) > 0:
            branch[-len(duplicate):] *= -1  # flip branch of duplicated atoms

    pts = frame.coords()
    disps = _loop_disp_z(pts, b_vec, loop_r, poisson=poisson, branch=branch)

    """
    # debugging
    from matplotlib import pyplot
    fig, axs = pyplot.subplots(ncols=3)
    axs[0].hist(disps[:, 0])
    axs[1].hist(disps[:, 1])
    axs[2].hist(disps[:, 2])
    pyplot.show()

    i_max_x = numpy.argmax(disps[:, 0])
    print(f"maximum x displacement: {disps[i_max_x]} @ {pts[i_max_x]}")
    """

    return atoms._replace_atoms(frame.with_coords(pts + disps + center), 'local')


def disloc_poly_z(atoms: AtomCollectionT, b: VecLike, poly: ArrayLike, center: t.Optional[VecLike] = None,
                  *, poisson: float = 0.25) -> AtomCollectionT:
    """
    Displace the structure, adding a dislocation loop defined by the polygon `poly`.

    `poly` should be a 2d polygon (shape (N, 2)). It will be placed at `center`, in the plane `z=center[2]`.
    For CCW winding order, traveling upwards through the plane of the loop results in a displacement of `b`.
    `poisson` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of `b dot nA`, where `nA` is the loop's normal times its area.
    this function will add or remove atoms to compensate for this volume change.
    """
    center = to_vec3(center or [0., 0., 0.])
    b_vec = to_vec3(b)

    poly = numpy.atleast_2d(poly)
    if poly.ndim != 2 or poly.shape[-1] != 2:
        raise ValueError(f"Expected a 2d polygon. Instead got shape {poly.shape}")

    frame = atoms.get_atoms('local')
    coords: NDArray[numpy.float_] = frame.coords()
    coords = coords - center

    branch = None
    d = numpy.dot(b_vec.view(numpy.ndarray), [0, 0, 1])
    if abs(d) > 1e-8:
        logging.info("Non-conservative dislocation.")
        windings = polygon_winding(poly, coords[..., :2])

        z = coords[..., 2]
        remove = (z >= windings * d/2.) & (z <= -windings * d/2.)
        n_remove = numpy.sum(remove, dtype=int)
        if n_remove:
            logging.info(f"Removing {n_remove} atoms")
            frame = frame.filter(_selection_to_expr(~remove))

        duplicate = (z >= -windings * d/2.) & (z <= windings * d/2.)
        n_duplicate = numpy.sum(duplicate, dtype=int)
        if n_duplicate:
            logging.info(f"Duplicating {n_duplicate} atoms")
            frame = AtomFrame(polars.concat([frame, frame.filter(polars.lit(duplicate, dtype=polars.Boolean))]))

            branch = numpy.ones(len(frame))
            branch[-n_duplicate:] = -1  # flip branch of duplicated atoms

        coords = frame.coords()
        coords = coords - center

    disp = _poly_disp_z(coords, b_vec, poly, poisson=poisson, branch=branch)

    """
    # Debugging
    frame = frame.with_columns((
        polars.lit(disp[..., 0]).alias('disp_x'),
        polars.lit(disp[..., 1]).alias('disp_y'),
        polars.lit(disp[..., 2]).alias('disp_z'),
        polars.lit(numpy.linalg.norm(disp, axis=-1)).alias('disp'),
    ))
    print(frame.filter(polars.col('disp') > 3.*numpy.linalg.norm(b_vec)))
    """

    return atoms._replace_atoms(frame.with_coords(coords + disp + center), 'local')



def _dot(v1: NDArray[numpy.float_], v2: NDArray[numpy.float_], keepdims: bool = True) -> NDArray[numpy.float_]:
        return numpy.add.reduce(v1 * v2, axis=-1, keepdims=keepdims)


def _poly_disp_z(pts: NDArray[numpy.float_], b_vec: NDArray[numpy.float_], poly: NDArray[numpy.float_], *,
                 poisson: float = 0.25, branch: t.Optional[numpy.ndarray] = None) -> NDArray[numpy.float_]:

    if branch is None:
        branch = numpy.array(1.)

    omega = branch * polygon_solid_angle(poly, pts)

    poly = numpy.concatenate((poly, numpy.zeros_like(poly, shape=(*poly.shape[:-1], 1))), axis=-1)
    r = poly - pts[..., None, :]
    r_n = numpy.roll(r, -1, axis=-2)

    eta = r_n - r
    eta /= numpy.linalg.norm(eta, axis=-1, keepdims=True)
    e2 = numpy.cross(eta, r)
    e2 /= numpy.linalg.norm(e2, axis=-1, keepdims=True)

    def _disp(r):
        r_norm = numpy.linalg.norm(r, axis=-1, keepdims=True)
        disps = (1-2*poisson)*numpy.cross(b_vec, eta) * numpy.log(r_norm + _dot(r, eta)) - _dot(b_vec, e2) * numpy.cross(r / r_norm, e2)
        return numpy.sum(disps, axis=-2)

    return b_vec * omega[:, None] / (4*numpy.pi) + 1/(8*numpy.pi*(1-poisson)) * (_disp(r_n) - _disp(r))


def _loop_disp_z(pts: NDArray[numpy.float_], b_vec: numpy.ndarray, loop_r: float, *,
                 poisson: float = 0.25, branch: t.Optional[numpy.ndarray] = None) -> numpy.ndarray:
    rho = numpy.linalg.norm(pts[..., :2], axis=-1)
    r = numpy.linalg.norm(pts, axis=-1)
    (x, y, z) = split_arr(pts, axis=-1)

    a = r**2 + loop_r**2
    b = 2*rho*loop_r
    m = 2*b/(a + b)

    with warnings.catch_warnings():
        n1 = 2*rho / (rho - r)
        n1 = numpy.where(numpy.abs(n1) < 1e-10, -numpy.inf, n1)

    n2 = 2*rho / (rho + r)

    k = ellipk(m)
    e = ellipe(m)

    if branch is None:
        branch = numpy.sign(z)

    with warnings.catch_warnings():
        omega = 2*numpy.pi*branch - 2/(z*numpy.sqrt(a + b)) * (
            (r**2 + rho*loop_r + r*(rho + loop_r)) * ellip_pi(n1, m) + 
            (r**2 + rho*loop_r - r*(rho + loop_r)) * ellip_pi(n2, m)
        )
        omega = numpy.where(numpy.abs(z) < 1e-10, 0., omega)

    f1 = -omega/(4*numpy.pi) + z / (4*numpy.pi*(1-poisson)*b**2*(a-b)*numpy.sqrt(a+b)) * (
        (2*loop_r**2*(2*a**2 - b**2) - a*b**2)*e + (a-b)*(b**2 - 4*a*loop_r**2)*k
    )
    g3 = -omega/(4*numpy.pi) + z / (4*numpy.pi*(1-poisson)*(a-b)*numpy.sqrt(a+b)) * (
        (a - 2*loop_r**2)*e - (a - b)*k
    )
    f2 = -omega/(4*numpy.pi) + z*loop_r**2 / (numpy.pi*(1-poisson)*b**2*numpy.sqrt(a+b)) * (
        -(a + b)*e + a*k
    )
    f3 = loop_r / (2*numpy.pi*(1-poisson)*b*(a-b)*numpy.sqrt(a+b)) * (
        ((a**2 - b**2) * (1 - 2*poisson) - a*z**2)*e + (a-b)*(z**2 - a*(1 - 2*poisson))*k
    )
    g1 = loop_r / (2*numpy.pi*(1-poisson)*b*(a-b)*numpy.sqrt(a+b)) * (
        ((b**2 - a**2) * (1 - 2*poisson) - a*z**2)*e + (a-b)*(z**2 + a*(1 - 2*poisson))*k
    )

    (b1, b2, b3) = -b_vec
    return numpy.stack([
        b1/rho**2 * (f1*x**2 + f2*y**2) + b2*x*y/rho**2 * (f1 - f2) + b3*x/rho * g1,
        b2/rho**2 * (f1*y**2 + f2*x**2) + b1*x*y/rho**2 * (f1 - f2) + b3*y/rho * g1,
        b1*x/rho * f3 + b2*y/rho * f3 + b3 * g3
    ], axis=-1)