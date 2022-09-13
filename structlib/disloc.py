import logging
import warnings
import typing as t

import numpy
import polars
from scipy.special import ellipe, ellipk, elliprf, elliprj

from . import AtomFrame
from .core import AtomCollectionT, VecLike, to_vec3
from .transform import AffineTransform, FuncTransform
from .util import split_arr


def ellip_pi(n, m):
    """Complete elliptic integral of the third kind"""
    y = 1 - m
    assert numpy.all(y > 0)

    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3


def _loop_disp_z(pts, b_vec: numpy.ndarray, loop_r: float,
                 branch: t.Optional[numpy.ndarray] = None,
                 poisson: float = 0.25) -> numpy.ndarray:
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

    (b1, b2, b3) = b_vec
    return numpy.stack([
        b1/rho**2 * (f1*x**2 + f2*y**2) + b2*x*y/rho**2 * (f1 - f2) + b3*x/rho * g1,
        b2/rho**2 * (f1*y**2 + f2*x**2) + b1*x*y/rho**2 * (f1 - f2) + b3*y/rho * g1,
        b1*x/rho * f3 + b2*y/rho * f3 + b3 * g3
    ], axis=-1)


def disloc_loop_z(atoms: AtomCollectionT, center: VecLike, b: VecLike,
                  loop_r: float, poisson: float = 0.25) -> AtomCollectionT:
    center = to_vec3(center)
    b_vec = to_vec3(b)

    atoms = atoms.transform_atoms(AffineTransform.translate(center).inverse())
    frame = atoms.get_atoms('local')
    branch = None

    d = numpy.dot(b_vec.view(numpy.ndarray), [0, 0, 1])
    if d > 1e-8:
        logging.info("Non-conservative dislocation. Removing atoms.")
        frame = frame.filter(~(
            (polars.col('x')**2 + polars.col('y')**2 < loop_r**2)
            & (polars.col('z') >= -d/2.) & (polars.col('z') <= d/2.)
        ))

    if -d > 1e-8:
        logging.info("Non-conservative dislocation. Duplicating atoms.")
        duplicate = frame[
            (frame['x']**2 + frame['y']**2 < loop_r**2)
            & (frame['z'] >= d/2.) & (frame['z'] <= -d/2.)
        ]
        logging.info(f"Adding {len(duplicate)} atoms")

        frame = AtomFrame(polars.concat([frame, duplicate]))  # type: ignore
        atoms = atoms._replace_atoms(frame)
        branch = numpy.sign(frame['z'].to_numpy())
        if len(duplicate) > 0:
            branch[-len(duplicate):] *= -1  # flip branch of duplicated atoms

    pts = frame.coords()
    disps = _loop_disp_z(pts, b_vec, loop_r, branch, poisson)

    # clip maximum displacement (is this even needed?)
    #if max_disp is None:
    #    max_disp = float(numpy.linalg.norm(b_vec))
    #disp_r: numpy.ndarray = numpy.linalg.norm(disps, axis=-1)
    #scale = numpy.clip(disp_r, None, max_disp) / disp_r
    #disps *= scale[:, None]

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