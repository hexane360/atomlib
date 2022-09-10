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


def disloc_loop_z(atoms: AtomCollectionT, center: VecLike, b: VecLike,
                  loop_r: float, poisson: float = 0.25) -> AtomCollectionT:
    center = to_vec3(center)
    b_vec = to_vec3(b)
    (b1, b2, b3) = b_vec

    atoms = atoms.transform_atoms(AffineTransform.translate(center).inverse())
    frame = atoms.get_atoms('local')

    d = numpy.dot(b_vec.view(numpy.ndarray), [0, 0, 1])
    if d > 1e-8:
        logging.info("Non-conservative dislocation. Removing atoms.")
        remaining = frame.filter(~(
            (polars.col('x')**2 + polars.col('y')**2 < loop_r**2)
            & (polars.col('z') > 0) & (polars.col('z') < d)
        ))
        atoms = atoms._replace_atoms(remaining)

    def f(pts: numpy.ndarray) -> numpy.ndarray:
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

        with warnings.catch_warnings():
            omega = 2*numpy.pi*numpy.sign(z) - 2/(z*numpy.sqrt(a + b)) * (
                (r**2 + rho * loop_r + r*(rho + loop_r)) * ellip_pi(n1, m) + 
                (r**2 + rho * loop_r - r*(rho + loop_r)) * ellip_pi(n2, m)
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
            -((a**2 - b**2) * (1 - 2*poisson) - a*z**2)*e + (a-b)*(z**2 - a*(1 - 2*poisson))*k
        )

        return pts + numpy.stack([
            b1*(x**2/rho**2 * f1 + y**2/rho**2 * f2) + b2*x*y/rho**2 * (f1 - f2) + b3*x/rho * g1,
            b2*(y**2/rho**2 * f1 + x**2/rho**2 * f2) + b1*x*y/rho**2 * (f1 - f2) + b3*y/rho * g1,
            b1*x/rho * f3 + b2*y/rho * f3 + b3 * g3
        ], axis=-1)

    atoms = atoms.transform_atoms(f)

    if -d > 1e-8:
        logging.info("Non-conservative dislocation. Duplicating atoms.")
        transformed_frame = atoms.get_atoms('local')

        duplicate = transformed_frame[
            (frame['x']**2 + frame['y']**2 < loop_r**2)
            & (frame['z'] > d) & (frame['z'] < 0.)
        ]
        duplicate = duplicate.transform(AffineTransform.translate(b1, b2, b3).inverse())

        logging.info(f"Adding {len(duplicate)} atoms")

        atoms = atoms._replace_atoms(AtomFrame(polars.concat([transformed_frame, duplicate])))

    return atoms.transform_atoms(AffineTransform.translate(center))