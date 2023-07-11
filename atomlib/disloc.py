"""
A collection of functions for inserting dislocations into structures.
"""
from __future__ import annotations

import logging
import warnings
import typing as t
from typing import cast

import numpy
from numpy.typing import NDArray, ArrayLike
import polars
from scipy.special import ellipe, ellipk, elliprf, elliprj

from .atomcell import VecLike, to_vec3
from .transform import AffineTransform3D, LinearTransform3D
from .atoms import Atoms, HasAtomsT, _selection_to_expr
from .vec import norm, dot, perp, split_arr, polygon_solid_angle, polygon_winding


def ellip_pi(n: NDArray[numpy.float_], m: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
    """
    Complete elliptic integral of the third kind, :math:`\\Pi(n | m)`.

    Follows the definition of `Wolfram Mathworld`_.
    
    .. _Wolfram Mathworld: https://mathworld.wolfram.com/EllipticIntegraloftheThirdKind.html
    """
    y = 1 - m
    assert numpy.all(y > 0)

    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3


CutType = t.Literal['shift', 'add', 'rm']


def disloc_edge(atoms: HasAtomsT, center: VecLike, b: VecLike, t: VecLike, cut: t.Union[CutType, VecLike] = 'shift',
                *, poisson: float = 0.25) -> HasAtomsT:
    r"""
    Add a Volterra edge dislocation to the structure.

    The dislocation will pass through ``center``, with line vector ``t`` and Burgers vector ``b``.
    ``t`` will be modified such that it is perpendicular to ``b``.

    The ``cut`` parameter defines the cut (discontinuity) plane used to displace the atoms.
    By default, ``cut`` is ``'shift'``, which defines the cut plane to contain ``b``, ensuring no atoms
    need to be added or removed.
    Instead, ``'add'`` or ``'rm'`` may be specified, which defines the cut plane as containing
    :math:`\mathbf{b} \times \mathbf{t}`. In this mode, atoms will be added or removed
    in the plane of `b` to create the dislocation. Alternatively, a vector ``cut``
    may be supplied. The cut plane will contain this vector.

    In the coordinate system where ``t`` is along ``z``, and ``b`` is along ``+x``, the displacement
    due to the edge dislocation can be calculated as follows:

    .. math::
       u_x &= \frac{b}{2\pi} \left( \arctan(x, y) + \frac{x y}{2(x^2 + y^2)(1-\nu)} \right) \\
       u_y &= -\frac{b}{4(1-\nu)} \left( (1-2\nu) \ln(x^2 + y^2) + \frac{x^2 - y^2}{x^2 + y^2} \right)

    Where ``x`` and ``y`` are distances from the dislocation center. This creates a discontinuity along
    the ``-x`` axis. This coordinate system is rotated to support branches along an arbitrary axis.

    The dislocation is defined by the FS/RH convention: Performing one loop of positive sense
    relative to the tangent vector displaces the real crystal one ``b`` relative to a reference
    crystal.
    """

    center = to_vec3(center)
    b_vec = to_vec3(b)
    #b_mag = norm(b_vec)

    # get component of t perpendicular to b, normalize
    t = to_vec3(t)
    t = perp(t, b)
    if norm(t) < 1e-10:
        raise ValueError("`b` and `t` must be different.")
    t /= norm(t)

    if isinstance(cut, str):
        cut = cast(CutType, cut.lower())
        if cut == 'shift':
            plane_v = b_vec.copy()
        elif cut == 'add':
            # FS/RH convention: t x b points to extra half plane
            plane_v = numpy.cross(t, b_vec)
        elif cut == 'rm':
            plane_v = -numpy.cross(t, b_vec)
        else:
            raise ValueError(f"Unknown cut plane type `{cut}`. Expected 'shift', 'add', 'rm', or a vector.")
        plane_v /= norm(plane_v)
    else:
        plane_v = to_vec3(cut)
        plane_v = plane_v / norm(plane_v)
        if numpy.linalg.norm(numpy.cross(plane_v, t)) < 1e-10:
            raise ValueError('`cut` and `t` must be different.')

    # translate center to 0., and align t to [0, 0, 1], plane to +y
    transform = AffineTransform3D.translate(center).inverse().compose(
        LinearTransform3D.align_to(t, [0., 0., 1.], plane_v, [-1., 0., 0.])
    )
    frame = atoms.get_atoms('local').transform(transform)
    b_vec = transform.transform_vec(b_vec)

    d = numpy.dot(b_vec, [0., 1., 0.])
    if -d > 1e-8:
        logging.info("Removing atoms.")
        old_len = len(frame)
        frame = frame.filter(~(
            (polars.col('x') < 0) & (polars.col('y') >= d/2.) & (polars.col('y') <= -d/2.)
        ))
        logging.info(f"Removed {old_len - len(frame)} atoms")
    if d > 1e-8:
        logging.info("Duplicating atoms.")
        duplicate = frame.filter(
            (polars.col('x') < 0) & (polars.col('y') >= -d/2.) & (polars.col('y') <= d/2.)
        )
        logging.info(f"Duplicated {len(duplicate)} atoms")

        frame = Atoms.concat((frame, duplicate))
        #atoms = atoms._replace_atoms(frame)
        branch = numpy.ones(len(frame), dtype=float)
        if len(duplicate) > 0:
            branch[-len(duplicate):] *= -1  # flip branch of duplicated atoms
    else:
        branch = numpy.ones(len(frame), dtype=float)

    pts = frame.coords()

    x, y, _ = split_arr(pts, axis=-1)
    r2 = x**2 + y**2

    # displacement parallel and perpendicular to b
    d_para = branch * numpy.arctan2(y, x) + x*y/(2*(1-poisson)*r2)
    d_perp = -(1-2*poisson)/(4*(1-poisson)) * numpy.log(r2) + (y**2 - x**2)/(4*(1-poisson)*r2)

    disps = numpy.stack([
        d_para * b_vec[0] + d_perp * b_vec[1],
        d_perp * b_vec[0] + d_para * b_vec[1],
        numpy.zeros_like(x)
    ], axis=-1) / (2*numpy.pi)

    return atoms.with_atoms(frame.with_coords(pts + disps).transform(transform.inverse()), 'local')


def disloc_screw(atoms: HasAtomsT, center: VecLike, b: VecLike, cut: t.Optional[VecLike] = None,
                 sign: bool = True) -> HasAtomsT:
    r"""
    Add a Volterra screw dislocation to the structure.

    The dislocation will pass through ``center``, with Burgers vector ``b``.

    The ``cut`` parameter defines the cut (discontinuity) plane used to displace the atoms.
    By default, ``cut`` is chosen automtically, but it may also be specified as a vector
    which points from the dislocation core towards the cut plane (not normal to the cut plane!)

    The screw dislocation in an isotropic medium has a particularily simple form, given by:

    .. math::
       \mathbf{u} = \frac{\mathbf{b}}{2\pi} \arctan(x, y)

    for a dislocation along ``+z`` with cut plane along ``-x``. To support arbitrary cut planes,
    ``x`` and ``y`` are replaced by the components of ``r`` parallel and perpendicular to the cut plane,
    evaluated in the plane of ``b``.

    The dislocation is defined by the FS/RH convention: Performing one loop of positive sense
    relative to the tangent vector displaces the real crystal one ``b`` relative to a reference
    crystal.
    """

    center = to_vec3(center)
    b_vec = to_vec3(b)
    t = b_vec / float(numpy.linalg.norm(b_vec))
    t = -t if not sign else t
    if cut is None:
        if numpy.linalg.norm(numpy.cross(t, [1., 1., 1.])) < numpy.pi/4:
            # near 111, choose x as cut plane direction
            cut = to_vec3([1., 0., 0.])
        else:
            # otherwise find plane by rotating around 111
            cut = cast(NDArray[numpy.float_], LinearTransform3D.rotate([1., 1., 1.], 2*numpy.pi/3).transform(t))
    else:
        cut = to_vec3(cut)
        cut /= norm(cut)
        if numpy.allclose(cut, t, atol=1e-2):
            raise ValueError("`t` and `cut` must be different.")

    print(f"Cut plane direction: {cut}")

    frame = atoms.get_atoms('local')
    pts = frame.coords() - center

    # components perpendicular to t
    cut_perp = -perp(cut, t)
    pts_perp = perp(pts, t)

    # signed angle around dislocation
    theta = numpy.arctan2(dot(t, numpy.cross(cut_perp, pts_perp)), dot(cut_perp, pts_perp))
    # FS/RH convention
    disp = b_vec * (theta / (2*numpy.pi))

    return atoms.with_atoms(frame.with_coords(pts + center + disp), 'local')


def disloc_loop_z(atoms: HasAtomsT, center: VecLike, b: VecLike,
                  loop_r: float, *, poisson: float = 0.25) -> HasAtomsT:
    r"""
    Add a square dislocation loop to the structure, assuming an elastically isotropic material.

    The loop will have radius ``loop_r``, be centered at ``center``, and oriented along the z-axis.

    The dislocation's sign is defined such that traveling upwards through the loop results in a displacement of ``b``.
    ``poisson`` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of :math:`\mathbf{b} \cdot \mathbf{n}A`, where :math:`\mathbf{n}A` is the loop's
    normal times its area. Consequently, this function adds or remove atoms to effect this volume change.
    """

    center = to_vec3(center)
    b_vec = to_vec3(b)

    atoms = atoms.transform_atoms(AffineTransform3D.translate(center).inverse())
    frame = atoms.get_atoms('local')
    branch = None

    d = numpy.dot(b_vec, [0, 0, 1])
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

        frame = Atoms.concat((frame, duplicate))
        #atoms = atoms._replace_atoms(frame)
        branch = numpy.sign(frame['z'].to_numpy())
        if len(duplicate) > 0:
            branch[-len(duplicate):] *= -1  # flip branch of duplicated atoms

    pts = frame.coords()
    disps = _loop_disp_z(pts, b_vec, loop_r, poisson=poisson, branch=branch)

    return atoms.with_atoms(frame.with_coords(pts + disps + center), 'local')


def disloc_square_z(atoms: HasAtomsT, center: VecLike, b: VecLike,
                    loop_r: float, *, poisson: float = 0.25) -> HasAtomsT:
    r"""
    Add a square dislocation loop to the structure, assuming an elastically isotropic material.

    The dislocation's sign is defined such that traveling upwards through the loop results in a displacement of ``b``.
    ``poisson`` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of :math:`\mathbf{b} \cdot \mathbf{n}A`, where :math:`\mathbf{n}A` is the loop's
    normal times its area. Consequently, this function adds or remove atoms to effect this volume change.
    """
    poly = loop_r * numpy.array([(1, 1), (-1, 1), (-1, -1), (1, -1)])
    return disloc_poly_z(atoms, b, poly, center, poisson=poisson)


def disloc_poly_z(atoms: HasAtomsT, b: VecLike, poly: ArrayLike, center: t.Optional[VecLike] = None,
                  *, poisson: float = 0.25) -> HasAtomsT:
    r"""
    Add a dislocation loop defined by the polygon ``poly``, assuming an elastically isotropic material.

    ``poly`` should be a 2d polygon (shape ``(N, 2)``). It will be placed at ``center``, in the plane ``z=center[2]``.
    For CCW winding order, traveling upwards through the plane of the loop results in a displacement of ``b``.
    ``poisson`` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of :math:`\mathbf{b} \cdot \mathbf{n}A`, where :math:`\mathbf{n}A` is the loop's
    normal times its area. Consequently, this function adds or remove atoms to effect this volume change.

    Follows the solution in `Hirth, J. P. & Lothe, J. (1982). Theory of Dislocations. ISBN 0-89464-617-6
    <https://www.google.com/books/edition/Theory_of_Dislocations/TAjwAAAAMAAJ>`_
    """
    center = to_vec3(center if center is not None else [0., 0., 0.])
    b_vec = to_vec3(b)

    poly = numpy.atleast_2d(poly)
    if poly.ndim != 2 or poly.shape[-1] != 2:
        raise ValueError(f"Expected a 2d polygon. Instead got shape {poly.shape}")

    frame = atoms.get_atoms('local')
    coords: NDArray[numpy.float_] = frame.coords() - center

    branch = None
    d = numpy.dot(b_vec, [0, 0, 1])
    if abs(d) > 1e-8:
        logging.info("Non-conservative dislocation.")
        windings = polygon_winding(poly, coords[..., :2])

        z = coords[..., 2]
        remove = (z >= windings * d/2.) & (z <= -windings * d/2.)
        duplicate = (z >= -windings * d/2.) & (z <= windings * d/2.)

        n_remove = numpy.sum(remove, dtype=int)
        if n_remove:
            logging.info(f"Removing {n_remove} atoms")
            frame = frame.filter(_selection_to_expr(~remove))

            duplicate = duplicate[~remove]

        n_duplicate = numpy.sum(duplicate, dtype=int)
        if n_duplicate:
            logging.info(f"Duplicating {n_duplicate} atoms")
            frame = Atoms.concat((frame, frame.filter(duplicate)))

            branch = numpy.ones(len(frame))
            branch[-n_duplicate:] = -1  # flip branch of duplicated atoms

        coords = frame.coords() - center

    disp = _poly_disp_z(coords, b_vec, poly, poisson=poisson, branch=branch)

    return atoms.with_atoms(frame.with_coords(coords + disp + center), 'local')


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

    def _disp(r: NDArray[numpy.float_]) -> NDArray[numpy.float_]:
        r_norm = numpy.linalg.norm(r, axis=-1, keepdims=True)
        disps = (1-2*poisson)*numpy.cross(b_vec, eta) * numpy.log(r_norm + dot(r, eta)) - dot(b_vec, e2) * numpy.cross(r / r_norm, e2)
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


__all__ = [
    'disloc_edge', 'disloc_screw', 'disloc_loop_z', 'disloc_poly_z', 'disloc_square_z',
    'ellip_pi', 'CutType',
]
