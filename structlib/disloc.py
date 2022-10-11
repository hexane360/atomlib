import logging
import warnings
import typing as t
from typing import cast

import numpy
from numpy.typing import NDArray, ArrayLike
import polars
from scipy.special import ellipe, ellipk, elliprf, elliprj

from .core import AtomCollectionT, VecLike, to_vec3
from .transform import AffineTransform, LinearTransform
from .util import split_arr, polygon_solid_angle, polygon_winding
from .frame import AtomFrame, _selection_to_expr


def ellip_pi(n, m):
    """Complete elliptic integral of the third kind"""
    y = 1 - m
    assert numpy.all(y > 0)

    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3


CutType = t.Union[t.Literal['shift'], t.Literal['add'], t.Literal['rm']]


def disloc_edge(atoms: AtomCollectionT, center: VecLike, b: VecLike, t: VecLike, cut: t.Union[CutType, VecLike] = 'shift',
                *, poisson: float = 0.25) -> AtomCollectionT:
    """
    Displace the structure, adding an edge dislocation which passes through `center`, with
    line vector `t` and burger's vector `b`. `t` will be modified so it is perpendicular to `b`.

    The `cut` parameter defines the cut (discontinuity) plane used to displace the atoms.
    By default, `cut` is `'shift'`, which defines the cut plane to contain `b`, ensuring no atoms
    need to be added or removed.
    Instead, `add` or `rm` may be specified, which defines the cut plane as containing `b x t`.
    this mode, atoms will be added or removed to create the dislocation.
    Alternatively, a vector `cut` may be supplied. The cut plane will contain this vector.

    The dislocation is defined by the FS/RH convention: Performing one loop of positive sense
    relative to the tangent vector displaces the real crystal one `b` relative to a reference
    crystal.
    """

    center = to_vec3(center)
    b_vec = to_vec3(b)
    b_mag = numpy.linalg.norm(b_vec)

    # get component of t perpendicular to b, normalize
    t = to_vec3(t)
    t = to_vec3(t - t * (t / t.norm()).view(numpy.ndarray).dot(b / b_mag))  # type: ignore
    if t.norm() < 1e-10:
        raise ValueError("`b` and `t` must be different.")
    t = t / t.norm()

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
        plane_v /= numpy.linalg.norm(plane_v)
    else:
        plane_v = to_vec3(cut)
        plane_v = plane_v / numpy.linalg.norm(plane_v.view(numpy.ndarray))
        if numpy.linalg.norm(numpy.cross(plane_v.view(numpy.ndarray), t)) < 1e-10:
            raise ValueError('`cut` and `t` must be different.')

    # translate center to 0., and align t to [0, 0, 1], plane to +y
    transform = AffineTransform.translate(center).inverse().compose(
        LinearTransform.align_to(t, [0., 0., 1.], plane_v, [-1., 0., 0.])
    )
    frame = atoms.get_atoms('local').transform(transform)
    b_vec = transform.transform_vec(b_vec)

    d = numpy.dot(b_vec.view(numpy.ndarray), [0., 1., 0.])
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

        frame = AtomFrame(polars.concat([frame, duplicate]))  # type: ignore
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

    return atoms._replace_atoms(frame.with_coords(pts + disps).transform(transform.inverse()), 'local')


def disloc_screw(atoms: AtomCollectionT, center: VecLike, b: VecLike, cut: t.Optional[VecLike] = None,
                 sign: bool = True) -> AtomCollectionT:
    """
    Displace the structure, adding a screw dislocation which passes through `center` with
    burger's vector `b`.

    The `cut` parameter defines the cut (discontinuity) plane used to displace the atoms.
    By default, `cut` is chosen automtically, but it may also be specified as a vector
    which points from the dislocation core towards the cut plane (not normal to the cut plane!)

    The dislocation is defined by the FS/RH convention: Performing one loop of positive sense
    relative to the tangent vector displaces the real crystal one `b` relative to a reference
    crystal.
    """

    b_vec = to_vec3(b)
    t = b_vec / float(numpy.linalg.norm(b_vec))
    t = -t if not sign else t
    if cut is None:
        if numpy.linalg.norm(numpy.cross(t, [1., 1., 1.])) < numpy.pi/4:
            # near 111, choose x as cut plane direction
            cut = to_vec3([1., 0., 0.])
        else:
            # otherwise find plane by rotating around 111
            cut = LinearTransform.rotate([1., 1., 1.], 2*numpy.pi/3).transform(t)
    else:
        cut = to_vec3(cut)
        cut = cut / float(numpy.linalg.norm(cut))
        if numpy.allclose(cut, t, atol=1e-2):
            raise ValueError("`t` and `cut` must be different.")

    print(f"Cut plane direction: {cut}")

    frame = atoms.get_atoms('local')
    pts = frame.coords() - center

    # components perpendicular to t
    cut_perp = -(cut - t * _dot(cut, t))
    pts_perp = pts - t * _dot(pts, t)

    # signed angle around dislocation
    theta = numpy.arctan2(_dot(t, numpy.cross(cut_perp, pts_perp)), _dot(cut_perp, pts_perp))
    # FS/RH convention
    disp = b_vec * (theta / (2*numpy.pi))

    return atoms._replace_atoms(frame.with_coords(pts + center + disp), 'local')


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


def disloc_square_z(atoms: AtomCollectionT, center: VecLike, b: VecLike,
                    loop_r: float, *, poisson: float = 0.25) -> AtomCollectionT:
    """
    Displace the structure, adding a square dislocation loop of radius `loop_r` centered at `center`.

    The dislocation's sign is defined such that traveling upwards through the loop results in a displacement of `b`.
    `poisson` is the material's poisson ratio, which affects the shape of dislocations with an edge component.

    Adding the loop creates (or removes) a volume of `b dot nA`, where `nA` is the loop's normal times its area.
    Consequently, this function will add or remove atoms to compensate for this volume change.
    """
    poly = loop_r * numpy.array([(1, 1), (-1, 1), (-1, -1), (1, -1)])
    return disloc_poly_z(atoms, b, poly, center, poisson=poisson)


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
        return numpy.add.reduce((v1 * v2).view(numpy.ndarray), axis=-1, keepdims=keepdims)


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