"""
Calculate the gravitational effects of a spherical shell and half-shell.

Calculates the effects on the polar axis.

But due to the symmetry of the shell, the values can be considered to be at any
latitude, longitude pair.

Components, gx, gy, gxy, gxz, and gyz are all equal to zero (0).

**Functions for the shell:**

* :func:`~fatiando.gravmag.spherical_shell.potential`: the gravitational
  potential
* :func:`~fatiando.gravmag.spherical_shell.gz`: the vertical component of the
  gravitational attraction
* :func:`~fatiando.gravmag.spherical_shell.gxx`: the xx (North-North) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.gyy`: the yy (East-East) component
  of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.gzz`: the zz (radial-radial)
  component of the gravity gradient tensor

**Functions for the half-shell:**

* :func:`~fatiando.gravmag.spherical_shell.half_potential`: the gravitational
  potential
* :func:`~fatiando.gravmag.spherical_shell.half_gz`: the vertical component of
  the gravitational attraction
* :func:`~fatiando.gravmag.spherical_shell.half_gxx`: the xx (North-North)
  component of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.half_gyy`: the yy (East-East)
   component of the gravity gradient tensor
* :func:`~fatiando.gravmag.spherical_shell.half_gzz`: the zz (radial-radial)
  component of the gravity gradient tensor

"""
from __future__ import division
import numpy

from fatiando.constants import MEAN_EARTH_RADIUS, G, SI2MGAL, SI2EOTVOS


def half_potential(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    """
    The potential of such a shell is:

    $V(r) = 2\pi G \rho \left[ \dfrac{l^3 + {r'}^3}{3r} - 0.5 {r'}^2 \right]
    \Biggr\rvert_{r'=r_1}^{r'=r_2} $

    where $\rho$ is the dentity, $l = \sqrt{r^2 + {r'}^2}$, $r$ is the radius
    coordinate of the observation point, $r_1$ and $r_2$ are the bottom and top
    of the spherical shell, respectively.
    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r ** 2 + rl ** 2)
        res += (-1) ** (i) * ((l ** 3 + rl ** 3) / (3. * r) - 0.5 * rl ** 2)
    res *= 2 * numpy.pi * G * density
    return res


def half_gz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    """
    The $g_z$ component of the gravitational attraction for this shell is the
    radial derivative of the potential.

    The sign should be inverted because our convention is that z points down
    (toward the center of the Earth)

    $g_z(r) = -\dfrac{\partial V}{\partial r} = 2\pi G \rho \left[ \dfrac{l^3 +
    {r'}^3}{3r^2} - l \right] \Biggr\rvert_{r'=r_1}^{r'=r_2}$
    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r ** 2 + rl ** 2)
        res += (-1) ** (i) * ((l ** 3 + rl ** 3) / (3. * r ** 2) - l)
    res *= 2 * numpy.pi * G * density * SI2MGAL
    return res


def half_gzz(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    """
    The gravity gradient tensor are the second derivatives of $V$:

    $g_{xx}(r) = -\dfrac{1}{2} g_{zz}$

    $g_{xy}(r) = 0$

    $g_{xz}(r) = 0$

    $g_{yy}(r) = -\dfrac{1}{2} g_{zz}$

    $g_{yz}(r) = 0$

    $g_{zz}(r) = 2\pi G \rho \left[ 2\dfrac{l^3 + {r'}^3}{3r^3} - \dfrac{l}{r}
    + \dfrac{r}{l} \right] \Biggr\rvert_{r'=r_1}^{r'=r_2}$
    """
    r = heights + radius
    r1 = bottom + radius
    r2 = top + radius
    res = numpy.zeros_like(r)
    for i, rl in enumerate([r2, r1]):
        l = numpy.sqrt(r ** 2 + rl ** 2)
        res += (-1) ** (i) * \
            (2. * (l ** 3 + rl ** 3) / (3. * r ** 3) - l / r + r / l)
    res *= 2 * numpy.pi * G * density * SI2EOTVOS
    return res


def half_gxx(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    return -0.5 * gzz(heights, top, bottom, density, radius)


def half_gyy(heights, top, bottom, density, radius=MEAN_EARTH_RADIUS):
    return gxx(heights, top, bottom, density, radius)
