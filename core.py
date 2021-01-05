import numpy as np
from scipy import interpolate, integrate, special
from scipy.spatial.transform import Rotation


class DifferentialCrossSection:
    """
    Base class for the differential crosssection.

    In this class, we assume that the differential cross section 
    is represented as
    \sigma(v, \theta) = v^{-\lambda} \sum_k c_k P_k(\cos\theta)^2

    where P_k is the legendre function.
    """
    def __init__(self, lam, legendre_coefs, m=10000):
        """
        lam: scalar
        legendre_coefs: 1d-array
        m: int. default 10000
            number of point to interpolate the scattering angle
        """
        self.lam = lam
        self.legendre_coefs = legendre_coefs
        self.m = m
        self._prepare()
        
    def _prepare(self):
        """
        Prepare the interp1d instance, 
        self._cumsum_sigma
        and the proportional coefficient for the total crosssection
        self._total_crosssection
        """
        # cumsum_sigma
        theta = np.linspace(0, np.pi, self.m)
        diffpart = self.differential_part(theta)
        cumsum = integrate.cumtrapz(diffpart, theta, initial=0)
        cumsum /= cumsum[-1]  # normalize to one
        self._cumsum_sigma = interpolate.interp1d(cumsum, theta)

        # total crossection
        def func(theta):
            return np.sin(theta) * self.differential_part(theta)
        
        self._total_crosssection = integrate.quad(func, 0, np.pi)[0]
            
    def differential_part(self, theta):
        diffpart = np.zeros_like(theta)
        for i, coef in enumerate(self.legendre_coefs):
            pol = special.legendre(i)
            diffpart += coef * np.poly1d(pol)(np.cos(theta))**2
        return diffpart

    def total_crosssection(self, v):
        """ 
        Compute the total cross section, by
        \int \sigma(v, \theta) \sin\theta d\theta
        """
        return self._total_crosssection * v**(-self.lam)

    def scattering_angle(self, r):
        """
        Compute the scattering angle based on random variables r, 
        which is in [0, 1]
        """
        return self._cumsum_sigma(r)


def scattering(
    m1, u1, m2, u2, rng, differential_crosssection, density, dt
):
    """
    Compute the collision process among two particles

    Parameters
    ----------
    m1, m2: scalar
        mass of two particles
    u1, u2: 2d-array shaped (n, 3), indicating the velocities in the laboratory frames
    rng: np.random.RandomState instance
    differential_crosssection: 
        An instance of DifferentialCrossSection
    ft: scalar
        Considered timestep
    
    Returns
    -------
    v1, v2: post-collision velocities
    """
    n = u1.shape[0]
    # velocity in the center-of-mass system
    vel_cm = (m1 * u1 + m2 * u2) / (m1 + m2)
    # relative velocity
    dv = u1 - u2
    speed_rel = np.sqrt(np.sum(dv**2, axis=-1))
    u1_cm = u1 - vel_cm
    u2_cm = u2 - vel_cm
    # compute the scattering in the center-of-mass system
    phi = rng.uniform(0, 2 * np.pi, size=n)
    theta = differential_crosssection.scattering_angle(rng.uniform(0, 1, size=n))
    # compute the scattering
    # scattering angle in center-of-mass coordinate. For particle 1. For particle 2, multiply -1.
    rot = Rotation.from_euler('ZX', np.array([phi, theta]).T)
    v1_cm = rot.apply(u1_cm)
    v2_cm = rot.apply(u2_cm)
    v1 = v1_cm + vel_cm
    v2 = v2_cm + vel_cm
    # compute if this scattering happens during dt
    uni = rng.uniform(0, 1, size=n)
    probability = differential_crosssection.total_crosssection(speed_rel) * speed_rel * density * dt
    v1 = np.where((probability > uni)[:, np.newaxis], v1, u1)
    v2 = np.where((probability > uni)[:, np.newaxis], v2, u2)
    return v1, v2


class BoltzmannBase:
    """
    A class for 0d-Boltzmann equation based on monte-carlo integration
    """
    def __init__(self, n, seed=0):
        """
        n: number of particles
        """
        # velocity of each particles
        self.v = np.zeros((n, 3))
        self.n = n
        self.rng = np.random.RandomState(0)

    def update(self, dt, m=1):
        """
        update the collision term
        """
        # choose a pair of particles at random
        i0, i1 = self.rng(self.n, (2, m), replace=False)
        # the relative velocity of the selected pairs
        vrel = self.v[i0] - self.v[i1]

    def crosssection(self, v):
        """
        An abstract method for collision operator.
        
        """
        raise NotImplementedError
    
