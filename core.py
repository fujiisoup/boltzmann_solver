import numpy as np
from scipy import interpolate, integrate, special
from scipy.spatial.transform import Rotation


class DifferentialCrossSection:
    r"""
    Base class for the differential crosssection.

    In this class, we assume that the differential cross section 
    is represented as
    \sigma(v, \theta) = v^{-\lambda} \sum_k c_k P_k(\cos\theta)^2

    where P_k is the legendre function.
    """

    def __init__(self, lam, legendre_coefs, m=10000):
        r"""
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
        r"""
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
            diffpart += coef * pol(np.cos(theta))
        return diffpart ** 2

    def total_crosssection(self, v):
        r"""
        Compute the total cross section, by
        \int \sigma(v, \theta) \sin\theta d\theta
        """
        return self._total_crosssection * v ** (-self.lam)

    def scattering_angle(self, r):
        r"""
        Compute the scattering angle based on random variables r, 
        which is in [0, 1]
        """
        return self._cumsum_sigma(r)


def flag_scattering(u1, u2, rng, differential_crosssection, density, dt):
    r"""
    Compute if the scattering happens during dt
    """
    # relative velocity
    n = u1.shape[0]
    dv = u1 - u2
    speed_rel = np.sqrt(np.sum(dv ** 2, axis=-1))
    probability = (
        differential_crosssection.total_crosssection(speed_rel)
        * speed_rel
        * density
        * dt
    )
    uni = rng.uniform(0, 1, size=n)
    return (probability > uni)[:, np.newaxis]


def scattering(m1, u1, m2, u2, rng, differential_crosssection, density, dt):
    r"""
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
    number_of_collision: integer 
        number of reactions during dt
    """
    n = u1.shape[0]
    # velocity in the center-of-mass system
    vel_cm = (m1 * u1 + m2 * u2) / (m1 + m2)
    u1_cm = u1 - vel_cm
    u2_cm = u2 - vel_cm
    # compute the scattering in the center-of-mass system
    phi = rng.uniform(0, 2 * np.pi, size=n)
    theta = differential_crosssection.scattering_angle(rng.uniform(0, 1, size=n))
    # compute the scattering
    # scattering angle in center-of-mass coordinate. For particle 1. For particle 2, multiply -1.
    rot = Rotation.from_euler("ZX", np.array([phi, theta]).T)
    v1_cm = rot.apply(u1_cm)
    v2_cm = rot.apply(u2_cm)
    v1 = v1_cm + vel_cm
    v2 = v2_cm + vel_cm
    # compute if this scattering happens during dt
    flag = flag_scattering(u1, u2, rng, differential_crosssection, density, dt)
    v1 = np.where(flag, v1, u1)
    v2 = np.where(flag, v2, u2)
    return v1, v2, np.sum(flag)


def optimize_dt(u1, u2, diffsigma, dt_init, rng, change_rate=1.2, target_fraction=0.3):
    n = len(u1)
    target = n * target_fraction
    n_collided = np.sum(flag_scattering(u1, u2, rng, diffsigma, 1.0, dt_init))
    if n_collided < target / change_rate:  # dt is too small
        return optimize_dt(
            u1, u2, diffsigma, dt_init * change_rate, rng, change_rate, target_fraction
        )
    elif n_collided > target * change_rate:  # dt is too large
        return optimize_dt(
            u1, u2, diffsigma, dt_init / change_rate, rng, change_rate, target_fraction
        )
    return dt_init


def thermal_distribution(n, m, T, rng):
    r"""
    Construct the thermal velocity distribution

    Parameters
    ----------
    n: integer
        number of particles
    m: float
        mass
    T: float
        temperature
    rng: np.random.RandomState
    """
    return rng.randn(n, 3) * np.sqrt(T / m)


class BotlzmannBase:
    def __init__(self, n, lam, legendre_coefs, T=1.0, seed=0):
        self.n = int(n / 2) * 2
        self.rng = np.random.RandomState(0)
        self.diffsigma = DifferentialCrossSection(lam, legendre_coefs)
        self.index = np.arange(n)
        self.T = T


class BoltzmannLinear(BotlzmannBase):
    r"""
    A solver for the linear boltzmann's equation, where test particles (with mass m1) 
    collides only with heavier particles (with mass m2).
    The velocity of heavier particles does not change during the collision.
    """

    def __init__(self, n, m1, m2, lam, legendre_coefs, T=1.0, seed=0):
        r"""
        n: integer
            number of particles to be traced
        m1, m2: float
            mass of the test and heavier particles
        lam: float

        legendre_coefs: 1d-array
        T: float
            temperature of the heavier particles. In the unit of energy
        """
        super().__init__(n, lam, legendre_coefs, T, seed)
        self.m1 = m1
        self.m2 = m2
        # initialize with the thermal distribution
        self.v1 = thermal_distribution(n, m1, T, self.rng)
        self.v2 = thermal_distribution(n, m2, T, self.rng)

    def compute(
        self, heating_rate, heating_temperature, nsamples=1000, thin=1, burnin=1000
    ):
        r"""
        Compute the model.

        Parameters
        ----------
        heating_rate: float
            rate of the additional heating for the particle 1
        heating_temperature: float
            temperature of the additional heating.

        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        """
        index = np.arange(self.n)
        histogram = []

        for i in range(-burnin, nsamples * thin):
            if i in [-burnin, 0]:
                # compute the time step
                self.rng.shuffle(index)
                u1 = self.v1[index]
                u2 = self.v2[index]
                dt = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma,
                    1.0,
                    self.rng,
                    change_rate=1.2,
                    target_fraction=0.3,
                )
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            self.rng.shuffle(index)
            self.v1[index[:n_heating]] = thermal_distribution(
                n_heating, self.m1, heating_temperature, self.rng
            )

            self.rng.shuffle(index)
            u1 = self.v1[index]
            u2 = self.v2[index]
            v1, _, _ = scattering(
                self.m1, u1, self.m2, u2, self.rng, self.diffsigma, 1.0, dt
            )
            self.v1[index] = v1
            # overwrite v2 by thermal distribution
            self.v2 = thermal_distribution(self.n, self.m2, self.T, self.rng)

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v1))
        return np.array(histogram)


class BoltzmannNonlinear(BotlzmannBase):
    r"""
    A solver for the nonlinear boltzmann's equation, where test particles (with mass m) 
    collides only with the same-kind particles.
    """

    def __init__(self, n, m, lam, legendre_coefs, T=1.0, seed=0):
        r"""
        n: integer
            number of particles to be traced
        m: float
            mass of the test and heavier particles
        lam: float

        legendre_coefs: 1d-array
        T: float
            initial temperature. In the unit of energy
        """
        super().__init__(n, lam, legendre_coefs, T, seed)
        self.m = m
        # initialize with the thermal distribution
        self.v = thermal_distribution(n, m, T, self.rng)

    def compute(
        self,
        heating_rate,
        heating_temperature,
        cooling_rate,
        nsamples=1000,
        thin=1,
        burnin=1000,
    ):
        r"""
        Compute the model.

        Parameters
        ----------
        heating_rate: float
            rate of the additional heating for the particle 1
        heating_temperature: float
            temperature of the additional heating.

        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        """
        index = np.arange(self.n)
        nhalf = int(self.n / 2)
        histogram = []
        for i in range(-burnin, nsamples * thin):
            if i in [-burnin, 0]:
                # compute the time step
                self.rng.shuffle(index)
                u1 = self.v[index[:nhalf]]
                u2 = self.v[index[nhalf:]]
                dt = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma,
                    1.0,
                    self.rng,
                    change_rate=1.2,
                    target_fraction=0.3,
                )
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            self.rng.shuffle(index)
            self.v[index[:n_heating]] = thermal_distribution(
                n_heating, self.m, heating_temperature, self.rng
            )
            # cooling
            self.v = self.v * np.exp(-cooling_rate * dt)

            self.rng.shuffle(index)
            u1 = self.v[index[:nhalf]]
            u2 = self.v[index[nhalf:]]
            v1, v2, _ = scattering(
                self.m, u1, self.m, u2, self.rng, self.diffsigma, 1.0, dt
            )
            self.v[index[:nhalf]] = v1
            self.v[index[nhalf:]] = v2

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v))
        return np.array(histogram)


class BoltzmannMixture(BoltzmannLinear):
    def compute(
        self,
        heating_rate,
        heating_temperature,
        mixture,
        nsamples=1000,
        thin=1,
        burnin=1000,
    ):
        r"""
        Compute the model.

        Parameters
        ----------
        heating_rate: float
            rate of the additional heating for the particle 1
        heating_temperature: float
            temperature of the additional heating.
        mixture: float in [1, 0]
            mixture rate of the test among all the particles.
            If mixture == 1, then pure test particles are assumed (but the energy will diverge)
        
        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        """
        index = np.arange(self.n)
        nhalf = int(self.n / 2)

        histogram = []
        test_density = mixture
        bath_density = 1.0 - mixture

        for i in range(-burnin, nsamples * thin):
            if i in [-burnin, 0]:
                # compute the time step
                # compute dt from collisions with heat-bath particles
                self.rng.shuffle(index)
                u1 = self.v1[index]
                u2 = self.v2[index]
                dt_bath = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma,
                    bath_density,
                    self.rng,
                    change_rate=1.2,
                    target_fraction=0.3,
                )
                # compute dt from the collisions among the test particles
                u1 = self.v1[index[:nhalf]]
                u2 = self.v1[index[nhalf:]]
                dt_test = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma,
                    test_density,
                    self.rng,
                    change_rate=1.2,
                    target_fraction=0.3,
                )

                # determine dt from the dominant collision
                dt = np.minimum(dt_test, dt_bath)
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            self.rng.shuffle(index)
            self.v1[index[:n_heating]] = thermal_distribution(
                n_heating, self.m1, heating_temperature, self.rng
            )

            # collision with the other particles
            self.rng.shuffle(index)
            u1 = self.v1[index]
            u2 = self.v2[index]
            v1, _, _ = scattering(
                self.m1, u1, self.m2, u2, self.rng, self.diffsigma, bath_density, dt
            )
            self.v1[index] = v1
            # overwrite v2 by thermal distribution
            self.v2 = thermal_distribution(self.n, self.m2, self.T, self.rng)

            # collision among the test particles
            self.rng.shuffle(index)
            u1 = self.v1[index[:nhalf]]
            u2 = self.v1[index[nhalf:]]
            v1, v2, _ = scattering(
                self.m1, u1, self.m1, u2, self.rng, self.diffsigma, test_density, dt
            )
            self.v1[index[:nhalf]] = v1
            self.v1[index[nhalf:]] = v2

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v1))
        return np.array(histogram)
