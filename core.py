import warnings
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
        cumsum = cumsum / cumsum[-1]  # normalize to one
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

    def scattering_angle(self, u_rel, r):
        r"""
        Compute the scattering angle based on relative velocity and 
        random variables r, which is in [0, 1]
        """
        return self._cumsum_sigma(r)


def g_Lewkow(x):
    """
    Analytical form of the elastic cross section
    Lewkow, N. R., Kharchenko, V., & Zhang, P. (2012). 
    ENERGY RELAXATION OF HELIUM ATOMS IN ASTROPHYSICAL GASES. 
    The Astrophysical Journal, 756(1), 57. https://doi.org/10.1088/0004-637X/756/1/57

    where 
    x = E theta
    with
    E: in eV
    theta: in degree
    """
    a1, a2, a3, a4 = -0.136, 0.993, -3.042, 5.402
    return 10 ** (a1 * np.log10(x) ** 3 + a2 * np.log10(x) ** 2 + a3 * np.log10(x) + a4)


class CoulombCrossSection:
    r"""
    Differential cross section for coulomb scattering.

    We assume the differential cross section is represented as
    \sigma(E, \theta) = \frac{1}{\theta} g(x)

    where x = E \theta and g(x) is some positive function
    """

    def __init__(self, g, m=10000, xmin=1e2):
        r"""
        g: functional
            A function gives the differential cross section.
            See g_Lewkow as an example
        xmin: float
            minimum x values to compute the differential cross section.
        """
        self.g = g
        self.m = m
        self.xmin = xmin
        self._prepare()

    def _prepare(self):
        # cumsum_sigma
        x = np.logspace(np.log10(self.xmin), 5, self.m)
        g_values = self.g(x)
        cumsum = integrate.cumtrapz(g_values / x, x, initial=0)
        cumsum = cumsum / cumsum[-1]  # normalize to one
        self._cumsum_sigma = interpolate.interp1d(cumsum, x)

        # total crossection
        self._total_crosssection = integrate.quad(
            lambda x: self.g(x) / x, self.xmin, np.inf
        )[0]

    def total_crosssection(self, v):
        return self._total_crosssection

    def scattering_angle(self, u_rel, r):
        theta = self._cumsum_sigma(r) / u_rel ** 2  # in deg
        return np.minimum(theta / 180.0 * np.pi, np.pi)


class IsotropicCrossSections:
    """
    Crosssection for Isotropic scattering
    """
    def __init__(self, lam=-2/3, *args, **kwargs):
        self.lam = lam

    def total_crosssection(self, v):
        return v**self.lam

    def scattering_angle(self, u_rel, r):
        return r * np.pi
        
    def viscosity(self, u_rel, restitution_coef=None):
        """
        analytical viscosity cross section
        """
        if restitution_coef is not None:
            raise NotImplementedError
        return 1 / 2 * u_rel**self.lam

    def momentum_transfer(self, u_rel, restitution_coef=None):
        """Returns momentum crosssection """
        if restitution_coef is not None:
            raise NotImplementedError
        return u_rel**self.lam


class HardSphereCrossSections:
    """
    Crosssection for the hard sphere
    """
    def __init__(self, lam, restrict_2d=False, *args, **kwargs):
        self.lam = lam
        self.restrict_2d = restrict_2d

    def total_crosssection(self, v):
        return v**self.lam

    def scattering_angle(self, u_rel, r, restrict_2d=False):
        """
        differential cross section is proportional to ~sin(theta)
        """
        if self.restrict_2d:
            return np.pi - 2 * np.arcsin(r)
        else:
            return np.pi - 2 * np.arcsin(np.sqrt(r))

    def viscosity(self, u_rel, restitution_coef=None):
        """
        analytical viscosity cross section
        """
        if restitution_coef is None:
            if self.restrict_2d:
                raise NotImplementedError
            else: 
                return 2 / 3 * u_rel**self.lam
        if self.restrict_2d:
            raise NotImplementedError
        else:
            return 1 / 2 + restitution_coef / 6  
            # TODO find better approximation

    def momentum_transfer(self, u_rel, restitution_coef=None):
        """Returns momentum crosssection """
        if restitution_coef is not None:
            raise NotImplementedError
        if self.restrict_2d:
            raise NotImplementedError
        return 1.0 * u_rel**self.lam


class TheoreticalCrossSections:
    """
    Differential crosssection based on theory
    """

    def __init__(
        self, data, effective_mass=1.0, m=10000, 
        extrapolate_modes=(-1/3, 'log')
    ):
        """
        data: xr.dataarray
            dimensions should be ['energy', 'angle'] 
        effective_mass: m * M / (m + M)
        """
        import xarray as xr
        
        _data = (
            data.sortby("energy")
            .sortby("angle")
            .transpose("energy", "angle")
            .isel(angle=slice(None, -1))
        )

        if extrapolate_modes[0] != 'log':
            # using power scaling
            data0 = _data.isel(energy=0)
            scale = 0.01
            data0['energy'] = data0['energy'] * scale
            data0 = data0 * scale**extrapolate_modes[0]
            _data = xr.concat([data0, _data], dim='energy')            
        elif extrapolate_modes[1] != 'log':
            # using power scaling
            data0 = _data.isel(energy=-1)
            scale = 100
            data0['energy'] = data0['energy'] * scale
            data0 = data0 * scale**extrapolate_modes[1]
            _data = xr.concat([_data, data0], dim='energy') 
        self._data = _data.transpose("energy", "angle")

        self.effective_mass = effective_mass
        self.m = m
        self._prepare()

    def _prepare(self):
        total_crosssection = self._data.integrate("angle")
        # maybe better to avoid using xarray
        self._log_total_crosssection = interpolate.interp1d(
            np.log(total_crosssection["energy"].values),
            np.log(total_crosssection.values),
            bounds_error=False,
            fill_value='extrapolate',
            kind="linear",
            assume_sorted=True,
        )
        # angular part
        angular = integrate.cumtrapz(
            self._data.values, self._data["angle"].values, axis=-1, initial=0.0
        )
        x = np.sin(np.linspace(0, np.pi / 2, num=self.m)) ** 2
        cumsigma = []
        for i in range(len(angular)):
            cumsigma.append(
                np.interp(x, angular[i] / angular[i, -1], self._data["angle"].values)
            )
        self._cumsum_sigma = interpolate.RegularGridInterpolator(
            (np.log(self._data["energy"].values), x),
            cumsigma,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def log_Ecm(self, v):
        return np.log(v ** 2 * (self.effective_mass / 2))

    def total_crosssection(self, v):
        return np.exp(self._log_total_crosssection(self.log_Ecm(v)))

    def scattering_angle(self, u_rel, r):
        return self._cumsum_sigma(np.stack([self.log_Ecm(u_rel), r], axis=-1))


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


def _inelastic_scattering_angle(phi, restitution_coef):
    """
    Given the scattering angle phi for the elastic case,
    returns the actual scattering angle for the inelastic case
    """
    cos_theta = np.sin(phi / 2)
    return np.arccos(
        np.clip(
            -cos_theta * restitution_coef
            / (np.sqrt(1 + (restitution_coef**2 - 1) * cos_theta) + 1e-20)  # add a jitter
            , -1, 1
        )
    ) - (np.pi - phi) / 2


def scattering(m1, u1, m2, u2, rng, differential_crosssection, density, dt, restitution_coef=None, restrict_2d=False):
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
    restrict_2d: boolean
        if True, the scattering occurs only on the x-y plane
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
    # relative velocity
    u_rel = np.sqrt(np.sum(u1 ** 2 + u2 ** 2, axis=-1))
    theta = differential_crosssection.scattering_angle(u_rel, rng.uniform(0, 1, size=n))

    # compute the scattering
    if restitution_coef is not None:
        phi = theta  # scattering angle if elastic
        theta = _inelastic_scattering_angle(phi, restitution_coef)
        # velocity reduction rate
        rate = np.sqrt(
            1 + (restitution_coef**2 - 1) * (1 - np.cos(phi)) / 2
        )[..., np.newaxis]
        u1_cm *= rate
        u2_cm *= rate

    # scattering angle in center-of-mass coordinate for particle 1. For particle 2, multiply -1.
    if restrict_2d:
        rot = Rotation.from_euler("Z", theta)
    else:
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


def optimize_dt(
    u1,
    u2,
    diffsigma,
    density,
    rng,
    change_rate=1.2,
    reaction_rate_per_step=0.3,
    dt_init=1.0,
):
    n = len(u1)
    target = n * reaction_rate_per_step
    n_collided = np.sum(flag_scattering(u1, u2, rng, diffsigma, density, dt_init))
    if n_collided < target / change_rate:  # dt is too small
        return optimize_dt(
            u1,
            u2,
            diffsigma,
            density,
            rng,
            change_rate,
            reaction_rate_per_step,
            dt_init=dt_init * change_rate,
        )
    elif n_collided > target * change_rate:  # dt is too large
        return optimize_dt(
            u1,
            u2,
            diffsigma,
            density,
            rng,
            change_rate,
            reaction_rate_per_step,
            dt_init=dt_init / change_rate,
        )
    return dt_init


def thermal_distribution(n, m, T, rng, shape=None, restrict_2d=False):
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
    shape: optional, float
        Assuming Gamma-like distribution with shape parameter. If shape=1.5, it should be a Maxwell distribution
    restrict_2d: boolean
        if True, the scattering occurs only on the x-y plane
    """
    v = rng.randn(n, 3) * np.sqrt(T / m)
    if restrict_2d:
        v[:, -1] = 0  # z-component is always zero

    if shape is None:
        return v
    else:
        v2_abs = np.sum(v**2, axis=-1, keepdims=True)
        v1 = rng.gamma(scale=2 * T / m, shape=shape, size=(n, 1))
        v = v * np.sqrt(v1 / v2_abs / (shape / 1.5))
        if restrict_2d:
            v[:, -1] = 0  # z-component is always zero
        return v


class BotlzmannBase:
    def __init__(self, n, differential_crosssection, T=1.0, seed=0):
        self.n = int(n / 2) * 2
        self.rng = np.random.RandomState(0)
        self.diffsigma = differential_crosssection
        self.T = T

    def _heat_index(self, v, n_heating, heating_weight_index):
        if heating_weight_index is None:
            return self.rng.choice(len(v), n_heating)
        
        weight = np.sum(v**2, axis=-1)**heating_weight_index
        weight = weight / np.sum(weight)
        return self.rng.choice(len(v), n_heating, p=weight)


class BoltzmannLinear(BotlzmannBase):
    r"""
    A solver for the linear boltzmann's equation, where test particles (with mass m1) 
    collides only with heavier particles (with mass m2).
    The velocity of heavier particles does not change during the collision.
    """

    def __init__(self, n, m1, m2, differential_crosssection, T=1.0, seed=0):
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
        super().__init__(n, differential_crosssection, T, seed)
        self.m1 = m1
        self.m2 = m2
        # initialize with the thermal distribution
        self.v1 = thermal_distribution(n, m1, T, self.rng)
        self.v2 = thermal_distribution(n, m2, T, self.rng)

    def compute(
        self,
        heating_rate,
        heating_temperature,
        heating_weight_index=None,
        nsamples=1000,
        thin=1,
        burnin=1000,
        reaction_rate_per_step=0.3,
        heating_shape=None,
        restrict_2d=False
    ):
        r"""
        Compute the model.

        Parameters
        ----------
        heating_rate: float
            rate of the additional heating for the particle 1
        heating_temperature: float
            temperature of the additional heating.
        heating_weight_index:
            choose the heated particle depending on E^{index}

        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        heating_shape: optional. None or float
            Controls the velocity distribution of generated radicals. 
            If None or 1, the heating shape is Maxwellian. 
        """
        index = np.arange(self.n)
        histogram = []
        if restrict_2d:
            self.v1[:, -1] = 0
            self.v2[:, -1] = 0

        time = 0.0
        times = []
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
                    reaction_rate_per_step=reaction_rate_per_step,
                )
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            index_heating = self._heat_index(
                self.v1, n_heating, heating_weight_index)
            self.v1[index_heating] = thermal_distribution(
                n_heating, self.m1, heating_temperature, self.rng, shape=heating_shape,
                restrict_2d=restrict_2d,
            )

            self.rng.shuffle(index)
            u1 = self.v1[index]
            u2 = self.v2[index]
            v1, _, _ = scattering(
                self.m1, u1, self.m2, u2, self.rng, self.diffsigma, 1.0, dt,
                restrict_2d=restrict_2d,
            )
            self.v1[index] = v1
            # overwrite v2 by thermal distribution
            self.v2 = thermal_distribution(self.n, self.m2, self.T, self.rng,
                restrict_2d=restrict_2d,)

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v1))
                times.append(time)
            time += dt
        return np.array(histogram), times


class BoltzmannMixture(BoltzmannLinear):
    def __init__(
        self,
        n,
        m1,
        m2,
        differential_crosssection,
        differential_crosssection_test=None,
        T=1.0,
        seed=0,
    ):
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
        super().__init__(n, m1, m2, differential_crosssection, T=T, seed=seed)
        if differential_crosssection_test is not None:
            self.diffsigma_test = differential_crosssection_test
        else:
            self.diffsigma_test = self.diffsigma

    def compute(
        self,
        heating_rate,
        heating_temperature,
        mixture,
        heating_weight_index=None,
        nsamples=1000,
        thin=1,
        burnin=1000,
        reaction_rate_per_step=0.3,
        heating_shape=None,
        restrict_2d=False,
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
        heating_weight_index:
            choose the heated particle depending on E^{index}

        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        heating_shape: optional. None or float
            Controls the velocity distribution of generated radicals. 
            If None or 1, the heating shape is Maxwellian. 
        """
        if restrict_2d and not hasattr(self.diffsigma, 'restrict_2d'):
            warnings.warn('This cross section does not support restrict_2d option')
        self.diffsigma.restrict_2d = restrict_2d

        index = np.arange(self.n)
        nhalf = int(self.n / 2)
        if restrict_2d:
            self.v1[:, -1] = 0
            self.v2[:, -1] = 0

        histogram = []
        test_density = 1.0
        bath_density = mixture
        # heating rate should be proportional to bath density
        heating_rate = heating_rate * bath_density

        time = 0.0
        times = []
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
                    reaction_rate_per_step=reaction_rate_per_step,
                )
                # compute dt from the collisions among the test particles
                u1 = self.v1[index[:nhalf]]
                u2 = self.v1[index[nhalf:]]
                dt_test = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma_test,
                    test_density,
                    self.rng,
                    change_rate=1.2,
                    reaction_rate_per_step=reaction_rate_per_step
                )

                # determine dt from the dominant collision
                dt = np.minimum(dt_test, dt_bath)
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            index_heating = self._heat_index(
                self.v1, n_heating, heating_weight_index)
            self.v1[index_heating] = thermal_distribution(
                n_heating, self.m1, heating_temperature, self.rng, shape=heating_shape,
                restrict_2d=restrict_2d
            )

            # collision with the other particles
            self.rng.shuffle(index)
            u1 = self.v1[index]
            u2 = self.v2[index]
            v1, _, _ = scattering(
                self.m1, u1, self.m2, u2, self.rng, self.diffsigma, bath_density, dt,
                restrict_2d=restrict_2d
            )
            self.v1[index] = v1
            # overwrite v2 by thermal distribution
            self.v2 = thermal_distribution(self.n, self.m2, self.T, self.rng,
                restrict_2d=restrict_2d)

            # collision among the test particles
            self.rng.shuffle(index)
            u1 = self.v1[index[:nhalf]]
            u2 = self.v1[index[nhalf:]]
            v1, v2, _ = scattering(
                self.m1, u1, self.m1, u2, self.rng, self.diffsigma_test, test_density, dt,
                restrict_2d=restrict_2d
            )
            self.v1[index[:nhalf]] = v1
            self.v1[index[nhalf:]] = v2

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v1))
                times.append(time)
            time += dt
        return np.array(histogram), times


class BoltzmannDissipative(BotlzmannBase):
    def __init__(
        self,
        n,
        m,
        differential_crosssection,
        T=1.0,
        seed=0,
    ):
        r"""
        n: integer
            number of particles to be traced
        m: float
            mass of the particles

        T: float
            temperature of the heavier particles. In the unit of energy
        """
        super().__init__(n, differential_crosssection, T=T, seed=seed)
        self.m = m
        self.v = thermal_distribution(n, m, T, self.rng)

    def compute(
        self,
        heating_rate,
        heating_temperature,
        restitution_coef,
        heating_weight_index=None,
        nsamples=1000,
        thin=1,
        burnin=1000,
        reaction_rate_per_step=0.3,
        heating_shape=None,
        restrict_2d=False,
    ):
        r"""
        Compute the model.

        Parameters
        ----------
        heating_rate: float
            rate of the additional heating for the particle 1
        heating_temperature: float
            temperature of the additional heating.
        restitutison_coef: float in [1, 0]
            restitutison rate of the collision.
            If restitutison == 1, then pure elastic collision is assumed
        heating_weight_index:
            choose the heated particle depending on E^{index}

        nsamples: integer
            number of samples to be stored
        thin: integer
            number of skip
        burnin: integer:
            number of samples to be used to make the system in the equilibrium
        heating_shape: optional. None or float
            Controls the velocity distribution of generated radicals. 
            If None or 1, the heating shape is Maxwellian. 
        """
        # make sure restrict_2d can be used for the crosssection
        if restrict_2d and not hasattr(self.diffsigma, 'restrict_2d'):
            warnings.warn('This cross section does not support restrict_2d option')
        self.diffsigma.restrict_2d = restrict_2d

        index = np.arange(self.n)
        nhalf = int(self.n / 2)
        if restrict_2d:
            self.v[:, -1] = 0

        histogram = []
        heating_rate = heating_rate

        time = 0.0
        times = []
        for i in range(-burnin, nsamples * thin):
            if i in [-burnin, 0]:
                # compute the time step
                # compute dt from the collisions among the test particles
                u1 = self.v[index[:nhalf]]
                u2 = self.v[index[nhalf:]]
                dt = optimize_dt(
                    u1,
                    u2,
                    self.diffsigma,
                    1,
                    self.rng,
                    change_rate=1.2,
                    reaction_rate_per_step=reaction_rate_per_step,
                )
                n_heating_rate = dt * heating_rate * self.n

            if n_heating_rate < 10:  # if small use the probabilistic method
                n_heating = np.minimum(self.rng.poisson(n_heating_rate), self.n)
            else:
                n_heating = np.minimum(int(n_heating_rate), self.n)

            # randomly choose the heated particles
            index_heating = self._heat_index(
                self.v, n_heating, heating_weight_index)
            self.v[index_heating] = thermal_distribution(
                n_heating, self.m, heating_temperature, self.rng, shape=heating_shape,
                restrict_2d=restrict_2d
            )

            # collision among the test particles
            self.rng.shuffle(index)
            u1 = self.v[index[:nhalf]]
            u2 = self.v[index[nhalf:]]
            v1, v2, _ = scattering(
                self.m, u1, self.m, u2, self.rng, self.diffsigma, 1.0, dt, 
                restitution_coef=restitution_coef,
                restrict_2d=restrict_2d
            )
            self.v[index[:nhalf]] = v1
            self.v[index[nhalf:]] = v2

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v))
                times.append(time)
            time += dt
        return np.array(histogram), times
