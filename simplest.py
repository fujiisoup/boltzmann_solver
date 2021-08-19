import numpy as np


def thermal_distribution(n, T, rng):
    r"""
    Construct the thermal velocity distribution

    Parameters
    ----------
    n: integer
        number of particles
    T: float
        temperature
    rng: np.random.RandomState
    """
    return rng.exponential(scale=T, size=n)


def gaussian_distribution(n, d, T, rng):
    r"""
    Construct the thermal velocity distribution in d-dimension

    Parameters
    ----------
    n: integer
        number of particles
    d: integer
        number of dimensions
    T: float
        temperature
    rng: np.random.RandomState
    """
    return rng.randn(n, d) * np.sqrt(T)


def angular_distribution(n, d, rng):
    r"""
    Make the uniform distribution along the angular direction,
    but along the radial direction, the size should be 1

    Parameters
    ----------
    n: integer
        number of particles
    d: integer
        number of dimensions
    T: float
        temperature
    rng: np.random.RandomState
    """
    v = rng.randn(n, d)
    return v / np.sqrt(np.sum(v**2, axis=-1, keepdims=True))


class SimplestBotlzmann:
    def __init__(self, n, seed=0):
        """
        n: integer
            number of particles to be traced
        seed: float
            seed for random number
        """
        self.n = int(n / 2) * 2
        self.rng = np.random.RandomState(0)
        
    def is_collide(self, E1, E2, beta=None, delta=None, energy_min=None, energy_max=None):
        """
        Find if particles collide

        Parameters
        ----------
        E: average energy divided
        beta: float. The cross section should scale E^beta
        energy_min: minimum energy
        energy_max: maximum energy

        Returns
        -------
        1d array of bool: will collide if True
        """
        if beta is None:
            return np.full_like(E1, True, dtype=bool)

        E = E1 + E2
        if energy_min is None:
            energy_min = np.min(E1 + E2)
        if energy_max is None:
            energy_max = np.max(E1 + E2)
        
        if delta is None:
            coef = 1.0
        else:
            coef = (E1 * E2 * 0.25 / E**2)**delta

        if beta > 0:
            rate = (E / energy_max)**beta * coef
        else:
            rate = (E / energy_min)**beta * coef
        return self.rng.uniform(size=E.shape) < rate


    def compute(
        self, heating_rate, heating_temperature, 
        eta, d=None, beta=None, delta=None,
        nsamples=1000, thin=1, burnin=1000
    ):
        r"""
        Compute the Boltzmann equation
        d: statistical weight as a function of E
        beta: energy dependence of the collision
        """
        # initial distribution
        energy_min = heating_temperature * 1e-5
        self.E = thermal_distribution(
            self.n, heating_temperature * 1e-4, self.rng
        )

        index = np.arange(self.n)
        histogram = []
        nhalf = int(self.n / 2)

        n_heating = int(self.n * heating_rate)
        n_cascade = int(self.n * eta)

        for i in range(-burnin, nsamples * thin):
            # randomly choose the heated particles
            self.rng.shuffle(index)
            index_heating = index[:n_heating]
            self.E[index_heating] = thermal_distribution(
                n_heating, heating_temperature, self.rng)

            # randomly choose the cascading particles
            self.rng.shuffle(index)
            index_cascade = index[:n_cascade]
            if beta is not None:
                is_collide = self.is_collide(
                    self.E[index_cascade], energy_min, beta, delta, energy_min, heating_temperature)
                index_cascade = index_cascade[is_collide]
            decay = self.rng.uniform(size=len(index_cascade))
            self.E[index_cascade] = self.E[index_cascade] * decay

            # randomly choose the collision
            self.rng.shuffle(index)
            index1 = index[:nhalf]
            index2 = index[nhalf:]
            E1 = self.E[index1]
            E2 = self.E[index2]
            K = E1 + E2
            if d is None:
                ratio = self.rng.uniform(size=nhalf)
            else:
                ratio = self.rng.beta(d+1, d+1, size=nhalf)

            if beta is not None:
                is_collide = self.is_collide(
                    E1, E2, beta, delta, energy_min * 2, heating_temperature * 2)
                index1 = index1[is_collide]
                index2 = index2[is_collide]
                ratio = ratio[is_collide]
                K = K[is_collide]

            self.E[index1] = K * ratio
            self.E[index2] = K * (1 - ratio)

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.E))

        return np.array(histogram)


class SimplestDilute(SimplestBotlzmann):
    def compute(
        self, heating_rate, heating_temperature, 
        eta, d=None, beta=None, delta=None,
        nsamples=1000, thin=1, burnin=1000
    ):
        r"""
        Compute the Boltzmann equation
        d: statistical weight as a function of E
        beta: energy dependence of the collision
        """
        # initial distribution
        energy_min = heating_temperature * 1e-5
        self.E = thermal_distribution(
            self.n, heating_temperature * 1e-4, self.rng
        )

        index = np.arange(self.n)
        histogram = []
        nhalf = int(self.n / 2)

        n_heating = int(self.n * heating_rate)
        
        dillute_coef = 1 - eta  # kinetic energy loss rate

        for i in range(-burnin, nsamples * thin):
            # randomly choose the heated particles
            self.rng.shuffle(index)
            index_heating = index[:n_heating]
            self.E[index_heating] = thermal_distribution(
                n_heating, heating_temperature, self.rng)

            # randomly choose the collision
            self.rng.shuffle(index)
            index1 = index[:nhalf]
            index2 = index[nhalf:]
            E1 = self.E[index1]
            E2 = self.E[index2]
            K = (E1 + E2) * dillute_coef
            if d is None:
                ratio = self.rng.uniform(size=nhalf)
            else:
                ratio = self.rng.beta(d+1, d+1, size=nhalf)

            if beta is not None:
                is_collide = self.is_collide(K, beta, delta)
                index1 = index1[is_collide]
                index2 = index2[is_collide]
                ratio = ratio[is_collide]
                K = K[is_collide]

            self.E[index1] = K * ratio
            self.E[index2] = K * (1 - ratio)

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.E))

        return np.array(histogram)


class Levy(SimplestBotlzmann):
    """
    A simple model to mimic the generalized central limit theorem

    Parameters
    ----------
    n: integer. 
        Number of particles to be tracked.
    """

    def compute(
        self, heating_rate, dillute_coef, d, 
        heating_temperature=1, beta=None, delta=None,
        nsamples=1000, thin=1, burnin=1000
    ):
        r"""
        Compute the Boltzmann equation
        with beta = 0, it reduces to the central limit theorem

        d: statistical weight as a function of E
        beta: energy dependence of the collision
        """
        # initial distribution
        energy_min = heating_temperature * 1e-5
        self.v = gaussian_distribution(
            self.n, d, heating_temperature * 1e-4, self.rng
        )

        index = np.arange(self.n)
        histogram = []
        nhalf = int(self.n / 2)

        n_heating = int(self.n * heating_rate)

        for i in range(-burnin, nsamples * thin):
            # randomly choose the heated particles
            self.rng.shuffle(index)
            index_heating = index[:n_heating]
            self.v[index_heating] = gaussian_distribution(
                n_heating, d, heating_temperature, self.rng)

            # randomly choose the collision
            self.rng.shuffle(index)
            index1 = index[:nhalf]
            index2 = index[nhalf:]
            v1 = self.v[index1]
            v2 = self.v[index2]
            v = (v1 - v2)

            if beta is not None:
                K = np.sqrt(np.sum(v**2, axis=-1))
                is_collide = self.is_collide(K, beta, delta)
                index1 = index1[is_collide]
                index2 = index2[is_collide]
                v = v[is_collide]

            v = v / np.sqrt(2 / dillute_coef)
            self.v[index1] = v
            self.v[index2] = v

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.v))

        return np.array(histogram)
