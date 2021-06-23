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
        
    def compute(
        self, heating_rate, heating_temperature, 
        eta,
        nsamples=1000, thin=1, burnin=1000, reaction_rate_per_step=0.3
    ):
        r"""
        Compute the Boltzmann equation
        """
        # initial distribution
        self.E = thermal_distribution(
            self.n, heating_temperature * 1e-4, self.rng
        )

        index = np.arange(self.n)
        histogram = []
        nhalf = int(self.n / 2)

        n_heating = int(self.n * reaction_rate_per_step)
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
            decay = self.rng.uniform(size=n_cascade)
            self.E[index_cascade] = self.E[index_cascade] * decay

            # randomly choose the collision
            self.rng.shuffle(index)
            index1 = index[:nhalf]
            index2 = index[nhalf:]
            E1 = self.E[index1]
            E2 = self.E[index2]
            K = E1 + E2
            ratio = self.rng.uniform(size=nhalf)
            self.E[index1] = K * ratio
            self.E[index2] = K * (1 - ratio)

            if i > 0 and i % thin == 0:
                histogram.append(np.copy(self.E))

        return np.array(histogram)