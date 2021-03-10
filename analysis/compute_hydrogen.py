from multiprocessing import Pool
import numpy as np
import xarray as xr

import sys

sys.path.append("../")
import core


def compute_mixture(heating_rate, mixture, n=3000, nsample=1000, T=3.0e-2):
    diffsigma = core.TheoreticalCrossSections(
        xr.load_dataarray('../crosssections/H-H2/elastic_differential.nc'),
        effective_mass=2.0 / 3.0
    )
    diffsigma_test = core.TheoreticalCrossSections(
        xr.load_dataarray('../crosssections/H-H/elastic_differential.nc'),
        effective_mass=1.0 / 2.0
    )    # parameters
    m1 = 1.0
    m2 = 1.0
    # T = 3.0e-2
    heating_temperature = 3.0

    if mixture == 0:
        model = core.BoltzmannLinear(
            n=n, m1=m1, m2=m2, T=T, differential_crosssection=diffsigma,
        )
        args = {}
    else:
        model = core.BoltzmannMixture(
            n=n,
            m1=m1,
            m2=m2,
            T=T,
            differential_crosssection=diffsigma,
            differential_crosssection_test=diffsigma_test,
        )
        args = {"mixture": mixture}

    result, time = model.compute(
        heating_rate=heating_rate,
        heating_temperature=heating_temperature,
        nsamples=nsample,
        thin=3,
        burnin=nsample * 3,
        **args
    )
    vsq = np.sum(result ** 2, axis=-1) * m1 / 2

    Emax = heating_temperature / m1 * 10
    Emin = Emax * 1e-5
    E_bins = np.logspace(np.log10(Emin), np.log10(Emax), num=51, base=10)
    E_center = (E_bins[:-1] + E_bins[1:]) / 2
    E_size = E_bins[1:] - E_bins[:-1]

    hist_E = [np.histogram(vsq1, bins=E_bins)[0] / E_size / len(vsq1) for vsq1 in vsq]

    vx = np.abs(result[..., 0])
    vy = np.abs(result[..., 1])
    vz = np.abs(result[..., 2])
    vmax = np.sqrt(heating_temperature / m1 * 10)
    vmin = vmax * 1e-3
    v_bins = np.logspace(np.log10(vmin), np.log10(vmax), num=51, base=10)
    v_center = (v_bins[:-1] + v_bins[1:]) / 2
    v_size = v_bins[1:] - v_bins[:-1]

    hist_v = [
        (
            np.histogram(vx1, bins=v_bins)[0]
            + np.histogram(vz1, bins=v_bins)[0]
            + np.histogram(vy1, bins=v_bins)[0]
        )
        / v_size
        / len(vx1)
        / 3
        for vx1, vy1, vz1 in zip(vx, vy, vz)
    ]
    result = xr.Dataset(
        {"hist_E": (("sample", "E"), hist_E), "hist_v": (("sample", "v"), hist_v),},
        coords={
            "E": E_center,
            "v": v_center,
            "n": n,
            "m1": m1,
            "m2": m2,
            "T": T,
            "heating_rate": heating_rate,
            "heating_temperature": heating_temperature,
            "mixture": mixture,
            "time": ("sample", time),
        },
        attrs={"type": "coulomb"},
    )
    return result.mean('sample')


heating_rates = np.logspace(-2, 2, 41)
mixture_ratios = np.logspace(-3, 2, 31)[::-1]
mixture_ratios = np.concatenate([[0], 1 / (1 + mixture_ratios)])
T2 = [1e-4, 0.03]

histograms_all2 = []
for T in T2:
    histograms_all = []
    for heating_rate in heating_rates:
        print("working at {}".format(heating_rate))

        def compute(mixture_ratio):
            return compute_mixture(
                heating_rate,
                mixture_ratio,
                n=10000,
                nsample=1000,
                T=T,
                # mixture_ratio, n=1000, nsample=10, T=T,
            )

        with Pool(8) as p:
            histograms = p.map(compute, mixture_ratios)
        histograms_all.append(xr.concat(histograms, dim="mixture"))
    histograms_all2.append(xr.concat(histograms_all, dim="heating_rate"))
histograms_all2 = xr.concat(histograms_all2, dim="T")
histograms_all2.to_netcdf("hydrogen.nc")
