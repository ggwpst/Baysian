import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from IPython.display import SVG, display
from pandas.plotting import parallel_coordinates
from scipy import stats
from theano import shared, tensor

data = stats.norm(4,.5).rvs(size=57)

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 10)
    sd = pm.HalfNormal("sd", 25)
    y = pm.Normal("y,", mu, sd, observed=data)
    # Compute both prior, and prior predictive
    prior_predictive = pm.sample_prior_predictive()
    # Compute posterior
    trace = pm.sample()
    # Compute posterior predictive
    posterior_predictive = pm.sample_posterior_predictive(trace)

dataset = az.from_pymc3(trace=trace, posterior_predictive=posterior_predictive, prior=prior_predictive)
dataset

az.plot_posterior(dataset.prior, var_names=["mu", "sd"])

az.plot_posterior(dataset)

dataset.prior

print(dataset.prior_predictive["y,"].values.shape)
prior_predictive = dataset.prior_predictive["y,"].values.flatten()
prior_predictive.shape

az.plot_kde(prior_predictive)

az.plot_ppc(dataset)