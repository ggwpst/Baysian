import pymc3 as pm
import arviz as az
import numpy as np
from scipy.stats import bernoulli

np.random.seed(123)
theta_real = 63/97
data = bernoulli.rvs(p = theta_real, size = 97)

with pm.Model() as model_a:
    theta = pm.Beta('theta', alpha=0.01, beta=0.01)
    y = pm.Bernoulli('y', p=theta, observed=data)
    
    trace = pm.sample(1000, random_seed = 123)


az.plot_posterior(trace)
az.summary(trace)