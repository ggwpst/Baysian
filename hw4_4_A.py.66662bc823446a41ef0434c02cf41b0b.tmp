import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats
from scipy.special import expit as logistic
np.random.seed(123)

iris = pd.read_csv('iris.csv')
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
varnames = ['α', 'β', 'bd']

for feature in ["sepal_length", "petal_width", "petal_length"]:
    x_n = feature
    x_0 = df[x_n].values
    x_c = x_0 - x_0.mean()
    with pm.Model() as model_0:
        α = pm.Normal('α', mu=0, sd=10)
        β = pm.Normal('β', mu=0, sd=10)
        μ = α + pm.math.dot(x_c, β)    
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β)
        yl = pm.Bernoulli('yl', p=θ, observed=y_0)
        trace_0 = pm.sample(1000)
        print("Feature {} summary".format(feature))
        print(az.summary(trace_0, varnames, credible_interval=.95))

for nu in [1, 10, 30]:
    x_0 = df["petal_length"].values
    x_c = x_0 - x_0.mean()
    with pm.Model() as model_0:
        # Priors have been changed
        α = pm.StudentT('α', nu=nu, mu=0, sd=10)
        β = pm.StudentT('β', nu=nu, mu=0, sd=10)
        μ = α + pm.math.dot(x_c, β)    
        θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
        bd = pm.Deterministic('bd', -α/β)
        yl = pm.Bernoulli('yl', p=θ, observed=y_0)
        trace_0 = pm.sample(1000)
        print(f"Feature {feature} nu {nu} summary")
        print(az.summary(trace_0, varnames, credible_interval=.95))