import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from scipy import stats
from scipy.special import expit as logistic

iris = pd.read_csv('iris.csv')
df = iris.query("species == ('setosa', 'versicolor')") 
df = df[22:78]
y_3 = pd.Categorical(df['species']).codes 
x_n = ['sepal_length', 'sepal_width'] 
x_3 = df[x_n].values
varnames = ['α', 'β'] 

with pm.Model() as model_3: 
    α = pm.Normal('α', mu=0, sd=10) 
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n)) 
    μ = α + pm.math.dot(x_3, β) 
    θ = 1 / (1 + pm.math.exp(-μ)) 
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_3[:,0]) 
    yl = pm.Bernoulli('yl', p=θ, observed=y_3) 
    trace_3 = pm.sample(1000)

az.plot_trace(trace_3, varnames)

idx = np.argsort(x_3[:,0]) 
bd = trace_3['bd'].mean(0)[idx]
plt.scatter(x_3[:,0], x_3[:,1], c= [f'C{x}' for x in y_3]) 
plt.plot(x_3[:,0][idx], bd, color='k')
az.plot_hpd(x_3[:,0], trace_3['bd'], color='k')
plt.xlabel(x_n[0]) 
plt.ylabel(x_n[1])