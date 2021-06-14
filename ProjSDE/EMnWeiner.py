#%% 
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10

#%%
dt = 0.1
T = 1
times = np.arange(0,T+dt,dt)
W = np.zeros(len(times))
X = np.zeros(len(times))
W[0] = 0
X[0] = 1
mu = 1.5
sig = 1
for i in range(1,len(W)):
    W[i] = W[i-1] + np.sqrt(dt)*np.random.randn(1)

def F(x,mu): return mu*x 
def G(x,sig): return sig*x 

for i in range(1,len(X)):
    X[i] = X[i-1] + F(X[i-1],mu)*dt + G(X[i-1],sig)*(W[i]-W[i-1])

plt.plot(times,X)
# %%
X_true = np.zeros(len(X))
X_true[0] = 1
for i in range(1,len(times)):
    X_true[i] = np.exp( (mu-(sig**2)/2)*times[i] + sig*W[i])

error = np.mean(np.abs(X_true-X))
print(error)
# %%
# Testing strong convergence
dts = [0.1,0.01,0.001,0.0001,0.00001]
num_runs = 100
errors = np.zeros(len(dts))
for d in range(len(dts)):
    temp_errors = np.zeros(num_runs)
    dt = dts[d]
    times = np.arange(0,T+dt,dt)
    for j in range(num_runs):
        W = np.zeros(len(times))
        X = np.zeros(len(times))
        W[0] = 0
        X[0] = 1
        for i in range(1,len(W)):
            W[i] = W[i-1] + np.sqrt(dt)*np.random.randn(1)
        for i in range(1,len(X)):
            X[i] = X[i-1] + F(X[i-1],mu)*dt + G(X[i-1],sig)*(W[i]-W[i-1])
        X_true = np.zeros(len(X))
        X_true[0] = 1
        for i in range(1,len(times)):
            X_true[i] = np.exp( (mu-(sig**2)/2)*times[i] + sig*W[i])
        temp_errors[j] = np.mean(np.abs(X_true-X))
    errors[d] = np.mean(temp_errors)
#%%
plt.loglog(dts,errors,'ob')

log_slope, log_intercept = np.polyfit(log10(dts), log10(errors), 1)
coeffs = np.polyfit(log10(dts), log10(errors),1)
polyn = np.poly1d(coeffs)
log10_fit = polyn(log10(dts))
#plot fit
plt.loglog(dts, 10**log10_fit, '-r')
plt.title('Slope = ' + str(log_slope))
print(log_slope)
# %%
