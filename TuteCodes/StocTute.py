import numpy as np
import matplotlib.pyplot as plt

#%%
sig = 0.5

tmin = 0

tmax = 1

#%%

### dt = 0.1
dt = 0.1
t = np.arange(tmin,tmax+dt, dt)


x = np.zeros(len(t))

for i in range(len(x)):
    x[i] = sig*np.random.normal(0,t[i])

plt.plot(t,x)
# %%
### dt = 0.01
dt = 0.01
t = np.arange(tmin,tmax+dt, dt)


x = np.zeros(len(t))

for i in range(len(x)):
    x[i] = sig*np.random.normal(0,t[i])

plt.plot(t,x)
# %%
### dt = 0.001
dt = 0.001
t = np.arange(tmin,tmax+dt, dt)


x = np.zeros(len(t))

for i in range(len(x)):
    x[i] = sig*np.random.normal(0,t[i])

plt.plot(t,x)

# %%
t = 0.5
Xt = sig*np.random.normal(0,t,(1000,1))
print(sig**2*t - np.var(Xt))
Yt = 2*np.random.normal(0,5,(1000,1))
print(np.var(Yt))
# %%
