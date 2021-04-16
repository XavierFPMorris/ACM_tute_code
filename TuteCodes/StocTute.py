#%%
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

for i in range(1,len(x)):
    x[i] = x[i-1] + np.random.normal(0,dt)

x*= sig
plt.plot(t,x)
# %%
### dt = 0.01
dt = 0.01
t = np.arange(tmin,tmax+dt, dt)


x = np.zeros(len(t))

for i in range(1,len(x)):
    x[i] = x[i-1] + np.random.normal(0,dt)

x*= sig
plt.plot(t,x)
# %%
### dt = 0.001
dt = 0.001
t = np.arange(tmin,tmax+dt, dt)


x = np.zeros(len(t))

for i in range(1,len(x)):
    x[i] = x[i-1] + np.random.normal(0,dt)

x*= sig
plt.plot(t,x)

# %%
dt = 0.001
t = np.arange(tmin,tmax+dt, dt)
C = 1000
X = np.zeros((len(t), C))

for k in range(C):
    for i in range(1,len(t)):
        X[i,k] = X[i-1,k] + np.random.normal(0,np.sqrt(dt))
X *= sig
vars = np.copy(t)
for i in range(len(t)):
    vars[i] = np.var(X[i,:])

plt.plot(t,vars)
print(sig**2)
print(vars[-1])
# %%
print(sig**2*dt)
# %%
