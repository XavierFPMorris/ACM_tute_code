#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%
p0 = 1
q0 = 1

h = 0.01

nsteps = 1000

p = np.zeros(nsteps)
q = np.zeros(nsteps)

p[0] = p0
q[0] = q0

#%%
### Explicit

p = np.zeros(nsteps)
q = np.zeros(nsteps)

p[0] = p0
q[0] = q0

for i in range(1,nsteps):
    q[i] = q[i-1] + h*p[i-1]
    p[i] = p[i-1] - h*q[i-1]

plt.plot(q,p)
plt.show()
#%%
### Implicit

p = np.zeros(nsteps)
q = np.zeros(nsteps)

p[0] = p0
q[0] = q0

A = np.array([[0,-1],[1,0]])
for i in range(1,nsteps):
    yn = np.array([[p[i-1]],[q[i-1]]])
    yn1 = np.linalg.inv(np.eye(2)-h*A)@yn
    p[i],q[i] = yn1[0], yn1[1]

plt.plot(q,p)
plt.show()
#%%
### Leap Frog

p = np.zeros(nsteps)
q = np.zeros(nsteps)

p[0] = p0
q[0] = q0

for i in range(1,nsteps):
    pmid = p[i-1] - h*0.5*q[i-1]
    q[i] = q[i-1] + h*pmid
    p[i] = pmid - h*0.5*q[i]

plt.plot(q,p)
plt.show()
# %%
### Runge-Kutta

p = np.zeros(nsteps)
q = np.zeros(nsteps)

p[0] = p0
q[0] = q0

def f(inp):
    A = np.array([[0,-1],[1,0]])
    return A @ inp

for i in range(1,nsteps):
    yn = np.array([[p[i-1]],[q[i-1]]])
    Y1 = yn + h*0.5*f(yn)
    Y2 = yn + h*0.5*f(Y1)
    Y3 = yn + h*f(Y2)
    yn1 = yn + h/6*(f(yn) + 2*f(Y1) + 2*f(Y2) + f(Y3))
    p[i],q[i] = yn1[0], yn1[1]

plt.plot(q,p)
plt.show()

# %%
