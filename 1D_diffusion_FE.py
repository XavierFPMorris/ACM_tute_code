import numpy as np
import matplotlib.pyplot as plt


Nx = 100
Nt = 100
dt = 0.1
Ln = 10

x = np.linspace(0, Ln, Nx + 1)
t = np.linspace(0, dt*Nt, Nt + 1)

dx = x[2]-x[1]
dt = t[2]-t[1]

temp = np.ones(len(x)-1)

L = np.eye(len(x))*-2 + np.diag(temp, 1) + np.diag(temp,-1)

L[0][-1] = 1
L[-1][0] = 1

u = np.zeros(Nx+1)
u_n = np.zeros(Nx+1)

[val, vec] = np.linalg.eig(L)

val[::-1].sort()

plt.scatter([i for i in range(np.size(val))] ,val)
plt.show()
