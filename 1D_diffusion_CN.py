import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

Nx = 100
Nt = 100
dt = 0.5
Ln = 1

num_iter = 1000*3

x = np.linspace(0, Ln, Nx + 1)
t = np.linspace(0, dt*Nt, Nt + 1)

dx = x[2]-x[1]
dt = t[2]-t[1]


temp = np.ones(len(x)-1)

L = np.eye(len(x))*-2 + np.diag(temp, 1) + np.diag(temp,-1)

L[0][-1] = 1
L[-1][0] = 1

#u = np.zeros(Nx+1)

u_n = np.zeros((num_iter, Nx+1))

[val, vec] = np.linalg.eig(L)

val[::-1].sort()

#plt.scatter([i for i in range(np.size(val))] ,val)
#plt.show()

u = np.sin(x)
u = np.exp(-(x-Ln/2)**2/0.1)
u_n[0,:] = u

mn = np.mean(u)

# plt.plot(x,u)
# plt.show()

for i in range(1,num_iter):
    #u_n[i,:] = (1-dt/2*L) @ np.linalg.inv(1+dt/2*L) @ u
    u_n[i,:] = np.linalg.solve((np.eye(len(x)) - dt*L/(2)), (np.eye(len(x))+dt*L/(2))@u)
    u = u_n[i,:]

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if(i%15==0):
        ax1.clear()
        ax1.plot(x,u_n[i,:])
        ax1.plot([0,1], [mn,mn])
        plt.ylim([0,1])

ani = FuncAnimation(fig, animate, frames = list(range(0,num_iter)), interval = 1)
plt.show()
