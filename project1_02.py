from __future__ import division
from math import sqrt, pi
import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# N = 256

# length = 30

N = 256

length = 50


### space discretisation 
dx = length/N
x = dx*np.arange(N)


### wavenumber discretisation

k = np.fft.fftfreq(N)*N
# k = np.fft.ifftshift(k)
k = k*2*np.pi/length




#k = np.concatenate(( np.arange(0, N/2), np.array([0]), np.arange(-N/2 + 1, 0, 1) )).reshape(N,)

#print(k)

#def lin(y):
#    return 1*(1j*k**3)*y

L = -(1j*k**3)



def non_lin(y):
    return  .5*1j*k*fft(ifft(y,axis=0)**2.,axis=0) * 6


x = (2.*length/N)*np.arange(-N/2,N/2).reshape(N,1) # Space discretization
s, shift = 0.025, 0. # Initial data is a soliton
y0 = (.5*s*np.cosh(.5*(sqrt(s)*(x+shift)))**(-2.)).reshape(N,)

#print(x)

dt = 0.00001

y0 = fft(y0, axis = 0)

nsteps = 500

Y = np.zeros((len(k), nsteps), dtype=np.complex_)

#print([len(x), len(k), len(y0)])
y0[abs(k)>N/3] = 0
Y[:,0] = y0


y1 = Y[:,0]*dt*L + Y[:,0] + dt*non_lin(Y[:,0])
y1[abs(k)>N/3] = 0
Y[:,1] = y1

for i in range(2, nsteps):
    Yc = (1-dt/2*L)**(-1.) * ( (1+dt/2*L)*Y[:,i-1] + 3/2*dt*non_lin(Y[:,i-1]) - 1/2*dt*non_lin(Y[:,i-2]) )
    Yc[abs(k)>N/3] = 0
    Y[:,i] = Yc

#plt.plot(ifft(Y[:, 0], axis=0))

#plt.plot(ifft(Y[:, 1], axis=0))

#plt.plot(ifft(Y[:,-1], axis=0))
#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if i%25==0:
        ax1.clear()
        ax1.plot(x, (.5*s*np.cosh(.5*(sqrt(s)*(x+shift)))**(-2.)).reshape(N,))
        ax1.plot(x,ifft(Y[:,i]))

ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)

plt.show()