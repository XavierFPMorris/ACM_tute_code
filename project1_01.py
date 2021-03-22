import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as sciInt
from numpy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from numpy import sqrt, exp, cosh as sqrt, exp, cosh
from matplotlib.animation import FuncAnimation
import time


num_points = 256

length = 15

#space discretisation
x = np.linspace(0,length,num_points)

dx = x[1] - x[0]

#wavenumber? discretisation
k = fftfreq(num_points)

#initial Gaussian profile
a = 0.5
b = length/2
gauss_i = a*exp(-b*(x - length/2)**2)

#initial soliton solution profile
beta = 1
soliton_i = beta/2*cosh( sqrt(beta)/2 *(x-length/2))**-2

#nth derivative in fourier space, space input 
def f_deriv(u,n):
    return ((1j*k)**n)*fft(u)

#linear part of time derivative
def L_(u):
    return -1*f_deriv(u,3)

#non linear part of time derivative
def N_(u):
    return -6*(0.5*f_deriv(u**2, 1))

#time step
dt = 0.1

#time steps 
nsteps = 1000*2

#evolution matrix, time x space 
U = np.zeros((nsteps, num_points), dtype = np.complex_)

#set initial profile 

U[0,:] = soliton_i

#perform one FE step

U[1,:] = ifft( fft(U[0,:]) + dt*L_(U[0,:]) + dt*N_(U[0,:]) )

#time evolution 

for i in range(2, nsteps):
    L = L_(U[i-1, :])
    Ni = N_(U[i-1,:])
    Ni1 = N_(U[i-2,:])
    U_k = (1-dt/2*L)**(-1) * ( (fft(U[i-1,:])+dt/2*L) + 3/2*dt*Ni - 1/2*dt*Ni1 )
    #anti-aliasing
    U_k[abs(k*num_points) > num_points/3] = 0
    U[i,:] = ifft(U_k)

#plotting 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if i%10==0:
        ax1.clear()
        ax1.plot(x, soliton_i)
        ax1.plot(x,np.real(U[i,:]))


ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)
plt.show()
