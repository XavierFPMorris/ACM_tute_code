import matplotlib.pyplot as plt 
import numpy as np
from numpy.fft import fft, fftfreq,  ifft
from numpy import cosh as cosh
from matplotlib.animation import FuncAnimation

N = 400

length = 100

dx = length/N

x = dx*np.arange(N)

k = 2*np.pi/length*fftfreq(N)*N



#initial soliton solution profile
beta = 0.5
soliton_i = 0.5*beta/(cosh( (beta)**0.5/2*(x-length/2)))**2 

#soliton_i = beta/2 * ( cosh( sqrt(beta)/2 * (x-length/2) ) )**(-2.)

L_= -(1j*k)**3

def N_(u):
    return -6*0.5*1j*k*fft(u**2)

#time step
dt = 0.05
#time steps 
nsteps = 1000
#evolution matrix, time x space 
U = np.zeros((nsteps, N), dtype = np.complex_)
#set initial profile 
U[0,:] = soliton_i

U[1,:] = ifft( fft(U[0,:])  + dt*L_*fft(U[0,:]) +  dt*N_(U[0,:]) )    

for i in range(2, nsteps):

    U_k = ( (1-dt/2*L_)**(-1) )    *    (    (1+dt/2*L_)*fft(U[i-1,:])  +  3/2*dt*N_(U[i-1,:])  -  1/2*dt*N_(U[i-2,:])   )
    #U_k[abs(k)*np.pi > N/3] = 0
    U[i,:] = ifft(U_k)

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if i%100==0:
        ax1.clear()
        ax1.plot(x, soliton_i)
        ax1.plot(x,np.real(U[i,:]))


ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)

plt.show()