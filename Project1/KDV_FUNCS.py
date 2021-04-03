import matplotlib.pyplot as plt 
import numpy as np
from math import pi
from numpy.fft import fft, fftfreq, ifft
from numpy import cosh, exp, cos
from matplotlib.animation import FuncAnimation
#from .Kdv import KDV

def KDV_INT(N, length, dt, nsteps, IC):
    ### SETUP ###

    #grid spacing
    #dx = length/N 

    #spatial discretisation
    #x = dx*np.arange(N) 

    #wavenumber discretisation
    k = 2*np.pi*N/length*fftfreq(N) 

    ### DERIVATIVE OPERATORS

    # LINEAR TERM 1 (U_xxx)

    L_= -(1j*k)**3

    # NON-LINEAR TERM (6UU_x)

    def N_(u):
        return -6*0.5*1j*k*fft(u**2)

    # Integrate :)
    U = CNAB(IC, nsteps, N, L_, N_, dt)

    return U



### CRANK NIC + ADAMS BASH PSEUDO SPECTRAL INTEGRATOR FUNC
def CNAB(IC, nsteps, N, L_, N_, dt):

        #matrix to store U in spatial space at each time step
        U = np.zeros((nsteps, N), dtype = np.complex_) 

        #select initial condition
        U[0,:] = IC

        #perform one forward euler step 
        U[1,:] = ifft( fft(U[0,:])  + dt*L_*fft(U[0,:]) +  dt*N_(U[0,:]) )  

        #integrate :)
        for i in range(2, nsteps):
            U_k = (1-dt/2*L_)**(-1) * ( (1+dt/2*L_)*fft(U[i-1,:]) + 3/2*dt*N_(U[i-1,:]) - 1/2*dt*N_(U[i-2,:]) )
            U[i,:] = ifft(U_k)
        
        return U

### EVOLUTION PLOTTING
def evo_plot(x, U, nsteps):
    fig = plt.figure()

    ax1 = fig.add_subplot(1,1,1)

    def animate(i):
        if i%100==0:
            ax1.clear()
            ax1.plot(x, U[0,:])
            ax1.plot(x,np.real(U[i,:]))


    ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)

    plt.show()


def KDV_INT_2(N, length, dt, nsteps, IC, beta, alpha, nu):
    ### SETUP ###

    #grid spacing
    #dx = length/N 

    #spatial discretisation
    #x = dx*np.arange(N) 

    #wavenumber discretisation
    k = 2*np.pi*N/length*fftfreq(N) 

    ### DERIVATIVE OPERATORS

    # LINEAR TERM 1 (U_xxx)

    L_= -beta*(1j*k)**3 -alpha*(1j*k)**2 -nu*(1j*k)**4

    # NON-LINEAR TERM (6UU_x)

    def N_(u):
        return -6*0.5*1j*k*fft(u**2)

    # Integrate :)
    U = CNAB(IC, nsteps, N, L_, N_, dt)

    return U