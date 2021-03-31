import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as sciInt
from numpy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from numpy import sqrt, exp, cosh as sqrt, exp, cosh
from matplotlib.animation import FuncAnimation
import time

num_points = 256

length = 20

#space discretisation
x = np.linspace(0,length,num_points)

dx = x[1] - x[0]

#print(dx - length/(num_points-1))
#wavenumber? discretisation
k = fftfreq(num_points)
print(k)
#print(dx)




#my_K = ifftshift(np.arange(-np.pi/dx, np.pi/dx, dx))

#initial Gaussian profile
a = 1
b = 4
gauss_i = a*exp(-b*(x - length/2)**2)

#initial soliton solution profile
beta = 0.05
soliton_i = beta/2 * cosh( sqrt(beta)/2 *(x-length/2))**-2

#initial cos profile
sin_i = np.sin(2*np.pi/length*x)

XX = fft(sin_i)*1j*k*num_points
XXX = fft(sin_i)*(1j*k*num_points)**2
XX[k*num_points > num_points/3] = 0
XXX[k*num_points > num_points/3] = 0
plt.plot( ifft(XX))
plt.plot(ifft(XXX))
plt.plot(sin_i)
plt.show()

#nth derivative in fourier space, space input 
def f_deriv(u,n):
    return ((1j*k)**n)#*fft(u)

#linear part of time derivative
def L_(u):
    return -1*f_deriv(u,3)

#non linear part of time derivative
def N_(u):
    return -6*(0.5*f_deriv(u**2, 1))

#plt.plot(x, ifft(L_(gauss_i)))
#plt.show()

#time step
dt = 0.5

#time steps 
nsteps = 1000*1

#evolution matrix, time x space 
U = np.zeros((nsteps, num_points), dtype = np.complex_)

#set initial profile 

U[0,:] = soliton_i

#perform one FE step

L_new = (-1j*k)**3*num_points
print(L_new)
U[1,:] = ifft( fft(U[0,:])  + dt*L_new)#+ dt*N_(U[0,:]) )



# dt*3/2*1/2*(1j)*k*fft(U[i-1,:]**2) -  dt*1/2*1/2*(1j)*k*fft(U[i-2,:]**2)



#time evolution 

for i in range(2, nsteps):
    #L = L_(U[i-1, :])
    #Ni = N_(U[i-1,:])
    #Ni1 = N_(U[i-2,:])
    #U_k = (1-dt/2*L)**(-1) *  ( fft(U[i-1,:])*(1+dt/2*L) + 3/2*dt*Ni - 1/2*dt*Ni1 )
    U_k = (1-dt/2*L_new)**(-1) *  (fft(U[i-1,:])*(1+dt/2*L_new) + 1/dx*dt*6*3/2*1/2*(-1j)*k*fft((U[i-1,:])**2) - 1/dx*dt*6*1/2*1/2*(-1j)*k*fft((U[i-2,:])**2))
    #anti-aliasing
    U_k[abs(k*num_points) > num_points/3] = 0
    U[i,:] = ifft(U_k)

#plotting 

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if i%100==0:
        ax1.clear()
        ax1.plot(x, soliton_i)
        ax1.plot(x,np.real(U[i,:]))


ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)
#plt.plot(sin_i)
#plt.plot(U[-1,:])
plt.show()

# mass = np.sum(U, 1)

# u2 = np.sum(U**2, 1)

# print((mass))

# plt.plot(u2)
# plt.show()
