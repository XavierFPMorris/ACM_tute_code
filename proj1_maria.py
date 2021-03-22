#!/usr/bin/env python
# coding: utf-8

# # Analytical and Numerical solutions to the Korteweg de Vries (KdV) equation
# 

# ### Analytical solution with a Gaussian initial profile
# 
# <br>
# Consider a system:
# $$
# \begin{cases}
# u_t + 6uu_x +u_{xxx} = 0 \\
# u(x,0) = u_0(x) = A \exp(-B(x-L/2)^2) 
# \end{cases}
# $$
# <br>

# ### Plotting the initial profile

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


A = 1
B = 8
L = 4

x = np.linspace(0,4,100)
u_0 = A*np.exp(-B*(x-L/2)**2)

plt.plot(x,u_0)
plt.xlabel(r'$x$')
plt.ylabel(r'$u_0$')
plt.title(r'$u_0(x)$ (Initial Profile)')
plt.show()


# ### Plotting the analytical solution
# 
# The system has an analytical solution:
# 
# $$u(x, t)=\frac{c}{2} \operatorname{sech}^{2}\left[\frac{\sqrt{c}}{2}(x-c t)\right]$$

# In[ ]:


def kdv_exact(x, t, c):
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*(x-c*t))**(-2)
    return u


# In[ ]:


import matplotlib.animation as animation
get_ipython().run_line_magic('matplotlib', 'notebook')

n_iter = 100
c = 1
x = np.linspace(0,4,n_iter)
t = np.linspace(0,5,n_iter)

# create matrix of updated solution at each step
u_xt = np.zeros([n_iter,len(x)])


    
for i in range(1,n_iter):
    u_xt[i,:] = kdv_exact(x, t[i], c)



fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if(i%2==0):
        ax1.clear()
        ax1.plot(x,u_xt[i,:])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x,t)$')
        plt.ylim([0,1])

ani = animation.FuncAnimation(fig, animate, frames = list(range(0,n_iter)), interval = 0.01)
plt.show()


# ## Numerical solution to KdV using spectral methods
# 
# ### Plotting the initial condition

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

L  = 2*np.pi
N = 100

x = np.linspace(0,L,N)
# x = (2.*np.pi/N)*np.arange(-N/2,N/2).reshape(N,)

del_x = x[1]-x[0]


lam_min = 2*del_x

kmax = 2*np.pi/lam_min
del_k = 2*np.pi/L

# k = np.arange(-kmax,kmax+del_k,del_k)
k = np.fft.fftfreq(len(x))*N

# anti-aliasing
anti_alias = abs(k) > N/3
k[anti_alias] = 0

A = 2
B = 1
u = A*np.exp(-B*(x-L/2)**2)
uk = np.fft.fft(u)

plt.plot(k,(uk.real))
plt.xlim([-15,15])
plt.xlabel(r'$k$')
plt.ylabel(r'$u(k)$')
plt.title('Initial Gaussian Profile in Fourier Space')
plt.show()


# ### Define a function to find the n-th x-derivative of u 
# 
# These are the linear terms in the PDE.

# In[ ]:


def nth_xderiv(u, k, N):
    """
    u: function to be partially differentiated
    k: discretised points in Fourier space
    N: n-th order partial differential
    
    returns the n-th partial derivative at finite grid points
    
    """

    n = len(x)
    
    a_k = np.fft.fft(u,n = n)  
    
    deriv = (1j*k)**N * a_k * np.exp(2*np.pi*k/n)
                    
    return deriv


# In[ ]:


def lin(u,k):
    return nth_xderiv(u, k, 3)


# ### Plotting derivatives of u

# In[ ]:


u = A*np.exp(-B*(x-L/2)**2)
u_x = -2*B*(x-L/2)*u
u_xx = A*B*(4*B*x**2-4*B*L*x+B*L**2-2) * np.exp(-B*(x-L/2)**2)

k = np.fft.fftfreq(len(x))*N


fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize = (18,3))

ax0.plot(x, u, label = r'$u$')
ax0.plot(x, ((np.fft.ifft(nth_xderiv(u,k,0))).real), label = r'$u^{-1}(k)$')
ax0.legend(loc="upper right")
ax0.set_xlabel(r'$x$')

ax1.plot(x, u_x, label = r'$u_x$')
ax1.plot(x, ((np.fft.ifft(nth_xderiv(u,k,1))).real), label = r'$u_x^{-1}(k)$')
ax1.legend(loc="upper right")
ax1.set_xlabel(r'$x$')

ax2.plot(x, u_xx, label = r'$u_{xx}$')
ax2.plot(x, ((np.fft.ifft(nth_xderiv(u,k,3))).real), label = r'$u_{xx}^{-1}(k)$')
ax2.legend(loc="upper right")
# ax2.set_ylim([-11,11])
ax2.set_xlabel(r'$x$')

plt.show()


# In[ ]:


second = nth_xderiv(u,k,2)

print(second.min())

plt.plot(x, np.fft.ifft(nth_xderiv(u,k,3)).real, label = r'$u_{xxx}$')
# plt.plot(x, third.real, label = r'$u_{xx}^{-1}(k)$')
plt.legend(loc="upper right")
# ax2.set_ylim([-11,11])
plt.xlabel(r'$x$')
plt.show()


# ### Define a function to describe the non-linear term
# 
# Note that the non-linear term can be written as:
# <br>
# $$6 uu_x = 6  \left(\frac{1}{2} u^2\right)_x = 3 u^2_x$$

# In[ ]:


def non_lin(u,k):
    return 3 * nth_xderiv(u**2, k, 1)


# In[ ]:


plt.plot(x,np.fft.ifft(non_lin(u,k)).real )
plt.xlabel(r'$x$')
plt.ylabel(r'$6uu_x$')
plt.title('Non-linear term')
plt.show()


# ### Crank-Nicolson Scheme
# 
# Use the Crank-Nicolson scheme to step forward in time.
# 
# $$ \frac{U_j^{n+1}-U_j^{n}}{\Delta t} = - L U^{n+\frac{1}{2}}_j - N U^{n+\frac{1}{2}}_j = - \frac{L}{2}\left(U_{j}^{n}+U_{j}^{n+1}\right) - \frac{3}{2} N(U^n) + \frac{1}{2}N(U^{n-1})
# $$
# 
# $$
# \implies U_j^{n+1} = \frac{1}{1-\frac{\Delta t}{2} L} \bigg( - (1+\frac{\Delta t}{2} L) \ U_{j}^{n} - \frac{3 \Delta t}{2} N(U^n) + \frac{\Delta t}{2}N(U^{n-1}) \bigg)
# $$

# In[ ]:


n_iter = 1800

# create matrix of updated solution at each step
updates = np.zeros([n_iter,N])

# set initial condition
# u0 = A*np.exp(-B*(x-L/2)**2)
u0 = 18/(np.cosh(3*(x-1)))**2  #+ 2/(np.cosh(x-1))**2
# u0 = 2*np.cos(np.pi*x)
updates[0,:] = u0

# time discretisation
dt = 0.0001

n = len(x)
k = np.fft.fftfreq(n)

# use forward Euler scheme for first time step
updates[1,:] = np.fft.ifft(np.fft.fft(updates[0,:]) + dt*( -lin(updates[0,:],k) - non_lin(updates[0,:],k)) ) 

 
for i in range(2,n_iter):
    u_n = updates[i-1,:]
    
    L_n = (lin(u_n,k))
    N_n = (non_lin(u_n,k))
    N_n_1 = (non_lin(updates[i-2,:],k))
    
    u = -1*(u_n+dt/2*L_n) - 3*dt/2*N_n + dt/2*N_n_1 
    u /= (1-dt/2*L_n)
    
    anti_alias = abs(k*n) > n/3
    u[anti_alias] = 0
    
    updates[i,:] = np.fft.ifft(u)
    


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(x,updates[0,:].real)
plt.plot(x,updates[100,:].real)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    if(i%10==0):
        ax1.clear()
        ax1.plot(x,updates[i,:].real)
        plt.ylim([-5,20])

ani = animation.FuncAnimation(fig, animate, frames = list(range(0,n_iter)), interval = 1)
plt.show()


# In[ ]:




