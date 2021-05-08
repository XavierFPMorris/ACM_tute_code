#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%%
#initial positions 
q = np.array([[0,0,0], 
[-3.5023653, -3.8169847, -1.5507963], 
[9.0755314, -3.0458353, -1.6483708], 
[8.3101420, -16.2901086, -7.2521278], [11.4707666, -25.7294829, -10.8169456], 
[-15.5387357, -25.2225594, -3.1902382]])


#initial velocities 
v = np.array([[0,0,0], 
[0.00565429, -0.00412490, -0.00190589], 
[0.00168318, 0.00483525, 0.00192462], 
[0.00354178, 0.00137102, 0.00055029], 
[0.00288930, 0.00114527, 0.00039677],
[0.00276725, -0.00170702, -0.00136504]])


#mass
m = np.array([[1.00000597682],
[0.000954786104043],
[0.000285583733151],
[0.0000437273164546],
[0.0000517759138449],
[1/(1.3*10**8)]])


#initial momentum
p = m*v

#%%

#Gravity constant
G = 2.95912208286*10**(-4)
#%%
#partial functions
def d_pH(i,t,P):
    return P[i,:,t]/m[i]

def d_pH_(p_,i):
    return p_/m[i]

def d_qH(i,t,Q):
    temp = np.zeros((1,3))
    for j in range(i):
        if j!=i:
            temp += -G*m[i]*m[j]*(Q[j,:,t] - Q[i,:,t])\
                /(np.linalg.norm(Q[i,:,t]-Q[j,:,t])**3)
    return temp


#%%
#Symp Euler
def SE(nsteps,h):
    Q = np.zeros((6,3,int(nsteps)))
    P = np.zeros((6,3,int(nsteps)))
    Q[:,:,0] = q.copy()
    P[:,:,0] = p.copy()
    for t in range(1,nsteps):
        for i in range(6):
            P[i,:,t] = P[i,:,t-1] - h*d_qH(i,t-1,Q)
            Q[i,:,t] = Q[i,:,t-1] + h*d_pH(i,t,P)
    return Q,P
#%%
#Forward Euler
def FE(nsteps,h):
    Q = np.zeros((6,3,int(nsteps)))
    P = np.zeros((6,3,int(nsteps)))
    Q[:,:,0] = q.copy()
    P[:,:,0] = p.copy()
    for t in range(1,nsteps):
        for i in range(6):
            Q[i,:,t] = Q[i,:,t-1] + h*d_pH(i,t-1,P)
            P[i,:,t] = P[i,:,t-1] - h*d_qH(i,t-1,Q)
    return Q,P

#%%
#Stormer-Verlet
def SV(nsteps,h):
    Q = np.zeros((6,3,int(nsteps)))
    P = np.zeros((6,3,int(nsteps)))
    Q[:,:,0] = q.copy()
    P[:,:,0] = p.copy()
    for t in range(1,nsteps):
        for i in range(6):
            p_temp = P[i,:,t-1] - h*0.5*d_qH(i,t-1,Q)
            Q[i,:,t] = Q[i,:,t-1] + h*d_pH_(p_temp,i)
            P[i,:,t] = p_temp - h*0.5*d_qH(i,t,Q)
    return Q,P
#%%
# Plotting functions
def plot_orbits(Q):
    plt.rcParams["figure.figsize"] = (12,12)
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(Q[0,0,:], Q[0,1,:],Q[0,2,:])
    ax.plot3D([0],[0],[0],'*',label = 'Sun')
    ax.plot3D(Q[1,0,:], Q[1,1,:],Q[1,2,:],label = 'Jupiter')
    ax.plot3D(Q[2,0,:], Q[2,1,:],Q[2,2,:],label = 'Saturn')
    ax.plot3D(Q[3,0,:], Q[3,1,:],Q[3,2,:],label = 'Uranus')
    ax.plot3D(Q[4,0,:], Q[4,1,:],Q[4,2,:],label = 'Neptune')
    ax.plot3D(Q[5,0,:], Q[5,1,:],Q[5,2,:],label = 'Pluto')
    ax.set(xlabel  = 'x (AU)',ylabel = 'y (AU)', zlabel = 'z (AU)')
    ax.legend(loc = 'best')
    ax.set_aspect('auto')
    plt.show()
#phase portrait plot
def port_plot(Q):
    plt.rcParams["figure.figsize"] = (12,12)
    plt.rcParams.update({'font.size': 16})
    #fig,ax = plt.figure()
    y = np.roll(Q[1,0,:],1)
    x = Q[1,0,:]
    plt.plot(x[1:],y[1:])
    plt.xlabel('x(n)')
    plt.ylabel('x(n+1)')
    plt.show()
#%%
# Plot FE
h = 10
time = 200000
nsteps = int(time/h)


Q,P = FE(nsteps,h)
plot_orbits(Q)

#%%
# Plot SE
h = 10
time = 200000
nsteps = int(time/h)


Q,P = SE(nsteps,h)
plot_orbits(Q)

#%%
# Plot SV
h = 10
time = 200000
nsteps = int(time/h)

Q,P = SV(nsteps,h)
plot_orbits(Q)

#%%
#Calculate H (energy)
def H(q,p):
    temp1 = 0
    for i in range(6):
        temp1 += 0.5/m[i]*(np.dot(p[i,:], p[i,:]))
    
    for i in range(1,6):
        temp2 = 0
        for j in range(i):
            temp2 += -G*m[i]*m[j]/np.linalg.norm(q[i,:]-q[j,:])
        temp1 += temp2
    return temp1
#%%
#simple H plot
plt.rcParams["figure.figsize"] = (12,6)
plt.rcParams.update({'font.size': 10})
h = 10
time = 200000
nsteps = int(time/h)
plt.subplot(1,3,1)
Q,P = FE(nsteps,h)
H_vec = np.zeros(nsteps)
for i in range(nsteps):
    H_vec[i] = H(Q[:,:,i], P[:,:,i])
plt.plot(H_vec)
plt.title('Forward Euler')
plt.xlabel('timestep')
plt.ylabel('H')
plt.subplot(1,3,2)
Q,P = SE(nsteps,h)
H_vec = np.zeros(nsteps)
for i in range(nsteps):
    H_vec[i] = H(Q[:,:,i], P[:,:,i])
plt.plot(H_vec)
plt.title('Symp Euler')
plt.xlabel('timestep')
plt.ylim([-3.21e-8,-3.22e-8])
#plt.ylabel('H')
plt.subplot(1,3,3)
Q,P = SV(nsteps,h)
H_vec = np.zeros(nsteps)
for i in range(nsteps):
    H_vec[i] = H(Q[:,:,i], P[:,:,i])
plt.plot(H_vec)
plt.title('Stormer-Verlet')
plt.xlabel('timestep')
plt.ylim([-3.21e-8,-3.22e-8])
#plt.ylabel('H')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.5)
plt.show()
#%%
from numpy import log10
# %%
#calculate order with h
def order_calc(f,N):
    #time steps to look at
    hs = np.logspace(0,1,num=N)+5
    time = 200000
    #initial H value
    H_0 = H(q,p)
    er = np.zeros(N)
    # for each time step, integrate and calculate the RMS of H
    for i in range(N):
        h = hs[i]
        nsteps = int(time/h)
        Q,P = f(nsteps,h)
        H_vec = np.zeros(nsteps)
        for j in range(nsteps):
            H_vec[j] = H(Q[:,:,j], P[:,:,j])
        res = np.ones(nsteps)*H_0 - H_vec
        er[i] = np.sqrt(np.mean(res**2))
    fig, ax = plt.subplots(1,1)
    ax.loglog(hs,er,'ob')

    ## FITTING
    plot_text = "Slope = {s:.2f}"
    #calculate fit
    log_slope, log_intercept = np.polyfit(log10(hs), log10(er), 1)
    coeffs = np.polyfit(log10(hs), log10(er),1)
    polyn = np.poly1d(coeffs)
    log10_fit = polyn(log10(hs))
    #plot fit
    ax.loglog(hs, 10**log10_fit, '-r')
    ax.text(hs[5], 10**(log10_fit[5])  , plot_text.format(s = log_slope), horizontalalignment='right', c = 'r')

    ax.set(xlabel = 'h', ylabel = 'RMS(H)', title = 'RMS of H as a function of h, log axes')
    plt.show()
# %%
#FE order
order_calc(FE,10)
#%%
#SE order
order_calc(SE,10)
# %%
#SV order
order_calc(SV,10)
# %%
# phase portrait plots
h = 100
time = 200000
nsteps = int(time/h)

#Q,P = SE(nsteps,h)
Q,P = SV(nsteps,h)

port_plot(Q)
