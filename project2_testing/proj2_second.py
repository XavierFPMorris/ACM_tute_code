
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


# #time stepping constraints
# h = 10
# time = 200000
# nsteps = int(time/h)

# #Set up tensors to store Q, P
# Q = np.zeros((6,3,int(nsteps)))
# P = np.zeros((6,3,int(nsteps)))

# #apply initial conditions
# Q[:,:,0] = q
# P[:,:,0] = p

#Gravity constant
G = 2.95912208286*10**(-4)

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



#SE
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

#FE
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


#SV
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

# Plotting function
def plot_orbits(Q):
    plt.rcParams["figure.figsize"] = (12,12)
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(Q[0,0,:], Q[0,1,:],Q[0,2,:])
    ax.plot3D([0],[0],[0],'*',label = 'Sun')
    ax.plot3D(Q[1,0,:], Q[1,1,:],Q[1,2,:],label = 'Jupiter')
    # ax.plot3D(Q[2,0,:], Q[2,1,:],Q[2,2,:],label = 'Saturn')
    # ax.plot3D(Q[3,0,:], Q[3,1,:],Q[3,2,:],label = 'Uranus')
    # ax.plot3D(Q[4,0,:], Q[4,1,:],Q[4,2,:],label = 'Neptune')
    # ax.plot3D(Q[5,0,:], Q[5,1,:],Q[5,2,:],label = 'Pluto')
    ax.set(xlabel  = 'x (AU)',ylabel = 'y (AU)', zlabel = 'z (AU)')
    ax.legend(loc = 'best')
    ax.set_aspect('auto')
    plt.savefig("SE2.svg",bbox_inches='tight')
    plt.show()

def time_plot(Q):
    plt.rcParams["figure.figsize"] = (12,12)
    plt.rcParams.update({'font.size': 16})
    #fig,ax = plt.figure()
    plt.plot(Q[1,0,:])
    plt.xlabel('t (days)')
    plt.ylabel('x (AU)')
    plt.show()

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

# # Plot FE
# h = 10
# time = 200000
# nsteps = int(time/h)


# Q,P = FE(nsteps,h)
# plot_orbits(Q)


#Plot SE
h = 1
time = 200000
nsteps = int(time/h)


#Q,P = SE(nsteps,h)
Q,P = SV(nsteps,h)

#time_plot(P)
port_plot(Q)
#plot_orbits(Q)




# Plot SV
# h = 10
# time = 2000000*5
# nsteps = int(time/h)


# plot_orbits(Q)