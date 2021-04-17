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
print(p)

#%%
#time stepping constraints
h = 10
time = 200000
nsteps = int(time/h)

#Set up tensors to store Q, P
Q = np.zeros((6,3,int(nsteps)))
P = np.zeros((6,3,int(nsteps)))

#apply initial conditions
Q[:,:,0] = q
P[:,:,0] = p

#Gravity constant
G = 2.95912208286*10**(-4)
#%%
def d_pH(i,t):
    return P[i,:,t]/m[i]

def d_qH(i,t):
    temp = np.zeros((1,3))
    for j in range(6):
        if j != i:
            temp += -G*m[i]*m[j]*(Q[j,:,t] - Q[i,:,t])\
                /(np.linalg.norm(Q[i,:,t]-Q[j,:,t])**3)
    return temp


#%%
for t in range(1,nsteps):
    for i in range(6):
        P[i,:,t] = P[i,:,t-1] - h*d_qH(i,t-1)
        Q[i,:,t] = Q[i,:,t-1] + h*d_pH(i,t)
#%%
# for t in range(1,nsteps):
#     for i in range(6):
#         Q[i,:,t] = Q[i,:,t-1] + h*d_pH(i,t-1)
#         P[i,:,t] = P[i,:,t-1] - h*d_qH(i,t-1)
#%%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(Q[0,0,:], Q[0,1,:],Q[0,2,:])
ax.plot3D(Q[1,0,:], Q[1,1,:],Q[1,2,:])
ax.plot3D(Q[2,0,:], Q[2,1,:],Q[2,2,:])
ax.plot3D(Q[3,0,:], Q[3,1,:],Q[3,2,:])
ax.plot3D(Q[4,0,:], Q[4,1,:],Q[4,2,:])
ax.plot3D(Q[5,0,:], Q[5,1,:],Q[5,2,:])
#ax.plot3D(Q[6,0,:], Q[6,1,:],Q[6,2,:])

plt.show()



#%%
def H(q,p):
    temp1 = 0
    for i in range(6):
        temp1 += 0.5*m[i]*(np.dot(p[i,:], p[i,:]))
    
    for i in range(1,6):
        temp2 = 0
        for j in range(i):
            temp2 += -G*m[i]*m[j]/np.linalg.norm(q[i,:]-q[j,:])
        temp1 += temp2
    return temp1
H_vec = np.zeros(nsteps)
for i in range(nsteps):
    H_vec[i] = H(Q[:,:,i], P[:,:,i])

plt.plot(H_vec)
plt.show()

# %%
