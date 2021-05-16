#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import DOP853,RK45
#%%
#derivative function
def Lorenz63(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ 10*(y-x),28*x - y - x*z,  -8/3*z + x*y]
#%%
#set up initial conditions
dt = 0.02
N = 10000
initial_state = [0.95,1.,1.05]
t0 = 0
t_bound = dt*N

#create solver object
solver_obj = DOP853(Lorenz63,t0,initial_state,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X = np.zeros((N+1,3))
X[0] = initial_state
#%%
#compute output
for i in range(N):
    solver_obj.step()
    X[i+1] = solver_obj.y
#%%
#plot the phase space (attractor)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X[:,0],X[:,1],X[:,2], lw = 0.4)
ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*')
plt.show()
#%%
#calculate exact derivatives
X_dot = np.zeros((N+1,3))
X_dot[:,0] = 10*(X[:,1] - X[:,0])
X_dot[:,1] = 28*X[:,0] - X[:,1] - X[:,0]*X[:,2]
X_dot[:,2] = -8/3*X[:,2] + X[:,0]*X[:,1]

#%%
# Function for finite differences
# Used central, + 2nd O forward and backward for the end points
def FD_deriv(vec,dt):
    f = [1/(2*dt),0,-1/(2*dt)]
    out = np.convolve(vec,f,'same')
    out[0] = (-1/2*vec[2] + 2*vec[1] - 3/2*vec[0])/dt
    out[-1] = (3/2*vec[-1] -2*vec[-2]+1/2*vec[-3])/dt
    return out
#%%
#calculate derivatives from finite differences 
X_dot_approx = np.zeros((N+1,3))

X_dot_approx[:,0] = FD_deriv(X[:,0],dt)
X_dot_approx[:,1] = FD_deriv(X[:,1],dt)
X_dot_approx[:,2] = FD_deriv(X[:,2],dt)

#%%
#Check derivatives are close
print(np.mean(X_dot - X_dot_approx))
ind = 2 #or 0,1
print(X[:,ind])
print(X_dot[:,ind])
print(X_dot_approx[:,ind])
# %%
'''
Now for SINDy
We first need to create our library Theta
Theta will be restricted to cubic polynomials 
'''

import itertools

inds = [0, 1, 2, 3]
combos = [list(p) for p in itertools.product(inds, repeat=3) if np.sum(p)<=3]
##-1*x.count(0),np.max(x),
combos = sorted(combos, key = lambda x: (np.sum(x),x[2],x[1],x[0]),reverse=False)
#print(combos)
'''
Our library is then:
'''
for c in combos:
    print('x^{} * y^{} * z^{}'.format(c[0],c[1],c[2]))
'''
x^0 * y^0 * z^0
x^1 * y^0 * z^0
x^0 * y^1 * z^0
x^0 * y^0 * z^1
x^2 * y^0 * z^0
x^1 * y^1 * z^0
x^0 * y^2 * z^0
x^1 * y^0 * z^1
x^0 * y^1 * z^1
x^0 * y^0 * z^2
x^3 * y^0 * z^0
x^2 * y^1 * z^0
x^1 * y^2 * z^0
x^0 * y^3 * z^0
x^2 * y^0 * z^1
x^1 * y^1 * z^1
x^0 * y^2 * z^1
x^1 * y^0 * z^2
x^0 * y^1 * z^2
x^0 * y^0 * z^3
'''

#%%
# Library matrix func
def gen_Theta(combos,N,X):
    Theta = np.zeros((N+1,len(combos)))
    for (i,combo) in enumerate(combos):
        x,y,z = combo
        Theta[:,i] = (X[:,0]**x)*(X[:,1]**y)*(X[:,2]**z)
    return Theta
#%%
# defining our library matrix
Theta = gen_Theta(combos,N,X)

# %%
# Least squares function
def xlstsq(Thet,Xdot):
    return np.linalg.inv( np.transpose(Thet)@Thet)@np.transpose(Thet)@Xdot
# %%
# initial guess from least squares
Xi = xlstsq(Theta,X_dot)
print(Xi)

# %%
#Sequential thresholding function
def STLS(lam, Xi, X_dot,Theta):
    cond = True
    count = 0
    while cond:
        smalls = (np.abs(Xi)<lam)
        Xi[smalls] = 0
        old = Xi.copy()
        for ind in range(3):
            bigs= ~smalls[:,ind]
            Xi[bigs,ind] = xlstsq( Theta[:,bigs],X_dot[:,ind])
        if np.abs(np.mean(Xi-old))<10**(-10):
            cond = False
        count += 1
        if count>1000:
            cond = False
            print("failed")
    return Xi
# %%
# Solve our system
lam =0.025
Xi = STLS(lam, Xi,X_dot,Theta)
print(Xi)
# %%
# function to print equations
def print_eq(combos, Xi):
    symbs = ["x`", "y`","z`"]
    for i in range(3):
        s = symbs[i] + " = "
        current_coeffs = Xi[:,i]
        for j in range(len(current_coeffs)):
            if current_coeffs[j] != 0:
                c = combos[j]
                s += '{:.3g}'.format(current_coeffs[j]) + 'x'*c[0] + 'y'*c[1] + 'z'*c[2] + ' + '
        print(s[:-2])
# %%
print_eq(combos,Xi)
# %%
# WHO AM I?

# read in the csv and select columns as required
import pandas as pd

df = pd.read_csv('whoamI.csv',header =None)

X_new = df.iloc[:, 0:3].to_numpy()
X_dot_new = df.iloc[:, 3:].to_numpy()
# %%
# lets find the equation, using just the cubics from before 

Theta_new = gen_Theta(combos,N,X_new)
Xi_new = xlstsq(Theta_new,X_dot_new)
Xi_new = STLS(lam,Xi_new,X_dot_new,Theta_new)
print_eq(combos,Xi_new)


# %%
# lets test our found equation vs the given data 

#diff func for found equation
def whoami_diff(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ -y -z ,x + 0.1*y,  0.1 -18*z + x*y]


# %%
dt = 0.02
N = 10000
initial_state_new = X_new[0]
t0 = 0
t_bound = dt*N

#create solver object
solver_obj_new = DOP853(whoami_diff,t0,initial_state_new,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X_new_gen = np.zeros((N+1,3))
X_new_gen[0] = initial_state_new
#%%
#compute output
for i in range(N):
    solver_obj_new.step()
    X_new_gen[i+1] = solver_obj_new.y
#%%
#plot the phase space (attractor)
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 16})
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_new[:,0],X_new[:,1],X_new[:,2], lw = 1, label = 'real trajectory')
ax.plot3D(X_new_gen[:,0],X_new_gen[:,1],X_new_gen[:,2], lw = .4, label = 'reconstructed trajectory')
ax.legend(loc = 'best')
plt.show()
# %%
# lets check the actual time series in each dimension 
times = np.arange(0,dt*N+dt,dt)
tits = ['x','y','z']
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(times, X_new[:,i],'k', lw = 2)
    plt.plot(times, X_new_gen[:,i],'--r', lw = 1)
    plt.xlabel('time')
    plt.ylabel(tits[i])

# %%

# %%
