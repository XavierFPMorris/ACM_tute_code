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
initial_state = [2,10,5]
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
initial_state_2 = [2.05,9.95,5.05]
t0 = 0
t_bound = dt*N

#create solver object
solver_obj_2 = DOP853(Lorenz63,t0,initial_state_2,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X_2 = np.zeros((N+1,3))
X_2[0] = initial_state
#%%
#compute output
for i in range(N):
    solver_obj_2.step()
    X_2[i+1] = solver_obj_2.y
#%%
#plot the phase space (attractor)
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 16})
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X[:,0],X[:,1],X[:,2], lw = 0.4, label = 'Traj 1')
ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*', label = 'IC 1')
ax.plot3D(X_2[:,0],X_2[:,1],X_2[:,2], lw = 0.4, label = 'Traj 2')
ax.plot3D(initial_state_2[0],initial_state_2[1],initial_state_2[2],'*', label = 'IC 2')
ax.set(xlabel = 'x',ylabel = 'y', zlabel = 'z')
ax.legend(loc = 'best')
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
def FD_deriv_2nd(vec,dt):
    f = [1/(2*dt),0,-1/(2*dt)]
    out = np.convolve(vec,f,'same')
    out[0] = (-1/2*vec[2] + 2*vec[1] - 3/2*vec[0])/dt
    out[-1] = (3/2*vec[-1] -2*vec[-2]+1/2*vec[-3])/dt
    return out


#%%
#calculate derivatives from finite differences 
X_dot_approx = np.zeros((N+1,3))

X_dot_approx[:,0] = FD_deriv_2nd(X[:,0],dt)
X_dot_approx[:,1] = FD_deriv_2nd(X[:,1],dt)
X_dot_approx[:,2] = FD_deriv_2nd(X[:,2],dt)

#%%
# plot to compare derivatives
st = 3200
en = 3250
plt.plot(X_dot[st:en,0],lw = 3, label = 'Analytic')
plt.plot(FD_deriv_2nd(X[st:en,0],dt),label = '2nd O')
plt.plot(FD_deriv_3rd(X[st:en,0],dt),label = '3rd O')
plt.plot(FD_deriv_8th(X[st:en,0],dt),label = '8th O')
plt.ylabel('x`')
plt.xlabel('t')
plt.legend(loc = 'best')
#%%
#Check derivatives are close
print(np.mean(np.abs(X_dot - X_dot_approx)))
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

for c in combos:
    print('x^{} * y^{} * z^{}'.format(c[0],c[1],c[2]))

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
# initial guess from least squares using exact derivatives
Xi = xlstsq(Theta,X_dot)

# %%
#Sequential thresholding function
def STLS(lam, Xi, X_dot,Theta):
    cond = True
    count = 0
    while cond:
        smalls = (np.abs(Xi)<lam)
        Xi[smalls] = 0
        old = Xi.copy()
        for ind in range(np.min(X_dot.shape)):
            bigs= ~smalls[:,ind]
            Xi[bigs,ind] = xlstsq( Theta[:,bigs],X_dot[:,ind])
        if np.abs(np.mean(Xi-old))<10**(-10):
            cond = False
        count += 1
        if count>1000:
            cond = False
            print("failed")
        #print(Xi)## THIS ONE
    return Xi
# %%
# Solve our system using exact derivatives
lam =0.025
Xi = STLS(lam, Xi,X_dot,Theta)
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
                s += '{:.10f}'.format(current_coeffs[j]) + 'x'*c[0] + 'y'*c[1] + 'z'*c[2] + ' + '
        print(s[:-2])
# %%
print_eq(combos,Xi)

#%%
# function to return equations, allow automated testing
def ret_eq(combos, Xi):
    #symbs = ["x`", "y`","z`"]
    strs = [0]*3
    for i in range(3):
        switch = 0
        s = ""
        current_coeffs = Xi[:,i]
        for j in range(len(current_coeffs)):
            if current_coeffs[j] != 0:
                c = combos[j]
                s += '{:.10f}'.format(current_coeffs[j]) + '*x'*c[0] + '*y'*c[1] + '*z'*c[2] + ' + '
                switch = 1
        if switch == 1:
            strs[i] = s[:-3]
        else:
            strs[i] = '0'
    return (lambda x,y,z: eval(strs[0]),lambda x,y,z: eval(strs[1]) , lambda x,y,z: eval(strs[2]))

#%% testing the ideal result
(fx,fy,fz) = ret_eq(combos,Xi)

t0 = 0
t_bound = dt*N
solver_obj = DOP853(Lorenz63,t0,initial_state,t_bound,max_step=dt, first_step=dt)
X = np.zeros((N+1,3))
X[0] = initial_state
#compute output
for i in range(N):
    solver_obj.step()
    X[i+1] = solver_obj.y

def Lorenz_Reco(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ fx(x,y,z) ,fy(x,y,z) , fz(x,y,z) ]


t0 = 0
t_bound = dt*N

#create solver object
solver_obj_reco = DOP853(Lorenz_Reco,t0,initial_state,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X_reco= np.zeros((N+1,3))
X_reco[0] = initial_state

#compute output
for i in range(N):
    solver_obj_reco.step()
    X_reco[i+1] = solver_obj_reco.y

#errors 

errors = np.mean(np.abs(X[100:601,:] - X_reco[100:601,:]) , axis = 0)

#plots

#time series
times = np.arange(0,dt*N+dt,dt)
tits = ['x','y','z']
for i in range(1):
    plt.subplot(3,1,i+1)
    plt.plot(times[100:601], X[100:601,i],'k', lw = 2, label = 'true TS')
    plt.plot(times[100:601], X_reco[100:601,i],'--r', lw = 1,  label = 'reco TS')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.ylabel(tits[i])


print(np.mean(errors))


# %%
#now what happens if we use the 2nd O FD derivatives?
Xi_Lorenz_FD = xlstsq(Theta,X_dot_approx)
Xi_Lorenz_FD = STLS(lam, Xi_Lorenz_FD,X_dot_approx,Theta)
print_eq(combos,Xi_Lorenz_FD)

#this isnt great

# %%
#what about 4th central 3rd orderforward and back  ? (forgive me for the mislabeled functions)
def FD_deriv_3rd(vec,dt):
    f = [-1/(12*dt), 2/(3*dt),0,-2/(3*dt),1/(12*dt)]
    out = np.convolve(vec,f,'same')
    out[0] = (1/3*vec[3]-3/2*vec[2] + 3*vec[1] - 11/6*vec[0])/dt
    out[1] = (1/3*vec[4]-3/2*vec[3] + 3*vec[2] - 11/6*vec[1])/dt
    out[-1] = (11/6*vec[-1] -3*vec[-2]+3/2*vec[-3]-1/3*vec[-4])/dt
    out[-2] = (11/6*vec[-2] -3*vec[-3]+3/2*vec[-4]-1/3*vec[-5])/dt
    return out

# %%

# 8th central, 6th forward n back

def FD_deriv_8th(vec,dt):
    f = [-1/(280*dt),4/(105*dt),-1/(5*dt),4/(5*dt),0,-4/(5*dt),1/(5*dt),-4/(105*dt),1/(280*dt)]
    out = np.convolve(vec,f,'same')
    for i in range(0,4):
        out[0+i] = (-1/6*vec[6+i] + 6/5*vec[5+i]-15/4*vec[4+i] + 20/3*vec[3+i]-15/2*vec[2+i] + 6*vec[1+i] - 49/20*vec[0+i])/dt
        out[-(i+1)] = (1/6*vec[-(7+i)] - 6/5*vec[-(6+i)]+15/4*vec[-(5+i)] - 20/3*vec[-(4+i)]+15/2*vec[-(3+i)] - 6*vec[-(2+i)] + 49/20*vec[-(1+i)])/dt
    return out
#%%
#calculate derivatives from finite differences 

X_dot_approx_3rd = np.zeros((N+1,3))

X_dot_approx_3rd[:,0] = FD_deriv_3rd(X[:,0],dt)
X_dot_approx_3rd[:,1] = FD_deriv_3rd(X[:,1],dt)
X_dot_approx_3rd[:,2] = FD_deriv_3rd(X[:,2],dt)

#%%
# test 
print(np.mean(np.abs(X_dot - X_dot_approx_3rd)))
# this is better, lets use it for SINDy

#%%
Xi_Lorenz_FD_3rd = xlstsq(Theta,X_dot_approx_3rd)
Xi_Lorenz_FD_3rd = STLS(lam, Xi_Lorenz_FD_3rd,X_dot_approx_3rd,Theta)
print_eq(combos,Xi_Lorenz_FD_3rd)

# almost perfect :)

# %%

#Lets take it a step further and try the 8th order accuracy center and 6th forward and back
N = 10000
X_dot_approx_8th = np.zeros((N+1,3))

X_dot_approx_8th[:,0] = FD_deriv_8th(X[:,0],dt)
X_dot_approx_8th[:,1] = FD_deriv_8th(X[:,1],dt)
X_dot_approx_8th[:,2] = FD_deriv_8th(X[:,2],dt)

#%%
# test 
print(np.mean(np.abs(X_dot - X_dot_approx_8th)))
print([X[:,0]])
print(X_dot[:,0])
print(X_dot_approx_8th[:,0])
#%%
Xi_Lorenz_FD_8th = xlstsq(Theta,X_dot_approx_8th)
Xi_Lorenz_FD_8th = STLS(lam, Xi_Lorenz_FD_8th,X_dot_approx_8th,Theta)
print_eq(combos,Xi_Lorenz_FD_8th)

#pretty much exact, just like the analytical derivative
#%%
# lets now compare the real attractor to that recreated by our FD equations 

## 2nd order 

def Lorenz2nd(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ -9.9112*x + 9.9112*y ,27.204*x + -0.84456*y + -0.97777*x*z , 2.0922 + -2.7547*z + 0.032008*x*x + 0.96766*x*y ]
#%%
#set up initial conditions
dt = 0.02
N = 10000
initial_state = [2,10,5]
t0 = 0
t_bound = dt*N

#create solver object
solver_obj_L_2nd = DOP853(Lorenz2nd,t0,initial_state,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X_L_2nd = np.zeros((N+1,3))
X_L_2nd[0] = initial_state
#%%
#compute output
for i in range(N):
    solver_obj_L_2nd.step()
    X_L_2nd[i+1] = solver_obj_L_2nd.y

#%%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_L_2nd[:,0],X_L_2nd[:,1],X_L_2nd[:,2], lw = 1, label = '2nd O reco trajectory')
ax.plot3D(X[:,0],X[:,1],X[:,2], lw = 0.4, label = 'real trajectory')
ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*')
ax.legend(loc = 'best')
plt.show()
#%%
## 3rd order 

def Lorenz3rd(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ -9.9979*x + 9.998*y ,27.965*x + -0.99318*y + -0.99906*x*z , -2.6654*z + 0.99959*x*y ]
#%%
#set up initial conditions
dt = 0.02
N = 10000
initial_state = [2,10,5]
t0 = 0
t_bound = dt*N

#create solver object
solver_obj_L_3rd = DOP853(Lorenz3rd,t0,initial_state,t_bound,max_step=dt, first_step=dt)

#set up data matrix
X_L_3rd = np.zeros((N+1,3))
X_L_3rd[0] = initial_state
#%%
#compute output
for i in range(N):
    solver_obj_L_3rd.step()
    X_L_3rd[i+1] = solver_obj_L_3rd.y

#%%

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_L_3rd[:,0],X_L_3rd[:,1],X_L_3rd[:,2], lw = 1, label = '3rd O reco trajectory')
ax.plot3D(X[:,0],X[:,1],X[:,2], lw = 0.4, label = 'real trajectory')
ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*')
ax.legend(loc = 'best')
plt.show()

# %%
# Now what if there is measurement noise? 
# We will only consider additive noise to the X, the calculate our FD and then SINDy, 
# not much point adding noise to analytical derivative as this has no experimental relevance
# lets use gaussian noise, start with a 10% signal to noise ratio

xsd = np.std(X)

noise = 0.1*xsd*np.random.randn(10001,3)

noise_X = X + noise


#%%

# lets start with the 4th order derivative
N = 10000

X_dot_approx_4th_Noise = np.zeros((N+1,3))

X_dot_approx_4th_Noise[:,0] = FD_deriv_3rd(noise_X[:,0],dt)
X_dot_approx_4th_Noise[:,1] = FD_deriv_3rd(noise_X[:,1],dt)
X_dot_approx_4th_Noise[:,2] = FD_deriv_3rd(noise_X[:,2],dt)

#%%
# test 
print(np.mean(np.abs(X_dot - X_dot_approx_4th_Noise)))
# this is better, lets use it for SINDy

#%%
Theta_Noise  = gen_Theta(combos,N,noise_X)
Xi_Lorenz_FD_4th_Noise = xlstsq(Theta_Noise,X_dot_approx_4th_Noise)
Xi_Lorenz_FD_4th_Noise = STLS(0.025, Xi_Lorenz_FD_4th_Noise,X_dot_approx_4th_Noise,Theta_Noise)
print_eq(combos,Xi_Lorenz_FD_4th_Noise)


# Not great, lets try the 8th order derivative



#%%
X_dot_approx_8th_Noise = np.zeros((N+1,3))

X_dot_approx_8th_Noise[:,0] = FD_deriv_8th(noise_X[:,0],dt)
X_dot_approx_8th_Noise[:,1] = FD_deriv_8th(noise_X[:,1],dt)
X_dot_approx_8th_Noise[:,2] = FD_deriv_8th(noise_X[:,2],dt)

#%%
# test 
print(np.mean(np.abs(X_dot - X_dot_approx_8th_Noise)))
# this is better, lets use it for SINDy

#%%

Xi_Lorenz_FD_8th_Noise = xlstsq(Theta_Noise,X_dot_approx_8th_Noise)
Xi_Lorenz_FD_8th_Noise = STLS(0.35, Xi_Lorenz_FD_8th_Noise,X_dot_approx_8th_Noise,Theta_Noise)
print_eq(combos,Xi_Lorenz_FD_8th_Noise)

# still not great, showing that finite differencing is incredibly un-robust to noise
# could try the differentiaton used in the paper
#%%
# def ret_eqs(combos, Xi):
#     #symbs = ["x`", "y`","z`"]
#     strs = [0]*3
#     for i in range(3):
#         s = ""
#         current_coeffs = Xi[:,i]
#         for j in range(len(current_coeffs)):
#             if current_coeffs[j] != 0:
#                 c = combos[j]
#                 s += '{:.5g}'.format(current_coeffs[j]) + '*x'*c[0] + '*y'*c[1] + '*z'*c[2] + ' + '
#         strs[i] = s[:-3]
#     return strs

# print(ret_eqs(combos,Xi))
#%%

# Big function for automated testing

def auto_test_Lorenz63(dt,N,lam, deriv, initial_state, lib_combos, nnoise = 0, vis = False):
    t0 = 0
    t_bound = dt*N
    solver_obj = DOP853(Lorenz63,t0,initial_state,t_bound,max_step=dt, first_step=dt)
    X = np.zeros((N+1,3))
    X[0] = initial_state
    #compute output
    for i in range(N):
        solver_obj.step()
        X[i+1] = solver_obj.y
    X_dot = np.zeros((N+1,3))
    for i in range(3):
        X_dot[:,i] = deriv(X[:,i],dt)
    sdsX = np.std(X)
    sdsXdot = np.std(X_dot)
    X_n = X + nnoise*sdsX*np.random.randn(N+1,3)
    X_dot_n = X_dot + nnoise*sdsXdot*np.random.randn(N+1,3)
    Theta = gen_Theta(lib_combos,N,X_n)
    Xi = xlstsq(Theta,X_dot_n)
    Xi = STLS(lam,Xi,X_dot_n,Theta)
    (fx,fy,fz) = ret_eq(lib_combos,Xi)
    print_eq(lib_combos,Xi)
    def Lorenz_Reco(t,vals):
        x,y,z = vals[0],vals[1],vals[2]
        return [ fx(x,y,z) ,fy(x,y,z) , fz(x,y,z) ]
    t0 = 0
    t_bound = dt*N
    #create solver object
    solver_obj_reco = DOP853(Lorenz_Reco,t0,initial_state,t_bound,max_step=dt, first_step=dt)
    #set up data matrix
    X_reco= np.zeros((N+1,3))
    X_reco[0] = initial_state
    #compute output
    for i in range(N):
        solver_obj_reco.step()
        X_reco[i+1] = solver_obj_reco.y
        if solver_obj_reco.status != 'running': break
    #transient and error time 
    trans = int(2/dt)
    end_t = int(10/dt) + trans+1
    errors = np.mean(np.abs(X[trans:end_t,:] - X_reco[trans:end_t,:]) , axis = 0)
    #plots
    if vis:
    #time series
        times = np.arange(0,dt*N+dt,dt)
        tits = ['x','y','z']
        for i in range(1):
            plt.subplot(3,1,i+1)
            plt.plot(times[trans:end_t], X[trans:end_t,i],'k', lw = 2, label = 'true TS')
            plt.plot(times[trans:end_t], X_reco[trans:end_t,i],'--r', lw = 1,  label = 'reco TS')
            plt.xlabel('time')
            plt.legend(loc='best')
            plt.ylabel(tits[i])
    ## or attractor
    #     plt.rcParams["figure.figsize"] = (12,12)
    #     plt.rcParams.update({'font.size': 16})
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.plot3D(X_reco[:,0],X_reco[:,1],X_reco[:,2], lw = 0.4, label = 'Traj 1')
    #     ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*', label = 'IC 1')
    #     ax.set(xlabel = 'x',ylabel = 'y', zlabel = 'z')
    # #ax.legend(loc = 'best')
    return np.mean(errors)

#%%
dt = 0.02
N = 10000
lam = 0.025
initial_state = [2,5,10]

import time


ers = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, 0.2, True)
print(np.mean(ers))



#anal 1.3728870948740245e-05
#2nd 6.1855137264114175
#4th 2.8735153889858176
#8th 0.6255906336770726

#%%
# plotting errors for different diff methods

plt.bar(["Analytical","2nd O","4th O","8th O"], [1.3728870948740245e-05, 6.1855137264114175,2.8735153889858176,0.6255906336770726])
plt.ylabel("Error")

#%%
dt = 0.02
N = 10000
lam = 0.025
initial_state = [2,5,10]
# testing noise robustness
SNRS = np.array([0,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.12,0.15,0.17,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

noise_errors = np.zeros(len(SNRS))

for i,n in enumerate(SNRS):
    print(n)
    noise_errors[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, n, False)
#%%

plt.plot(SNRS,noise_errors, '-*k')
plt.xlabel('SNR')
plt.ylabel('Error')

#%%

#testing impact of lambda
dt = 0.02
N = 10000
lams = [0,0.005,0.01,0.03,0.05,0.1,0.5,0.8,0.9,1,1.5]
initial_state = [2,5,10]

noise = 0

lam_errors = np.zeros(len(lams))

for i,lam in enumerate(lams):
    print(lam)
    lam_errors[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, noise, False)


#%%
plt.plot(lams,lam_errors, '-*k')
plt.xlabel('lambda')
plt.ylabel('Error')

#%%
# testing 
er = auto_test_Lorenz63(dt,N,1, FD_deriv_8th, initial_state, combos, 0, True)


#%%
# testing dt 
dts = np.array([0.001,0.005,0.01,0.02,0.025,0.03,0.05,0.1,0.15])
dt_errs = np.zeros(len(dts))
initial_state = [2,5,10]
lam = 0.025
noise = 0
N = 10000
for i,dt in enumerate(dts):
    print(dt)
    dt_errs[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, noise, False)
#%%
plt.plot(dts,dt_errs, '-*k')
plt.xlabel('dt')
plt.ylabel('Error')
#%%
auto_test_Lorenz63(0.1,N,lam, FD_deriv_8th, initial_state, combos, noise, True)
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

# there is a perfect match
# %%

# now let us test this with our finite differences and some measurement noise


'''



YARN GOES HERE



'''
# %%

# Now let us test it with a tricky example. the 3D henon map with hyperchaotic parameters 

# generate for 50000 and then cut of 2000 to ensure there is no transient

N  = 40000
eta = 2000

x = np.zeros(N+eta)
y = np.zeros(N+eta)
z = np.zeros(N+eta)

x[0] = 0.1*np.random.randn()
y[0] = 0.1*np.random.randn()
z[0] = 0.1*np.random.randn()

a = 1.76
b = 0.1

for i in range(1,eta+N):
    x[i] = a - y[i-1]**2 - b*z[i-1]
    y[i] = x[i-1]
    z[i] = y[i-1]

x_Henon = x[eta-2:-1]
y_Henon = y[eta-2:-1]
z_Henon = z[eta-2:-1]

X_Henon = np.zeros((N+1,3))

X_Henon[:,0] = x_Henon
X_Henon[:,1] = y_Henon
X_Henon[:,2] = z_Henon
#%%

# plot the lovely attractor

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_Henon,y_Henon,z_Henon, 'k',s= 0.4)
plt.show()

#%%

#plot x time series 

plt.plot(x_Henon[eta+10000:eta+10500])

#%%
# now we use the discrete time representation




dX_Henon = np.zeros((N+1,3))

dX_Henon[:,0] = x[eta-1:]
dX_Henon[:,1] = y[eta-1:]
dX_Henon[:,2] = z[eta-1:]


 

# %%

#Lets restrict to just quadratic polynomials

inds = [0, 1, 2]
combos_H = [list(p) for p in itertools.product(inds, repeat=3) if np.sum(p)<=2]
##-1*x.count(0),np.max(x),
combos_H = sorted(combos_H, key = lambda x: (np.sum(x),x[2],x[1],x[0]),reverse=False)
print(combos_H)

# %%
#N  = 10000
Theta_Henon = gen_Theta(combos,N,X_Henon)
Xi_Henon = xlstsq(Theta_Henon,dX_Henon)
Xi_Henon = STLS(lam,Xi_Henon,dX_Henon,Theta_Henon)
print_eq(combos,Xi_Henon)

# this is perfect :) #can work with only 10 points !!!!
# %%





# show that it doesnt work with derivatives, but if we make the assumption
# that its a discrete map that it does 


# N  = 10
# eta = 200

# x = np.zeros(N+eta)
# y = np.zeros(N+eta)
# z = np.zeros(N+eta)

# x[0] = 0.1*np.random.randn()
# y[0] = 0.1*np.random.randn()
# z[0] = 0.1*np.random.randn()

# a = 1.76
# b = 0.1

# for i in range(1,eta+N):
#     x[i] = 1.1415 + -0.6614*y[i-1] + -0.66865*y[i-1]**2
#     y[i] = 1.1416 +  -0.66142*z[i-1] + -0.66855*z[i-1]**2
#     z[i] = -11.733 + 6.6202*x[i-1] + 0.66128*y[i-1]  + 6.6861*z[i-1]**2
#     print(x[i])
#     print(y[i])
#     print(z[i])

# x_Henon_reco = x[eta-1:]
# y_Henon_reco = y[eta-1:]
# z_Henon_reco = z[eta-1:]

#%%
# trying to recreate the logmap

N = 1000
mus = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 3.8, 3.85, 3.9, 3.95]
x = [np.zeros((N, 2)) for i in range(len(mus))]
for i, mu in enumerate(mus):
    x[i][0] = [0.5, mu]
    for k in range(1, N):
        x[i][k, 0] = mu * x[i][k - 1, 0] * (1 - x[i][k - 1, 0])
        x[i][k, 1] = mu

x_in = x

x_dot = [xi[1:] for xi in x_in]
x = [xi[:-1] for xi in x_in]

x_log = np.vstack(x)
x_dot_log = np.vstack(x_dot)
#%%
def gen_Theta_2d(combos,N,X):
    Theta = np.zeros((len(X),len(combos)))
    for (i,combo) in enumerate(combos):
        x,y = combo
        Theta[:,i] = (X[:,0]**x)*(X[:,1]**y)
    return Theta
# %%
inds = [0, 1, 2,3]
combos_2 = [list(p) for p in itertools.product(inds, repeat=2) if np.sum(p)<=3]
##-1*x.count(0),np.max(x),
combos_2 = sorted(combos_2, key = lambda x: (np.sum(x),x[1],x[0]),reverse=False)
print(combos_2)

#%%
def print_eq_2(combos, Xi):
    symbs = ["x`", "u`"]
    for i in range(2):
        s = symbs[i] + " = "
        current_coeffs = Xi[:,i]
        for j in range(len(current_coeffs)):
            if current_coeffs[j] != 0:
                c = combos[j]
                s += '{:.6g}'.format(current_coeffs[j]) + 'x'*c[0] + 'u'*c[1]+ ' + '
        print(s[:-2])
# %%
Theta_Log = gen_Theta_2d(combos_2,N,x_log)
Xi_Log= xlstsq(Theta_Log,x_dot_log)
Xi_Log = STLS(lam,Xi_Log,x_dot_log,Theta_Log)
print_eq_2(combos_2,Xi_Log)

#%%
print(x_log)
#%%
# What about a funky curve

#x' = mu + x(1-x^2)
#mu' = 0
N = 10000
dt = 0.02


t0 = 0
t_bound = dt*N
us = [-2,-1,-0.5,0,0.5,1,2]
xs = [np.zeros((N, 2)) for i in range(len(us))]
x_dots = [np.zeros((N, 2)) for i in range(len(us))]

for i,u in enumerate(us):
    initial_state = [-0.01,u]
    def diff_func(t,vals):
        x,U = vals[0],vals[1]
        return [U + x*(1-x**2),0]
    solver_hyst = DOP853(diff_func,t0,initial_state,t_bound,max_step=dt, first_step=dt)
    xs[i][0] = initial_state
    x_dots[i][0] = diff_func(0,initial_state)
    for k in range(1,N):
        solver_hyst.step()
        xs[i][k] = solver_hyst.y
        x_dots[i][k] = diff_func(0,solver_hyst.y)
#%%
plt.scatter(xs[-2][:,0], x_dots[-2][:,0])
#%%
x_hyst = np.vstack(xs)
x_dot_hyst = np.vstack(x_dots)
#%%
Theta_hyst = gen_Theta_2d(combos_2,N,x_hyst)
Xi_hyst= xlstsq(Theta_hyst,x_dot_hyst)
Xi_hyst = STLS(lam,Xi_hyst,x_dot_hyst,Theta_hyst)
print_eq_2(combos_2,Xi_hyst)

# %%


# %%

# doing some testing for the differencing 
import numpy as np
import matplotlib.pyplot as plt
dt = 0.1
tests = np.arange(-15,15,dt)

testssq = tests**3
testsder = 3*tests**2

fd_der = FD_deriv_8th(testssq,dt)

plt.plot(tests,fd_der, label = 'Finite Diff', lw = 3)
plt.plot(tests,testsder, label = 'Analytical')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'best')
print(np.mean(np.abs(fd_der-testsder)))
# %%
