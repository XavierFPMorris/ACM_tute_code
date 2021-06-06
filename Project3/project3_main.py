#%%
## DEPENDENCIES
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,tan,exp 
from scipy.integrate import DOP853,RK45
import itertools
import pandas as pd

#%%
## HELPER FUNCTIONS

# Lorenz System
def Lorenz63(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ 10*(y-x),28*x - y - x*z,  -8/3*z + x*y]

# FD 2nd Order
def FD_deriv_2nd(vec,dt):
    f = [1/(2*dt),0,-1/(2*dt)]
    out = np.convolve(vec,f,'same')
    out[0] = (-1/2*vec[2] + 2*vec[1] - 3/2*vec[0])/dt
    out[-1] = (3/2*vec[-1] -2*vec[-2]+1/2*vec[-3])/dt
    return out

# FD 4th/3rd Order
def FD_deriv_3rd(vec,dt):
    f = [-1/(12*dt), 2/(3*dt),0,-2/(3*dt),1/(12*dt)]
    out = np.convolve(vec,f,'same')
    out[0] = (1/3*vec[3]-3/2*vec[2] + 3*vec[1] - 11/6*vec[0])/dt
    out[1] = (1/3*vec[4]-3/2*vec[3] + 3*vec[2] - 11/6*vec[1])/dt
    out[-1] = (11/6*vec[-1] -3*vec[-2]+3/2*vec[-3]-1/3*vec[-4])/dt
    out[-2] = (11/6*vec[-2] -3*vec[-3]+3/2*vec[-4]-1/3*vec[-5])/dt
    return out

# FD 8th/6th Order

def FD_deriv_8th(vec,dt):
    f = [-1/(280*dt),4/(105*dt),-1/(5*dt),4/(5*dt),0,-4/(5*dt),1/(5*dt),-4/(105*dt),1/(280*dt)]
    out = np.convolve(vec,f,'same')
    for i in range(0,4):
        out[0+i] = (-1/6*vec[6+i] + 6/5*vec[5+i]-15/4*vec[4+i] + 20/3*vec[3+i]-15/2*vec[2+i] + 6*vec[1+i] - 49/20*vec[0+i])/dt
        out[-(i+1)] = (1/6*vec[-(7+i)] - 6/5*vec[-(6+i)]+15/4*vec[-(5+i)] - 20/3*vec[-(4+i)]+15/2*vec[-(3+i)] - 6*vec[-(2+i)] + 49/20*vec[-(1+i)])/dt
    return out

# Library matrix func
def gen_Theta(combos,N,X):
    Theta = np.zeros((N+1,len(combos)))
    for (i,combo) in enumerate(combos):
        x,y,z = combo
        if isinstance(x,(int,float)) and isinstance(y,(int,float)) and isinstance(z,(int,float)):
            Theta[:,i] = (X[:,0]**x)*(X[:,1]**y)*(X[:,2]**z)
        else:
            fx = lambda a: eval(str(x))
            fy = lambda a: eval(str(y))
            fz = lambda a: eval(str(z))
            Theta[:,i] = fx(X[:,0]) + fy(X[:,1]) + fz(X[:,2])
    return Theta

# Least squares function
def xlstsq(Thet,Xdot):
    return np.linalg.inv( np.transpose(Thet)@Thet)@np.transpose(Thet)@Xdot

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
    return Xi

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

# print for 2 params
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

# gen theta for two params
def gen_Theta_2d(combos,N,X):
    Theta = np.zeros((len(X),len(combos)))
    for (i,combo) in enumerate(combos):
        x,y = combo
        Theta[:,i] = (X[:,0]**x)*(X[:,1]**y)
    return Theta
#%%
# MAIN TESTING FUNCTION
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
    #print(Xi)
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
    # this is whats used to generate the reconstructed trajectory and attractor plots
    # uncomment as required etc.
    #     num_dim = 1
    # #time series
    #     times = np.arange(0,dt*N+dt,dt)
    #     tits = ['x','y','z']
    #     for i in range(num_dim):
    #         plt.subplot(3,1,i+1)
    #         plt.plot(times[trans:end_t], X[trans:end_t,i],'k', lw = 2, label = 'true TS')
    #         plt.plot(times[trans:end_t], X_reco[trans:end_t,i],'--r', lw = 1,  label = 'reco TS')
    #         plt.xlabel('time')
    #         plt.legend(loc='best')
    #         plt.ylabel(tits[i])
    #     plt.show()
    # or attractor
        plt.rcParams["figure.figsize"] = (12,12)
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X_reco[:,0],X_reco[:,1],X_reco[:,2], lw = 0.4, label = 'Traj 1')
        ax.plot3D(initial_state[0],initial_state[1],initial_state[2],'*', label = 'IC 1')
        ax.set(xlabel = 'x',ylabel = 'y', zlabel = 'z')
        ax.legend(loc = 'best')
        plt.show()
    return np.mean(errors)

#%%
# FD test plot
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
#%%
# Butterfly attractor plot (with two trajectories)
dt = 0.02
N = 10000
initial_state = [2,10,5]
initial_state_2 = [2.05,9.95,5.05]
t0 = 0
t_bound = dt*N
solver_obj = DOP853(Lorenz63,t0,initial_state,t_bound,max_step=dt, first_step=dt)
solver_obj_2 = DOP853(Lorenz63,t0,initial_state_2,t_bound,max_step=dt, first_step=dt)
#set up data matrix
X = np.zeros((N+1,3))
X[0] = initial_state
#compute output
for i in range(N):
    solver_obj.step()
    X[i+1] = solver_obj.y
X_2 = np.zeros((N+1,3))
X_2[0] = initial_state
for i in range(N):
    solver_obj_2.step()
    X_2[i+1] = solver_obj_2.y
# plot 
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
# plot to compare derivatives of lorenz system (x)
st = 3200
en = 3250
plt.plot(X_dot[st:en,0],lw = 3, label = 'Analytic')
plt.plot(FD_deriv_2nd(X[st:en,0],dt),label = '2nd O')
plt.plot(FD_deriv_3rd(X[st:en,0],dt),label = '4th O')
plt.plot(FD_deriv_8th(X[st:en,0],dt),label = '8th O')
plt.ylabel('x`')
plt.xlabel('t')
plt.legend(loc = 'best')
#%%
# method for generating cubic function library requires input with 
# list of powers for x,y,z
# cubic powers
inds = [0, 1, 2, 3]
combos = [list(p) for p in itertools.product(inds, repeat=3) if np.sum(p)<=3]
combos = sorted(combos, key = lambda x: (np.sum(x),x[2],x[1],x[0]),reverse=False)
#%%
# generate the ideal SINDy equations
N = 10000
lam = 0.025
Theta = gen_Theta(combos,N,X)
Xi = xlstsq(Theta,X_dot)
Xi = STLS(lam, Xi,X_dot,Theta)
print_eq(combos,Xi)
#%%
# bar plot for derivative method errors (this was semi hard coded)
dt = 0.02
N = 10000
lam = 0.025
initial_state = [2,5,10]
ers = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos) # do for each of the methods 
# just copy-pasted in answer
plt.bar(["Analytical","2nd O","4th O","8th O"], [1.3728870948740245e-05, 6.1855137264114175,2.8735153889858176,0.6255906336770726])
plt.ylabel("Error")
#%%
# NOISE ERROR
dt = 0.02
N = 10000
lam = 0.025
initial_state = [2,5,10]
SNRS = np.array([0,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.12,0.15,0.17,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
noise_errors = np.zeros(len(SNRS))
for i,n in enumerate(SNRS):
    print(n)
    noise_errors[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, n, False)
plt.plot(SNRS,noise_errors, '-*k')
plt.xlabel('SNR')
plt.ylabel('Error')
#%%
# LAMBDA ERROR
dt = 0.02
N = 10000
lams = [0,0.005,0.01,0.03,0.05,0.1,0.5,0.8,0.9,1,1.5]
initial_state = [2,5,10]
noise = 0
lam_errors = np.zeros(len(lams))
for i,lam in enumerate(lams):
    print(lam)
    lam_errors[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, noise, False)
plt.plot(lams,lam_errors, '-*k')
plt.xlabel('lambda')
plt.ylabel('Error')
#%%
# DT ERROR
dts = np.array([0.001,0.005,0.01,0.02,0.025,0.03,0.05,0.1,0.15])
dt_errs = np.zeros(len(dts))
initial_state = [2,5,10]
lam = 0.025
noise = 0
N = 10000
for i,dt in enumerate(dts):
    print(dt)
    dt_errs[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, noise, False)
plt.plot(dts,dt_errs, '-*k')
plt.xlabel('dt')
#%%
# Different library functions
# First just higher powers (up to 6)
initial_state = [2,5,10]
lam = 0.025
noise = 0
N = 10000
dt = 0.02
inds = [0, 1, 2, 3, 4, 5, 6]
combos_h = [list(p) for p in itertools.product(inds, repeat=3) if np.sum(p)<=6]
combos_h = sorted(combos_h, key = lambda x: (np.sum(x),x[2],x[1],x[0]),reverse=False)
higher_er = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos_h, 0, True)
# %%
# Trying some trig
combos.extend([['sin(a)',0,0],[0,'sin(a)',0],[0,0,'sin(a)'],['tan(a)',0,0],[0,'tan(a)',0],[0,0,'tan(a)']])
#%%
N = 10000
lam = 0.025
N = 10000
dt = 0.02
X_dot_approx_8th = np.zeros((N+1,3))
X_dot_approx_8th[:,0] = FD_deriv_8th(X[:,0],dt)
X_dot_approx_8th[:,1] = FD_deriv_8th(X[:,1],dt)
X_dot_approx_8th[:,2] = FD_deriv_8th(X[:,2],dt)
Theta = gen_Theta(combos,N,X)
Xi = xlstsq(Theta,X_dot_approx_8th)
Xi = STLS(lam, Xi,X_dot_approx_8th,Theta)
print(Xi)

# %%
# Trying different lambdas for 40% SNR
dt = 0.02
N = 10000
lams = np.linspace(0,0.25,10)
initial_state = [2,5,10]
noise = 0.4
lam_errors_2 = np.zeros(len(lams))
for i,lam in enumerate(lams):
    print(lam)
    lam_errors_2[i] = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos, noise, True)
plt.plot(lams,lam_errors_2, '-*k')
plt.xlabel('lambda')
plt.ylabel('Error')
# %%
# Library function increase with small noise
initial_state = [2,5,10]
lam = 0.025
noise = 0.03
N = 10000
dt = 0.02
inds = [0, 1, 2, 3, 4, 5, 6]
combos_h = [list(p) for p in itertools.product(inds, repeat=3) if np.sum(p)<=6]
combos_h = sorted(combos_h, key = lambda x: (np.sum(x),x[2],x[1],x[0]),reverse=False)
higher_er_2 = auto_test_Lorenz63(dt,N,lam, FD_deriv_8th, initial_state, combos_h, noise, True)
print(higher_er_2)
# %%
# WHO AM I?

# read in the csv and select columns as required

df = pd.read_csv('whoamI.csv',header =None)

X_new = df.iloc[:, 0:3].to_numpy()
X_dot_new = df.iloc[:, 3:].to_numpy()

# lets find the equation, using just the cubics from before 
lam = 0.025
Theta_new = gen_Theta(combos,N,X_new)
Xi_new = xlstsq(Theta_new,X_dot_new)
Xi_new = STLS(lam,Xi_new,X_dot_new,Theta_new)
print_eq(combos,Xi_new)
# %%
# plotting the attractor and the recreated attractor

#diff func for found equation
def whoami_diff(t,vals):
    x,y,z = vals[0],vals[1],vals[2]
    return [ -y -z ,x + 0.1*y,  0.1 -18*z + x*y]


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
#compute output
for i in range(N):
    solver_obj_new.step()
    X_new_gen[i+1] = solver_obj_new.y

#plot the phase space (attractor)
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams.update({'font.size': 16})
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X_new[:,0],X_new[:,1],X_new[:,2], 'k',lw = 2, label = 'real trajectory')
ax.plot3D(X_new_gen[:,0],X_new_gen[:,1],X_new_gen[:,2], '--r',lw = 1, label = 'reconstructed trajectory')
ax.set(xlabel = 'x',ylabel = 'y', zlabel = 'z')
ax.legend(loc = 'best')
plt.show()
# %%
# plot each time series
times = np.arange(0,dt*N+dt,dt)
tits = ['x','y','z']
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(times, X_new[:,i],'k', lw = 2,label = 'real traj')
    plt.plot(times, X_new_gen[:,i],'--r', lw = 1 ,label = 'reco traj')
    plt.xlabel('time')
    plt.ylabel(tits[i])
    plt.legend(loc='best')
# %%
### Henon map
# generate for some time and then cut off transient
N  = 20
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
X_Henon = np.zeros((N+1,3))
X_Henon[:,0] = x[eta-2:-1]
X_Henon[:,1] = y[eta-2:-1]
X_Henon[:,2] = z[eta-2:-1]
dX_Henon = np.zeros((N+1,3))
dX_Henon[:,0] = x[eta-1:]
dX_Henon[:,1] = y[eta-1:]
dX_Henon[:,2] = z[eta-1:]
#%%
# plot the lovely attractor
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_Henon[:,0],X_Henon[:,1],X_Henon[:,2], 'k',s= 0.4)
ax.set(xlabel = 'x',ylabel = 'y', zlabel = 'z')
plt.show()
#%%
# plot x time series
plt.plot(X_Henon[eta+10000:eta+10300,0],'-*k')
plt.xlabel('i')
plt.ylabel('x')
# %%
# SINDy
Theta_Henon = gen_Theta(combos,N,X_Henon)
Xi_Henon = xlstsq(Theta_Henon,dX_Henon)
Xi_Henon = STLS(lam,Xi_Henon,dX_Henon,Theta_Henon)
print_eq(combos,Xi_Henon)
#%%
# cubic helper for 2 variables (instead of 3 as previous)
inds = [0, 1, 2,3]
combos_2 = [list(p) for p in itertools.product(inds, repeat=2) if np.sum(p)<=3]
combos_2 = sorted(combos_2, key = lambda x: (np.sum(x),x[1],x[0]),reverse=False)

# %%
# Normal form test
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
    initial_state = [0.1*np.random.randn(),u]
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

x_hyst = np.vstack(xs)
x_dot_hyst = np.vstack(x_dots)

Theta_hyst = gen_Theta_2d(combos_2,N,x_hyst)
Xi_hyst= xlstsq(Theta_hyst,x_dot_hyst)
Xi_hyst = STLS(lam,Xi_hyst,x_dot_hyst,Theta_hyst)
print_eq_2(combos_2,Xi_hyst)

# Note that it doesnt print u` equal to anything because call entries of Xi are zero.
# %%
