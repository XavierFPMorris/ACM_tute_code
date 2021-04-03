import matplotlib.pyplot as plt 
import numpy as np
from math import pi
from numpy import cosh, exp, cos, zeros, log10, linspace, sqrt
from Project1.KDV_FUNCS import KDV_INT,KDV_INT_2, evo_plot
from scipy.signal import argrelextrema

import warnings
warnings.filterwarnings("ignore")


### INITIAL CONDITIONS ###

## SOLITON 
def soliton(c, x, length):
    return 0.5*c/(cosh((c)**0.5/2*(x-length/2)))**2 

## SOLITON 2
def soliton2(c, x, length):
    return 0.5*c/(cosh((c)**0.5/2*(x-length*(1+100*c)/2)))**2 

## GAUSSIAN
def gaussian(A,B,x,length):
    return A*exp(-B*(x-length/2)**2)



### GAUSSIAN AMPLITUDE DEPENDANCE ### 
gad = False
if gad:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #time step
    dt = 0.05 
    #time steps
    nsteps_ar = [0, 100, 300, 1000]
    #values of A to analyse
    A_ = 0.1*np.arange(5) + 0.05
    B = 0.1

    fig, axs = plt.subplots(2,2)

    #integrate for each value of A, to each time step
    for n, ax in enumerate(fig.axes):
        nsteps = nsteps_ar[n]
        #iterate over each amplitude
        for i in range(len(A_)):
            #initial condition case
            if n == 0:
                #generate initial condition
                ic = gaussian(A_[i], B, x, length)
                leg = "A = {a:.2f}"
                ax.plot(x, ic,label = leg.format(a = A_[i]))
            #evolution case
            else:
                #generate initial condition
                ic = gaussian(A_[i], B, x, length)
                u = KDV_INT(N, length, dt, nsteps, ic)
                leg = "A = {a:.2f}"
                ax.plot(x, np.real(u[-1,:]),label = leg.format(a = A_[i]))
        ax.set_title('Gaussian Solution after ' + str(nsteps) + ' steps with dt = ' + str(dt) + ', B = ' + str(B))
        ax.set_ylim([-0.1, 0.8])
        ax.legend(loc="upper left")
    plt.suptitle('Gaussian Long Time Dependance on initial amplitude')
    plt.show()


### ACCURACY ###

## DT ##
dt_ac = False 
if dt_ac:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #time step
    dt = np.logspace(-3,0.3, 20)
    #total time 
    tme = 5
    #time steps
    nsteps_ar = tme/dt
    #array to save rms error
    er = zeros(len(dt))
    #initial condition
    ic = soliton(0.05, x, length)

    #integrate for each dt
    for i in range(len(dt)):
        u = KDV_INT(N, length, dt[i], int(nsteps_ar[i]), ic)
        u0 = u[0,:]
        #initial Int(u^2) value that should be conserved
        u2_i = np.trapz(u0**2, x, dx)
        #array to save residuals
        res_ar = zeros(int(nsteps_ar[i]))
        res_ar[0] = 0
        #calculate deviation from expected value at each time step
        for n in range(1,int(nsteps_ar[i])):
            res_ar[n] = np.trapz(u[n,:]**2, x, dx) - u2_i
        #save rms error
        er[i] = np.sqrt(np.mean(res_ar**2))
    
    fig, ax = plt.subplots(1,1)
    #plot results
    ax.loglog(dt, er, 'ob')

    plot_text = "Slope = {s:.2f}"

    ## FITTING
    #calculate fit
    log_slope, log_intercept = np.polyfit(log10(dt), log10(er), 1)
    coeffs = np.polyfit(log10(dt), log10(er),1)
    polyn = np.poly1d(coeffs)
    log10_fit = polyn(log10(dt))
    #plot fit
    ax.loglog(dt, 10**log10_fit, '-r')
    ax.text(dt[15] - 0.01 , 10**(0.15 + log10_fit[15])  , plot_text.format(s = log_slope), horizontalalignment='right', c = 'r')

    ax.set(xlabel = 'log(dt)', ylabel = 'log(RMS($u^2$))', title = 'RMS of $u^2$ as a function of dt')
    plt.show()

## DX ##
dx_ac = False
if dx_ac:
    #number of grid points
    N = np.round(np.logspace(2,5,10))
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    print(dx)
    #time step
    dt = 0.01
    #time steps
    nsteps = 100
    #array to store RMS error
    er = zeros(len(N))

    #integrate for each dx
    for i in range(len(dx)):
        #spatial discretisation
        x = dx[i]*np.arange(N[i]) 
        #integrate (calculating the initial condition inline)
        u = KDV_INT(int(N[i]), length, dt, nsteps, soliton(0.05, x, length))
        u0 = u[0,:]
        #initial Int(u^2) value that should be conserved
        u2_i = np.trapz(u0**2, x, dx[i])
        #array to save residuals
        res_ar = zeros(nsteps)
        res_ar[0] = 0
        #calculate the deviation from expected value at each time step
        for n in range(1,nsteps):
            res_ar[n] = np.trapz(u[n,:]**2, x, dx[i]) - u2_i
        #save rms error 
        er[i] = np.sqrt(np.mean(res_ar**2))
    
    fig, ax = plt.subplots(1,1)
    #plot results
    ax.loglog(dx, er, 'ob')

    plot_text = "Slope = {s:.2f}"

    ## FIRST SEGMENT FIT

    log_slope1, log_intercept1 = np.polyfit(log10(dx[5:]), log10(er[5:]),1)
    coeffs1 = np.polyfit(log10(dx[5:]), log10(er[5:]),1)
    polyn1 = np.poly1d(coeffs1)
    log10_fit1 = polyn1(log10(dx[5:]))
    #plot fit1
    ax.loglog(dx[5:], 10**log10_fit1, '-r')
    ax.text(dx[-2] + 0.005 , 10**(0.1 + log10_fit1[2])  , plot_text.format(s = log_slope1), horizontalalignment='left', c = 'r')

    ## SECOND SEGMENT FIT

    log_slope2, log_intercept2 = np.polyfit(log10(dx[0:5]), log10(er[0:5]),1)
    coeffs2 = np.polyfit(log10(dx[0:5]), log10(er[0:5]),1)
    polyn2 = np.poly1d(coeffs2)
    log10_fit2 = polyn2(log10(dx[0:5]))
    #plot fit2
    ax.loglog(dx[0:5], 10**log10_fit2, '-k')
    ax.text(dx[2]  , 10**(0.1 + log10_fit2[2])  , plot_text.format(s = log_slope2), horizontalalignment='left', c = 'k')

    #check slopes
    #print(str(log_slope1))
    #print(str(log_slope2))
 
    ax.set(xlabel = 'log(dx)', ylabel = 'log(RMS($u^2$))', title = 'RMS of $u^2$ as a function of dx')
    plt.show()


### TWO SOLITONS ###
two_sol = False
if two_sol == True:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #time step
    dt = 0.01
    #time steps
    nsteps = 10000

    ic = soliton(0.7, x, length/2) + soliton(0.25, x, length*(1.1))

    u = KDV_INT(N, length, dt, nsteps, ic)

    big_pos = zeros(nsteps)
    small_pos = zeros(nsteps)

    for i in range(nsteps):
        ut = u[i,:]
        midx = argrelextrema(np.real(ut),np.greater)[0]
        my = np.real(ut[midx])

        midx = midx[my > 0.01]
        my = my[my > 0.01]

        if len(my) == 2:
            if my[0] > my[1]:
                bid = midx[0]
                sid = midx[1]
            else:
                bid = midx[1]
                sid = midx[0]
        else:
            bid = midx[0]
            sid = midx[0]
        big_pos[i] = bid
        small_pos[i] = sid
    

    time_vals = np.arange(nsteps)*dt

    

    #plt.plot(time_vals, dx*small_pos, c = 'k', LineWidth = 1)
    #plt.plot(time_vals, dx*big_pos, c = 'b', LineWidth = 1)
    sample = 100
    plt.scatter(time_vals[0::sample], dx*small_pos[0::sample], s = 15, c = 'k',marker = 'o', label = 'small peak')
    plt.scatter(time_vals[0::sample], dx*big_pos[0::sample], s = 25, c = 'b',marker = '*', label = 'big peak')

    plt.ylim([0, length])
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc = 'lower right')
    plt.title('Position of peaks as time evolves')
    plt.show()
        

    #evo_plot(x,u,nsteps)
    #print(ic[argrelextrema(ic,np.greater)[0]])


### Kuramoto-Sivashinsky ###
Ku_Si = False
if Ku_Si:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #time step
    dt = 0.05
    #time steps
    nsteps = 10000*2

    #initial condition
    ic = 0.1*np.sin(2*np.pi/length*x)

    #beta
    beta = 0
    #array of alpha to analyse
    alpha = linspace(0.1, 2, 5)
    #nu to set first k* to be 2pi/L
    nu = 0.1*(length**2/8/pi**2)

    #plot initial condition
    plt.plot(x, ic, label = 'Initial')

    #integrate for each alpha
    for i in range(len(alpha)):
        u = KDV_INT_2(N,length, dt, nsteps, ic, beta, alpha[i], nu)
        lab = r'$\alpha$ = {a:.2f} $k^*$ = {k:.2f}'
        plt.plot(x, u[-1,:], label = lab.format(a = alpha[i], k = sqrt(alpha[i]/2/nu)))
    plt.legend(loc = 'upper left')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Long time solution of Kuramoto-Sivashinsky PDE with Small sine perturbation IC')
    plt.show()
    

### Kawahara-Toh ###
Ka_To = False
if Ka_To:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #time step
    dt = 0.05
    #time steps
    nsteps = 10000*10

    #initial condition
    ic = 0.1*np.sin(2*2*np.pi/length*x)

    #array of beta to analyse
    beta = linspace(0.3, 4, 5)
    #alpha
    alpha = 0.1
    #nu to set k* to 20pi/L
    nu = alpha*(length**2/8/pi**2)/10**2
    #plot initial condition
    plt.plot(x, ic, label = 'Initial')

    #integrate for each beta
    for i in range(len(beta)):
        u = KDV_INT_2(N,length, dt, nsteps, ic, beta[i], alpha, nu)
        lab = r'$\beta$ = {a:.2f} $k^*$ = {k:.2f}'
        plt.plot(x, u[-1,:], label = lab.format(a = beta[i], k = sqrt(alpha/2/nu)))
    plt.legend(loc = 'upper left')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Long time solution of Kawahara-Toh PDE with Small sine perturbation IC')
    plt.show()