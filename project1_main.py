import matplotlib.pyplot as plt 
import numpy as np
from math import pi
from numpy.fft import fft, fftfreq, ifft
from numpy import cosh, exp,zeros, log10, linspace, sqrt
from Project1.KDV_FUNCS import KDV_INT,KDV_INT_2, quad_interp
from scipy.signal import argrelextrema
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams.update({'font.size': 18})
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
    plt.rcParams["figure.figsize"] = (16,10)
    plt.rcParams.update({'font.size': 12})
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
    nsteps_ar = [0, 100, 400, 1000]
    #values of A to analyse
    A_ = 0.15*np.arange(5) + 0.05
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
        ax.set_title('Gaussian Solution after time = ' + str(nsteps*dt))
        ax.set(xlabel = 'x', ylabel = 'u')
        ax.set_ylim([-0.1, 1])
        ax.legend(loc="upper left")
    plt.suptitle('Gaussian Long Time Dependance on initial amplitude')
    plt.show()
    


### ACCURACY ###

## DT ##
dt_ac = False 
if dt_ac:
    plt.rcParams["figure.figsize"] = (9,9)
    plt.rcParams.update({'font.size': 16})
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
    plt.rcParams["figure.figsize"] = (9,9)
    plt.rcParams.update({'font.size': 16})
    #number of grid points
    N = np.round(np.logspace(2,5,10))
    #length of domain
    length = 100
    #grid spacing
    dx = length/N
    #time step
    dt = 0.05
    #time steps
    nsteps = 1000
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
    #ax.loglog(dx, (dx-0.1)**(-2))

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
    nsteps = int(np.round(10000/1.2))

    ic = soliton(0.55, x, length/2) + soliton(0.18, x, length*(0.9))

    #initial plot
    plt.plot(x, ic)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim([0, 0.3])
    plt.title('Initial 2 Soliton Condition, c = 0.55, 0.18')
    plt.show()

    u = KDV_INT(N, length, dt, nsteps, ic)

    #merge plots
    plt.rcParams["figure.figsize"] = (14,10)
    plt.rcParams.update({'font.size': 10})
    times = (np.round(linspace(20, 60, 8)))
    for i in range(len(times)): 
        step = int(times[i]/dt)
        ax = plt.subplot(4,2,i+1)
        ax.plot(x, u[step, :])#, label = 't = '+str(times[i]))
        ax.set_xlabel('x')
        ax.set_title('t = '+str(times[i]))
        ax.set_ylabel('u')
        ax.set_ylim([0, 0.3])
        #plt.legend(loc = 'upper left')
    plt.suptitle('Merging of two soliton solutions')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    plt.show()

    plt.rcParams["figure.figsize"] = (8,8)
    plt.rcParams.update({'font.size': 18})

    #evo_plot(x,u,nsteps)

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
        big_pos[i] = quad_interp(ut,bid)
        small_pos[i] = quad_interp(ut,sid)
    

    time_vals = np.arange(nsteps)*dt
    id_same = small_pos == big_pos
    joined = big_pos[id_same]
    joined_time = time_vals[id_same ]
    
    small_pos[id_same ] = np.nan
    big_pos[id_same ] = np.nan
    
    plt.plot(time_vals, dx*small_pos, c = 'k', LineWidth = 1.5,label = 'small peak')
    plt.plot(time_vals, dx*big_pos, c = 'b', LineWidth = 1.5,label = 'big peak')
    plt.plot(joined_time, dx*joined, c = 'r', LineWidth = 1.8,label = 'joined')
    sample = 75
    #plt.scatter(time_vals[0::sample], dx*small_pos[0::sample], s = 15, c = 'k',marker = 'o', label = 'small peak')
    #plt.scatter(time_vals[0::sample], dx*big_pos[0::sample], s = 25, c = 'b',marker = '*', label = 'big peak')

    plt.ylim([0, length])
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc = 'lower right')
    plt.title('Position of peaks as time evolves')
    plt.show()

    # Plot predictions

    coeffs_s = np.polyfit(time_vals[0:100], small_pos[0:100],1)
    polyn_s = np.poly1d(coeffs_s)
    fit_s = polyn_s(time_vals)
    
    coeffs_b = np.polyfit(time_vals[0:100], big_pos[0:100],1)
    polyn_b = np.poly1d(coeffs_b)
    fit_b = polyn_b(time_vals)

    plt.plot(time_vals, dx*small_pos, c = 'k', LineWidth = 1.5,label = 'small peak')
    plt.plot(time_vals, dx*big_pos, c = 'b', LineWidth = 1.5,label = 'big peak')
    plt.plot(joined_time, dx*joined, c = 'r', LineWidth = 1.8,label = 'joined')
    plt.plot(time_vals, fit_s*dx,  '--k', label = 'small peak prediction')
    plt.plot(time_vals, fit_b*dx,  '--b',label = 'large peak prediction')
    sample = 75
    #plt.scatter(time_vals[0::sample], dx*small_pos[0::sample], s = 15, c = 'k',marker = 'o', label = 'small peak')
    #plt.scatter(time_vals[0::sample], dx*big_pos[0::sample], s = 25, c = 'b',marker = '*', label = 'big peak')

    plt.ylim([0, length])
    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend(loc = 'lower right')
    plt.title('Linear Peak Predictions')
    plt.show()

    #evo_plot(x,u,nsteps)
    #print(ic[argrelextrema(ic,np.greater)[0]])


### Kuramoto-Sivashinsky ###
Ku_Si = False
if Ku_Si:
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 12})
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
    nsteps = 10000

    #initial condition
    ic = 0.1*np.sin(2*np.pi/length*x)
    #ic = gaussian(0.1, 0.1, x, length)
    ic = soliton(0.5, x, length)

    #beta
    beta = 0
    
    i_alpha = 0.1
    # alpha = linspace(0.5, 2*pi/0.5, 5)
    #nu to set first k* to be 2pi/L
    nu = i_alpha*(length**2/8/pi**2)
    #array of alpha to analyse
    alpha = 2*nu*(np.arange(1,4.5,0.5)*2*pi/length)**2
    

    #plot initial condition
    ax = plt.subplot(4,2,1)

    ax.plot(x, ic, label = 'Initial')
    ax.set(title = 'Initial condition', xlabel = 'x', ylabel ='u')

    #integrate for each alpha
    for i in range(len(alpha)):
        u = KDV_INT_2(N,length, dt, nsteps, ic, beta, alpha[i], nu)
        lab = r'$\alpha$ = {a:.2f}, $k^*$= {k:.1f}($2\pi/L$)'
        ax = plt.subplot(4,2,i+2)
        ax.plot(x, u[-1,:])
        ax.set(title = lab.format(a = alpha[i], k = sqrt(alpha[i]/2/nu) * length/2/np.pi), xlabel = 'x', ylabel = 'u')
    tit = r'Soliton Solutions of Kuramoto-Sivashinsky PDE, t = {t:.1f}, $\nu$ = {nu_:.2f}'
    plt.suptitle(tit.format(t = nsteps*dt, nu_ = nu))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.5)
    plt.show()
  
    #Plotting chaotic divergence of sin perturbations
    chaos = False
    if chaos:
        ic1 = 0.01*np.sin(2*np.pi/length*x)
        ic2 = 0.02*np.sin(2*np.pi/length*x)
        ic3 = 0.03*np.sin(2*np.pi/length*x)
        ic4 = 0.04*np.sin(2*np.pi/length*x)
        u1 = KDV_INT_2(N,length, dt, nsteps, ic1, beta, alpha[-1], nu)
        u2 = KDV_INT_2(N,length, dt, nsteps, ic2, beta, alpha[-1], nu)
        u3 = KDV_INT_2(N,length, dt, nsteps, ic3, beta, alpha[-1], nu)
        u4 = KDV_INT_2(N,length, dt, nsteps, ic4, beta, alpha[-1], nu)

        fig, (ax1, ax2) = plt.subplots(2,1)

        ax1.plot(x,ic1, label = 'A = 0.01')
        ax1.plot(x,ic2, label = 'A = 0.02')
        ax1.plot(x,ic3, label = 'A = 0.03')
        ax1.plot(x,ic4, label = 'A = 0.04')
        ax1.set(title = 'Initial conditions', xlabel = 'x', label = 'u', ylim = [-0.3, 0.3])
        ax1.legend(loc = 'upper left')
        ax2.plot(x, u1[-1,:], label = 'A = 0.01')
        ax2.plot(x, u2[-1,:], label = 'A = 0.02')
        ax2.plot(x, u3[-1,:], label = 'A = 0.03')
        ax2.plot(x, u4[-1,:], label = 'A = 0.04')
        ax2.set(title = 'Evolution after t = 500', xlabel = 'x', label = 'u', ylim = [-0.3, 0.3])
        ax2.legend(loc = 'upper left')
        plt.show()

    

### Kawahara-Toh ###
Ka_To = False
if Ka_To:
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 12})
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
    ic = 0.1*np.sin(2*2*np.pi/length*x)
    ic = soliton(0.2, x, length)

    #array of beta to analyse
    beta = linspace(0.3, 2, 7)
    #alpha
    alpha = .1
    #nu to set k* to 20pi/L
    nu = alpha*(length**2/8/pi**2)/10**2
    #plot initial condition
    #plt.plot(x, ic, label = 'Initial')

    ax = plt.subplot(4,2,1)

    ax.plot(x, ic, label = 'Initial')
    ax.set(title = 'Initial condition', xlabel = 'x', ylabel ='u')

    #integrate for each beta
    for i in range(len(beta)):
        u = KDV_INT_2(N,length, dt, nsteps, ic, beta[i], alpha, nu)
        lab = r'$\beta$ = {a:.2f} $k^*$= {k:.1f}($2\pi/L$)'
        ax = plt.subplot(4,2,i+2)
        ax.plot(x, u[-1,:])
        ax.set(title = lab.format(a = beta[i], k = sqrt(alpha/2/nu) * length/2/np.pi), xlabel = 'x', ylabel = 'u')
    tit = r'Soliton Solutions of Kawahara-Toh PDE, t = {t:.1f}, $\nu$ = {nu_:.2f}, $\alpha$ = {a:.2f}'
    plt.suptitle(tit.format(t = nsteps*dt, nu_ = nu, a = alpha))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
    plt.show()

    #time evo of sine beta = 0.3
    tm_evo_sin = False
    if tm_evo_sin:
        ic = 0.1*np.sin(2*2*np.pi/length*x)
        beta = 0.3
        u = KDV_INT_2(N,length, dt, nsteps, ic, beta, alpha, nu)
        times = [10000, 11000, 12000]
        for i in range(len(times)):
            plt.plot(x, u[times[i],:], label = 't = ' + str(times[i]*dt))
        plt.title('Evolution of sin for beta = 0.3')
        plt.legend(loc = 'upper left')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.show()


### TESTING

## UNIT OPERATOR TEST 
un_op_test = False
if un_op_test:
    #number of grid points
    N = 400 
    #length of domain
    length = 10
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 

    #wavenumber discretisation
    k = 2*pi*N/length*fftfreq(N) 

    # LINEAR TERM (U_xxx)
    L_= -(1j*k)**3

    # NON-LINEAR TERM (6UU_x)
    def N_(u):
        return -6*0.5*1j*k*fft(u**2)

    #initial condition
    ic = np.sin(2*np.pi/length*x)

    #analytical U_xxx
    L_anal = (8*pi**3*np.cos((2*pi*x)/length))/length**3

    #numerical U_xxx
    L_num = 1*ifft(L_*fft(ic))

    #compare plots
    plt.plot(x, L_anal,'r', LineWidth = 4, label = 'Analytical')
    plt.plot(x, L_num, '--k', LineWidth = 2, label = 'Numerical')
    plt.legend(loc = 'upper right')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('L: Numerical vs Analytical')
    plt.show()

    #analytical UU_x
    N_anal = -12*pi/length * np.cos((2*pi*x)/length)*np.sin((2*pi*x)/length)

    #numerical UU_x
    N_num = ifft(N_(ic))

    #compare plots
    plt.plot(x, N_anal,'r', LineWidth = 4, label = 'Analytical')
    plt.plot(x, N_num, '--k', LineWidth = 2, label = 'Numerical')
    plt.legend(loc = 'upper right')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('N: Numerical vs Analytical')
    plt.show()

## INTEGRATED SOLITON TEST
sol_test = False
if sol_test:
    #number of grid points
    N = 400 
    #length of domain
    length = 100
    #grid spacing
    dx = length/N 
    #spatial discretisation
    x = dx*np.arange(N) 
    #timestep 
    dt = 0.05
    #number of steps
    nsteps = 1001

    #first soliton
    ic1 = soliton(0.5, x, length/2)
    #second soliton
    ic2 = soliton(1, x, length/2)

    #integrate the two initial conditions
    u1 = KDV_INT(N, length, dt, nsteps, ic1)
    u2 = KDV_INT(N, length, dt, nsteps, ic2)

    #plot 1st ic
    plt.plot(x, ic1, label = 't = 0')
    times = [100, 300, 600, 1000]
    for t in times:
        plt.plot(x, u1[t,:], label = 't = ' + str(t*dt))
    plt.legend(loc = 'upper left')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.ylim([-0.05, 0.8])
    plt.title('Soliton evolution, c = 0.5')
    plt.show()

    #plot 2nd ic
    plt.plot(x, ic2, label = 't = 0')
    times = [100, 300, 600, 1000]
    for t in times:
        plt.plot(x, u2[t,:], label = 't = ' + str(t*dt))
    plt.legend(loc = 'upper left')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Soliton evolution, c = 1')
    plt.ylim([-0.05, 0.8])
    plt.show()