import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as sciInt
 
#grid points
N = 1000

L = 2*np.pi

x = np.linspace(0,L,N+1)

del_x = x[1]-x[0]

lambda_min = 2*(x[1]-x[0])

k_max = 2*np.pi/lambda_min

del_k = 2*np.pi/L

k = np.arange(-k_max, k_max+del_k, del_k)

#print((k))
a= 2
b = 1
initial_space = a*np.exp(-b*(x - L/2)**2)
initial_space_deriv = -2*a*b*(x-L/2)*np.exp(-b*(x-L/2)**2)
initial_space_deriv_2 = a*b*(4*b*x**2-4*b*L*x+b*L**2-2)*np.exp(-b*(x-L/2)**2)
non_lin = initial_space*initial_space_deriv
# plt.plot(x,initial_space)
# plt.plot(x,initial_space_deriv)
# plt.plot(x,non_lin)
# plt.show()
#initial_F_space = np.zeros(len(k))

#l = np.linspace(-N/2, N/2,len(k))

# for i in range(len(l)):
#     temp_sum = 0
#     for J in range(len(initial_space)):
#         temp_sum += initial_space[J]*np.exp(-2*np.pi*1j*l[i]*J/N)
#     initial_F_space[i] = np.abs(del_x * temp_sum)

#print(initial_F_space)
# plt.plot(x,initial_space)
# plt.show()
# # plt.plot(k, initial_F_space)
# # plt.xlim([0, 100])
# # plt.show()
# plt.plot(np.fft.fft(initial_space, n = len(k)))
# plt.show()
# plt.plot(np.fft.ifft(np.fft.fft(initial_space, n = len(k))))
# plt.show()

## Generate linear derivatives from spectral modes

def spectral_deriv(u, N, k, dx):

    As = np.fft.fft(u, n = len(k))
    out_vec = np.zeros(len(As),dtype=np.complex_)
    for i in range(len(out_vec)):
        out_vec[i] = As[i]*((-1j*np.pi*k[i]/len(k))**N)*np.exp(-2*np.pi*1j*i/len(k))
    return out_vec

## Generate non-linear term

def N_bash(U, k, dx):
    u = np.fft.fft(U, n = len(k))
    out_vec = np.zeros(len(k),dtype=np.complex_)
    for K in range(len(k)):
        temp_sum = 0 + 0j
        for l in range(len(k)):
            temp_sum += u[K-l]*u[l]
        out_vec[K] = 1j*K/(dx*len(k))*temp_sum
    return out_vec

def L_calc(u,k):
    return -1*spectral_deriv(u, 3, k)


u_f_test = np.fft.fft(initial_space, n = len(k))
u_deriv_f_test = np.fft.fft(initial_space_deriv, n = len(k))
u_deriv_f_test_2 = np.fft.fft(initial_space_deriv_2, n = len(k))
non_lin_test = np.fft.fft(non_lin, n = len(k))

del_t = 0.001

# print(np.abs(spectral_deriv(initial_space, 0, k))-np.abs(u_f_test))
# print(np.abs(spectral_deriv(initial_space, 1, k))-np.abs(u_deriv_f_test))
# print(np.abs(spectral_deriv(initial_space, 2, k))-np.abs(u_deriv_f_test_2))
#print(np.abs(N_bash(non_lin, k, del_x)-np.abs(non_lin_test)))
fig, ax = plt.subplots()
plot1, = ax.plot(x, initial_space)
plot2, = ax.plot(x, 1*np.real(np.fft.ifft(spectral_deriv(initial_space, 0, k, del_x))))
plt.legend((plot1, plot2),('actual gauss', 'our ifft gauss'))
plt.show()
fig, ax = plt.subplots()
plot1, = ax.plot(x, initial_space_deriv)
plot2, = ax.plot(x, np.real(np.fft.ifft(spectral_deriv(initial_space, 1, k, del_x))))
plt.legend((plot1, plot2),('actual 1st deriv', 'our ifft first deriv'))
plt.show()
# plt.plot(x,non_lin)
# plt.plot(x, np.fft.ifft(N_bash(non_lin, k, del_x), n = len(k))/1000)
# plt.show()
#print(N_bash(u_f_test, k, del_x))

# nsteps = 3

# U = np.zeros((nsteps, len(k)))

# U[0,:] = initial_space

# U[1,:] = U[0,:]

# for i in range(2, nsteps):
#     temp_L = L_calc(U[i-1,:],k)
#     temp_N0 = N_bash(U[i-1,:], k, del_x)
#     temp_N1 = N_bash(U[i-2,:], k, del_x)
#     U_temp = np.zeros(len(k),dtype=np.complex_)
#     for K in range(len(k)):
#         #print((1/(1-del_t/2*temp_L[K]))*( (1 + del_t/2*temp_L[K])*U[i-1,K] + 3/2*del_t*temp_N0[K] - 0.5*del_t*temp_N1[K]))
#         U_temp[K] = ((1/(1-del_t/2*temp_L[K]))*( (1 + del_t/2*temp_L[K])*U[i-1,K] + 3/2*del_t*temp_N0[K] - 0.5*del_t*temp_N1[K]))
#     U[i,:] = np.abs(np.fft.ifft(U_temp))


# #print(U)







# from matplotlib.animation import FuncAnimation
# import time

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

# def animate(i):
#     if(i%1==0):
#         ax1.clear()
#         ax1.plot(x,np.abs((U[i,:])))


# ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1)
# plt.show()