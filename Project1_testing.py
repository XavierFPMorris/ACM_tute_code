import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as sciInt
from scipy.fftpack import fft, ifft
 
#grid points
N = 300

L = 4*np.pi

x = np.linspace(0,L,N+1)

del_x = x[1]-x[0]

lambda_min = 2*(x[1]-x[0])

k_max = 2*np.pi/lambda_min

del_k = 2*np.pi/L

k = np.arange(-k_max, k_max+del_k, del_k)

#print((k))
a = 1
b = 4
initial_space = a*np.exp(-b*(x - L/2)**2)
initial_space_deriv = -2*a*b*(x-L/2)*np.exp(-b*(x-L/2)**2)
initial_space_deriv_2 = a*b*(4*b*x**2-4*b*L*x+b*L**2-2)*np.exp(-b*(x-L/2)**2)
initial_space_deriv_3 = -a*b**2*(2*x-L)*(4*b*x**2-4*b*L*x+b*L**2-6)*np.exp(-b*(x-L/2)**2)
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

## Generate linear derivatives from space input, fourier output 

def spectral_deriv(u, p, k):

    return (1j)**p * np.fft.ifftshift(k)**p * np.fft.fft(u, n = len(k))


## Generate non-linear term

# def N_bash(u, k, dx):
#     k_shift = np.fft.ifftshift(k)
#     U = np.fft.fft(u, n = len(k))
#     U_shift = np.fft.fftshift(U)
#     out = np.zeros(len(k),dtype=np.complex_)
#     for i in range(len(k)):
#         temp_sum = 0
#         for K in range(len(k)):
#             if i-K < 0: continue
#             temp_sum += U[i - K]*U[K]
#         out[i] = 1j*k_shift[i] *dx/len(k) *temp_sum
#     return out


def L_calc(u, k):
    return -1*spectral_deriv(u, 3, k)

def N_calc(u, k):
    return -6*0.5*spectral_deriv(u**2, 1, k)


u_f_test = np.fft.fft(initial_space, n = len(k))
u_deriv_f_test = np.fft.fft(initial_space_deriv, n = len(k))
u_deriv_f_test_2 = np.fft.fft(initial_space_deriv_2, n = len(k))
non_lin_test = np.fft.fft(non_lin, n = len(k))

del_t = 0.005

# print(np.abs(spectral_deriv(initial_space, 0, k))-np.abs(u_f_test))
# print(np.abs(spectral_deriv(initial_space, 1, k))-np.abs(u_deriv_f_test))
# print(np.abs(spectral_deriv(initial_space, 2, k))-np.abs(u_deriv_f_test_2))
#print(np.abs(N_bash(non_lin, k, del_x)-np.abs(non_lin_test)))


#### THIS WORKS ####

# fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2)

# plot1, = ax1.plot(x, initial_space, label = 'actual 0th')
# plot2, = ax1.plot(x, (np.real(np.fft.ifft(spectral_deriv(initial_space, 0, k)))), label = 'my 0th')
# ax1.legend(loc="upper right")

# plot3, = ax2.plot(x, initial_space_deriv, label = 'actual 1st')
# plot4, = ax2.plot(x, np.real(np.fft.ifft(spectral_deriv(initial_space, 1, k))), label = 'my 1st')
# ax2.legend(loc="upper right")

# plot5, = ax3.plot(x, initial_space_deriv_2, label = 'actual 2nd')
# plot6, = ax3.plot(x, np.real(np.fft.ifft(spectral_deriv(initial_space, 2, k))), label = 'my 2nd')
# ax3.legend(loc="upper right")

# plot7, = ax4.plot(x, initial_space_deriv_3, label = 'actual 3rd')
# plot8, = ax4.plot(x, np.real(np.fft.ifft(spectral_deriv(initial_space, 3, k))), label = 'my 3rd')
# ax4.legend(loc="upper right")

# plt.show()


#### THIS WORKS ####

# plt.plot(x, initial_space*initial_space_deriv)
# plt.plot(x, np.real(np.fft.ifft(0.5*spectral_deriv(initial_space**2, 1, k))))
# plt.show()

#print(N_bash(u_f_test, k, del_x))

nsteps = 10

U = np.zeros((nsteps, len(k)),dtype=np.complex_)

U[0,:] = initial_space

U[1,:] =  initial_space + np.fft.ifft(L_calc(U[0,:],k))*del_t + np.fft.ifft(N_calc(U[0,:],k))*del_t 

for i in range(2, nsteps):
    t_L = np.fft.ifft((L_calc(U[i-1,:],k)))
    t_N0 = np.fft.ifft((N_calc(U[i-1,:], k)))
    t_N1 = np.fft.ifft((N_calc(U[i-2,:], k)))
    #U_temp = np.zeros(len(k),dtype=np.complex_)
    U[i,:] = (np.reciprocal(1-del_t/2*t_L)*( (1 + del_t/2*t_L)*(U[i-1,:]) + 3/2*del_t*t_N0 - 1/2*del_t*t_N1))


# #print(U)







from matplotlib.animation import FuncAnimation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    ax1.clear()
    ax1.plot(x,np.real(U[i,:]))


ani = FuncAnimation(fig, animate, frames = list(range(0,nsteps)), interval = 1000)
plt.show()