import matplotlib.pyplot as plt 
import numpy as np
import scipy.integrate as sciInt

lam = 1

kap = 2*np.pi / lam

a = 3

L = a*lam

Nx = 100000

x = np.linspace(0,L, Nx)

u = lambda X : 2*np.sin(kap*X) + np.cos(2*kap*X) - 3*np.cos(7*kap*X)
u2 = lambda X : (2*np.sin(kap*X) + np.cos(2*kap*X) - 3*np.cos(7*kap*X))**2

amp = np.fft.fft(u(x))

# plt.plot(amp)
# plt.show()

freqs = np.fft.fftfreq(Nx, d = L/Nx)


# plt.plot(freqs,np.abs(amp)/(Nx))

# plt.xlim([-10, 10])

# plt.show()

s1 = sum(np.abs(amp/(Nx))**2)
sss = sciInt.quad(u2, 0 ,lam)
s2 = sss[0]
#print(sss)
print(s1-s2)
#print(s2)

x2 = np.linspace(-0.5, 0.5, Nx)

amp = np.fft.fft(u(x2))

freqs = np.fft.fftfreq(Nx, d = 1/Nx)

# plt.plot(freqs,np.abs(amp)/(Nx))
# plt.xlim([-10, 10])

# plt.show()

u3 = lambda X : np.exp(-2*(X - 0)**2)

x3 = np.linspace(-10,10,Nx)

amp3 = np.fft.fft(u3(x3))

freqs3 = np.fft.fftfreq(Nx, d = x3[1] - x3[0])

#plt.plot(freqs3,np.abs(amp3)/(Nx))
plt.xlim([-10, 10])

iffreq = np.fft.ifft(amp3)

plt.plot(x3, iffreq)

plt.xlim([-3, 3])

plt.show()

