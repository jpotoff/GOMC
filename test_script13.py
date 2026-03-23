import numpy as np
N = 8
Q = np.random.rand(N)
S = np.fft.rfft(Q)
G = np.random.rand(len(S))

weight = np.ones(len(S)) * 2.0
weight[0] = 1.0
weight[-1] = 1.0

# 1. Total energy in reciprocal space
E_recip = 0.5 * np.sum(weight * G * np.abs(S)**2)

# 2. Total energy in real space via V
V = np.fft.irfft(G * S) * N

E_real = np.sum(Q * V)
print("E_recip:", E_recip)
print("E_real :", E_real)
