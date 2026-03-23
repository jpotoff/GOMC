import numpy as np
# Wait. For C2R on N grid points, 
N = 8
Q = np.random.rand(N)
dQ = np.random.rand(N) - 0.5
S = np.fft.rfft(Q)
dS = np.fft.rfft(dQ)

G = np.random.rand(len(S))

weight = np.ones(len(S)) * 2.0
weight[0] = 1.0
weight[-1] = 1.0

# Initial E
E_init = 0.5 * np.sum(weight * G * np.abs(S)**2)

# New E
S_new = S + dS
E_new = 0.5 * np.sum(weight * G * np.abs(S_new)**2)

dE = E_new - E_init

# DeltaE = 0.5 * sum G |dS|^2 + sum G Re(S dS*)
dE_self = 0.5 * np.sum(weight * G * np.abs(dS)**2)
dE_cross = np.sum(weight * G * np.real(S * np.conj(dS)))

print("true dE:", dE)
print("computed parts:", dE_self + dE_cross)

# using iFFT for cross:
# We know V = irfft(G * S) * N
H = G * S
V = np.fft.irfft(H) * N

cross_from_v = np.sum(dQ * V)
print("cross from V:", cross_from_v)
print("dE_cross:", dE_cross)
