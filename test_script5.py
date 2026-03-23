import numpy as np

# Let's test the scales.
N = 64
Q_r = np.random.rand(N)
dQ_r = np.random.rand(N) - 0.5

S_ref = np.fft.rfft(Q_r)
dS = np.fft.rfft(dQ_r)

# G is scale 1/L. Let's make it small like PME.
G = np.exp(- np.arange(len(S_ref))**2 / 10.0 ) / 100.0

E_init = 0.5 * np.sum(G * np.abs(S_ref)**2) * 2.0  # approximate full sum 
E_self = 0.5 * np.sum(G * np.abs(dS)**2) * 2.0

V_m = G * S_ref
V_r = np.fft.irfft(V_m, n=N) * N
dE_back = np.sum(dQ_r * V_r)

S_new = S_ref + dS
E_new = 0.5 * np.sum(G * np.abs(S_new)**2) * 2.0

print(f"E_init: {E_init:.6f}")
print(f"E_new: {E_new:.6f}")
print(f"True dE: {E_new - E_init:.6f}")
print(f"dE_back + dE_self: {dE_back + E_self:.6f}")

# And if we divide by N?
V_r_norm = V_r / N
dE_back_norm = np.sum(dQ_r * V_r_norm)
print(f"dE with division by N: {dE_back_norm + E_self:.6f}")
