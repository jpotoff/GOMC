import numpy as np

N = 8
Q_r = np.random.rand(N)
dQ_r = np.random.rand(N) - 0.5

S_ref = np.fft.fft(Q_r)
dS = np.fft.fft(dQ_r)

G = np.random.rand(N)

# Reference energy
E_ref = 0.5 * np.sum(G * np.abs(S_ref)**2)

# New energy
S_new = S_ref + dS
E_new = 0.5 * np.sum(G * np.abs(S_new)**2)

# True dE
true_dE = E_new - E_ref

# Using DeltaE formula
self_energy = 0.5 * np.sum(G * np.abs(dS)**2)
# Using unnormalized FFT
V_m = G * S_ref
V_r = np.fft.ifft(V_m) * N
background = np.sum(dQ_r * np.real(V_r))

calc_dE = background + self_energy

print(f"true_dE: {true_dE}")
print(f"calc_dE (unnormalized V): {calc_dE}")

# If we divide V_r by N:
V_r_normalized = V_r / N
calc_dE_norm = np.sum(dQ_r * np.real(V_r_normalized)) + self_energy
print(f"calc_dE (normalized V by N): {calc_dE_norm}")
